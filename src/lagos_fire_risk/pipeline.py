from __future__ import annotations
import argparse, io, json, logging, math, re, sys, time, zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path
import joblib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, rasterio, requests
import yaml
from shapely.geometry import Point
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data'/'raw'; PROCESSED=ROOT/'data'/'processed'; OUT=ROOT/'outputs'; FIG=OUT/'figures'; MAPS=OUT/'maps'; LOGS=OUT/'logs'
ATLAS_URL='https://zenodo.org/records/14065246/files/GeoTIFF_Qdeg_monthly_summaries.zip'
YEARS=range(2003,2017)
WEATHER_VARS='temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean'
NGA_BOUNDS=(2.5,4.0,15.0,14.0)
LAGOS_BOUNDS=(2.65,6.25,4.05,6.95)
BASE_FEATURES=['latitude','longitude','month_sin','month_cos','temp_c','rain_mm','rain_lag_mm','wind_kmh','humidity','dryness']
MODIS_FEATURES=['ndvi','evi','lst_day_c','lst_night_c']
FEATURES=BASE_FEATURES+MODIS_FEATURES

def setup_logging():
    for d in (RAW,PROCESSED,FIG,MAPS,LOGS): d.mkdir(parents=True,exist_ok=True)
    log=logging.getLogger('lagos_fire_risk'); log.setLevel(logging.INFO); log.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for h in (logging.FileHandler(LOGS/'pipeline.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    return log

def get(url,params=None,timeout=120,retries=4):
    for attempt in range(retries):
        try:
            r=requests.get(url,params=params,timeout=timeout,headers={'User-Agent':'lagos-wildfire-risk-capstone/1.0 research'})
            r.raise_for_status(); return r
        except requests.RequestException:
            if attempt+1==retries: raise
            time.sleep(2**attempt)

class HTTPRangeReader:
    """Seekable HTTP reader; ZipFile fetches only the archive ranges it needs."""
    def __init__(self,url):
        self.url=url; self.position=0; self.session=requests.Session(); self.block_size=1024*1024; self.cache=OrderedDict(); self.cache_limit=32
        response=self.session.head(url,allow_redirects=True,timeout=60,headers={'User-Agent':'lagos-wildfire-risk-capstone/1.0 research'})
        response.raise_for_status()
        self.size=int(response.headers.get('Content-Length','0'))
        if not self.size:
            raise OSError('The Global Fire Atlas host did not provide the archive size needed for remote range reads.')
        probe=self.session.get(url,headers={'Range':'bytes=0-0','User-Agent':'lagos-wildfire-risk-capstone/1.0 research'},timeout=60)
        if probe.status_code!=206 or not probe.headers.get('Content-Range','').startswith('bytes 0-0/'):
            raise OSError(f'The Global Fire Atlas host does not support partial reads (HTTP {probe.status_code}); refusing a full archive download.')
        self.bytes_read=len(probe.content); self.requests=1
        probe.close()
    def seek(self,offset,whence=io.SEEK_SET):
        if whence==io.SEEK_SET: position=offset
        elif whence==io.SEEK_CUR: position=self.position+offset
        elif whence==io.SEEK_END: position=self.size+offset
        else: raise ValueError('Invalid seek mode.')
        self.position=max(0,min(self.size,position)); return self.position
    def tell(self): return self.position
    def readable(self): return True
    def seekable(self): return True
    def _get_block(self,block_start):
        if block_start in self.cache:
            self.cache.move_to_end(block_start); return self.cache[block_start]
        block_end=min(self.size,block_start+self.block_size)-1
        response=self.session.get(self.url,headers={'Range':f'bytes={block_start}-{block_end}','User-Agent':'lagos-wildfire-risk-capstone/1.0 research'},timeout=120)
        if response.status_code!=206:
            raise OSError(f'Expected HTTP byte-range response 206, got {response.status_code}; refusing a full archive download.')
        expected=f'bytes {block_start}-{block_end}/'
        if not response.headers.get('Content-Range','').startswith(expected):
            raise OSError(f'Unexpected HTTP Content-Range: {response.headers.get("Content-Range", "missing")}')
        data=response.content
        if len(data)!=block_end-block_start+1: raise OSError(f'Incomplete HTTP range response for bytes {block_start}-{block_end}.')
        self.bytes_read+=len(data); self.requests+=1; self.cache[block_start]=data
        if len(self.cache)>self.cache_limit: self.cache.popitem(last=False)
        return data
    def read(self,size=-1):
        if size is None or size<0: size=self.size-self.position
        size=min(size,self.size-self.position)
        if size<=0: return b''
        remaining=size; pieces=[]
        while remaining:
            block_start=(self.position//self.block_size)*self.block_size
            block=self._get_block(block_start); within=self.position-block_start
            take=min(remaining,len(block)-within)
            if take<=0: break
            pieces.append(block[within:within+take]); self.position+=take; remaining-=take
        data=b''.join(pieces)
        if len(data)!=size: raise OSError(f'Incomplete cached HTTP range read: expected {size} bytes, received {len(data)}.')
        return data
    def close(self): self.session.close()

def atlas_month_rasters(log):
    """Yield only the selected monthly fire-number TIFF members."""
    archive=RAW/'global_fire_atlas_monthly.zip'
    if archive.exists():
        log.info('Using the existing local Global Fire Atlas archive and reading only selected monthly rasters.')
        with zipfile.ZipFile(archive) as zf:
            yield from _selected_atlas_members(zf,log)
        return
    reader=HTTPRangeReader(ATLAS_URL)
    with zipfile.ZipFile(reader) as zf:
        yield from _selected_atlas_members(zf,log)
        log.info('Retrieved %s bytes in %s HTTP range requests from the Global Fire Atlas archive (archive size: %s bytes).',reader.bytes_read,reader.requests,reader.size)

def _selected_atlas_members(zf,log):
    selected=[]
    for name in zf.namelist():
        if not name.lower().endswith('.tif') or 'fire_number' not in name.lower(): continue
        m=re.search(r'_(20\d{2})(\d{2})\.tif$',name,re.I)
        if not m: continue
        year,month=int(m.group(1)),int(m.group(2))
        if year in YEARS: selected.append((year,month,name))
    if len(selected)!=len(YEARS)*12:
        raise ValueError(f'Expected {len(YEARS)*12} Global Fire Atlas monthly fire-number rasters, found {len(selected)}.')
    log.info('Reading %s selected monthly rasters; each TIFF is opened in memory and only the Nigeria window is used.',len(selected))
    for year,month,name in sorted(selected): yield year,month,zf.read(name)

def load_state_polygons():
    path=RAW/'nigeria_state_polygons.json'
    if not path.exists():
        url='https://gist.githubusercontent.com/segebee/ff4a2f75be560462c369/raw/nigeria_state_polygons.json'
        path.write_bytes(get(url).content)
    from shapely.geometry import Polygon
    data=json.loads(path.read_text(encoding='utf-8'))
    polygons={name:Polygon([(lon,lat) for lat,lon in coords]) for name,coords in data.items() if len(coords)>=4}
    return polygons

def load_fire_grid(log):
    counts=defaultdict(float)
    polygons=load_state_polygons(); from shapely.ops import unary_union
    country=unary_union(list(polygons.values()))
    # Build the complete national 0.25-degree land grid first. Atlas nodata is zero, so zero must be represented as a negative label.
    land_cells=set()
    for lat in np.arange(4.125,14.0,.25):
        for lon in np.arange(2.625,15.0,.25):
            if country.covers(Point(float(lon),float(lat))):
                land_cells.add((round(float(lat),3),round(float(lon),3)))
    for year,month,raster_bytes in atlas_month_rasters(log):
        with rasterio.MemoryFile(raster_bytes) as mem, mem.open() as src:
            from rasterio.windows import from_bounds
            bounds=from_bounds(*NGA_BOUNDS,transform=src.transform).round_offsets().round_lengths()
            arr=src.read(1,window=bounds,masked=True); tr=src.window_transform(bounds)
            rows,cols=np.indices(arr.shape); xs,ys=rasterio.transform.xy(tr,rows,cols,offset='center')
            lon=np.asarray(xs).reshape(arr.shape); lat=np.asarray(ys).reshape(arr.shape); val=np.asarray(arr.filled(0),dtype=float)
            mask=(~np.ma.getmaskarray(arr))&(val>0)&(lon>=NGA_BOUNDS[0])&(lon<=NGA_BOUNDS[2])&(lat>=NGA_BOUNDS[1])&(lat<=NGA_BOUNDS[3])
            for y,x,v in zip(lat[mask],lon[mask],val[mask]):
                key=(round(math.floor(y/.25)*.25+.125,3),round(math.floor(x/.25)*.25+.125,3))
                if key in land_cells: counts[(key[0],key[1],year,month)]+=float(v)
    rows=[]
    for lat,lon in sorted(land_cells):
        for year in YEARS:
            for month in range(1,13): rows.append({'latitude':lat,'longitude':lon,'year':year,'month':month,'ignitions':counts.get((lat,lon,year,month),0.)})
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError('No valid Nigeria land-grid cells intersect the fire archive.')
    df['fire']=(df.ignitions>0).astype('uint8')
    log.info('Loaded %s native 0.25-degree grid-month records across %s national land cells; positives=%s (%.2f%%).',f'{len(df):,}',df[['latitude','longitude']].drop_duplicates().shape[0],int(df.fire.sum()),df.fire.mean()*100)
    return df

def fetch_weather(fire,log):
    path=PROCESSED/'era5_monthly.csv'
    if path.exists():
        cached=pd.read_csv(path)
        if {'climate_zone','year','month'}.issubset(cached.columns): log.info('Using cached regional ERA5 monthly features.'); return cached
    # Four anchor cells capture the broad north-south and coastal climate gradients without hundreds of API calls.
    anchors=[('southwest',5.75,3.5),('central',8.5,6.5),('northwest',11.5,6.5),('northeast',12.5,11.5)]
    results=[]
    for year in YEARS:
        params={'latitude':','.join(str(a[1]) for a in anchors),'longitude':','.join(str(a[2]) for a in anchors),'start_date':f'{year}-01-01','end_date':f'{year}-12-31','daily':WEATHER_VARS,'timezone':'UTC','wind_speed_unit':'kmh'}
        log.info('Fetching Open-Meteo ERA5 regional anchors for %d.',year)
        payload=get('https://archive-api.open-meteo.com/v1/archive',params=params,timeout=180).json()
        if isinstance(payload,dict): payload=[payload]
        if len(payload)!=len(anchors): raise ValueError(f'Expected {len(anchors)} weather locations for {year}; received {len(payload)}.')
        for item,anchor in zip(payload,anchors):
            daily=item.get('daily',{})
            if not daily or not daily.get('time'): continue
            d=pd.DataFrame(daily); d['date']=pd.to_datetime(d.pop('time'),errors='coerce'); d['month']=d.date.dt.month; d['year']=d.date.dt.year
            grouped=d.groupby(['year','month'],as_index=False).agg(temp_c=('temperature_2m_mean','mean'),rain_mm=('precipitation_sum','sum'),wind_kmh=('wind_speed_10m_max','mean'),humidity=('relative_humidity_2m_mean','mean'))
            grouped['climate_zone']=anchor[0]; results.append(grouped)
    if not results: raise ValueError('Open-Meteo returned no weather data.')
    weather=pd.concat(results,ignore_index=True); weather.to_csv(path,index=False); return weather

def build_features(fire,weather,log,use_modis=True):
    # Match each cell to its nearest regional climate anchor using a latitude-adjusted longitude distance.
    anchors={'southwest':(5.75,3.5),'central':(8.5,6.5),'northwest':(11.5,6.5),'northeast':(12.5,11.5)}
    cell=fire[['latitude','longitude']].drop_duplicates().copy()
    def closest(row):
        return min(anchors,key=lambda k:((row.latitude-anchors[k][0])**2+(math.cos(math.radians(row.latitude))*(row.longitude-anchors[k][1]))**2))
    cell['climate_zone']=cell.apply(closest,axis=1)
    df=fire.merge(cell,on=['latitude','longitude'],how='left',validate='many_to_one')
    df=df.merge(weather,on=['climate_zone','year','month'],how='left',validate='many_to_one')
    if df.temp_c.isna().mean()>.1: raise ValueError(f'Weather is missing for {df.temp_c.isna().mean():.1%} of records.')
    modis_path=PROCESSED/'modis_monthly.csv'
    global FEATURES
    if use_modis and not modis_path.exists():
        raise FileNotFoundError(f'MODIS feature table not found at {modis_path}. Run `python -m src.lagos_fire_risk.modis_export --project YOUR_EARTH_ENGINE_PROJECT_ID`, download the Drive CSV, and save it at this path.')
    if use_modis:
        modis=pd.read_csv(modis_path)
        modis_required={'latitude','longitude','year','month','ndvi','evi','lst_day_c','lst_night_c'}
        missing=modis_required-set(modis.columns)
        if missing: raise ValueError(f'MODIS feature table is missing columns: {sorted(missing)}')
        modis['latitude']=modis.latitude.round(3); modis['longitude']=modis.longitude.round(3)
        modis['year']=modis.year.astype(int); modis['month']=modis.month.astype(int)
        if modis.duplicated(['latitude','longitude','year','month']).any():
            raise ValueError('MODIS table has duplicate grid-cell/year/month rows.')
        df=df.merge(modis[list(modis_required)],on=['latitude','longitude','year','month'],how='left',validate='one_to_one')
        missing_modis=df[list(modis_required-{'latitude','longitude','year','month'})].isna().any(axis=1)
        if missing_modis.mean()>.1:
            raise ValueError(f'MODIS predictors are incomplete for {missing_modis.mean():.1%} of fire records; re-export the complete 2003–2016 table.')
        FEATURES=BASE_FEATURES+MODIS_FEATURES
        log.info('Using MODIS vegetation and land-surface-temperature features.')
    else:
        FEATURES=BASE_FEATURES
        log.warning('Training a real-data baseline without MODIS predictors; MODIS remains needed for the proposal-complete run.')
    df=df.sort_values(['latitude','longitude','year','month']).reset_index(drop=True)
    df['rain_lag_mm']=df.groupby(['latitude','longitude']).rain_mm.shift(1).fillna(df.rain_mm)
    df['month_sin']=np.sin(2*np.pi*df.month/12); df['month_cos']=np.cos(2*np.pi*df.month/12)
    df['dryness']=df.temp_c/(1+df.rain_lag_mm.clip(lower=0)); df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['fire'])
    df.to_csv(PROCESSED/'training_features.csv',index=False); log.info('Saved %s complete feature rows.',f'{len(df):,}'); return df

def _make_candidate(name, seed):
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=seed)
    if name == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(max_iter=200,learning_rate=.08,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.0,random_state=seed)
    raise ValueError(f'Unknown model candidate: {name}')

def spatial_cv(df,log,model_cfg=None):
    model_cfg=model_cfg or {}; block_size=float(model_cfg.get('spatial_block_degrees',2)); requested_folds=int(model_cfg.get('validation_folds',5))
    candidates=list(model_cfg.get('candidates',['RandomForest','HistGradientBoosting'])); seed=int(model_cfg.get('random_state',42))
    df=df.copy(); groups=df.longitude.floordiv(block_size).astype(str)+'_'+df.latitude.floordiv(block_size).astype(str)
    n=min(requested_folds,groups.nunique())
    if n<3 or df.fire.nunique()!=2: raise ValueError(f'Insufficient classes or spatial blocks for spatial validation (groups={groups.nunique()}, classes={df.fire.nunique()}).')
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan); y=df.fire.astype(int)
    if X.isna().any().any(): raise ValueError('Feature table contains missing values after MODIS/weather joins.')
    splitter=GroupKFold(n_splits=n); fold_rows=[]
    model_results={}
    for name in candidates:
        predictions=np.full(len(df),np.nan); rows=[]
        for fold,(tr,te) in enumerate(splitter.split(X,y,groups=groups),1):
            if y.iloc[tr].nunique()<2 or y.iloc[te].nunique()<2:
                log.warning('Skipping %s fold %d because it contains only one class.',name,fold); continue
            model=_make_candidate(name,seed)
            weights=compute_sample_weight(class_weight='balanced',y=y.iloc[tr])
            model.fit(X.iloc[tr],y.iloc[tr],sample_weight=weights)
            pred=model.predict_proba(X.iloc[te])[:,1]; predictions[te]=pred
            rows.append({'model':name,'fold':fold,'train_rows':len(tr),'test_rows':len(te),'roc_auc':roc_auc_score(y.iloc[te],pred),'average_precision':average_precision_score(y.iloc[te],pred),'brier':brier_score_loss(y.iloc[te],pred)})
        if not rows: raise ValueError(f'No spatial CV folds contained both classes for {name}.')
        fold_df=pd.DataFrame(rows); fold_rows.extend(rows)
        valid=np.isfinite(predictions)
        model_results[name]={'folds':int(len(fold_df)),'records':int(valid.sum()),'roc_auc_mean':float(fold_df.roc_auc.mean()),'roc_auc_sd':float(fold_df.roc_auc.std(ddof=1)) if len(fold_df)>1 else None,'average_precision_mean':float(fold_df.average_precision.mean()),'brier_mean':float(fold_df.brier.mean()),'threshold':0.5,'precision_at_0_5':float(precision_score(y[valid],predictions[valid]>=.5,zero_division=0)),'recall_at_0_5':float(recall_score(y[valid],predictions[valid]>=.5,zero_division=0)),'f1_at_0_5':float(f1_score(y[valid],predictions[valid]>=.5,zero_division=0)),'confusion_matrix':confusion_matrix(y[valid],predictions[valid]>=.5).tolist()}
    fold_table=pd.DataFrame(fold_rows); fold_table.to_csv(OUT/'spatial_validation_folds.csv',index=False)
    comparison=(fold_table.groupby('model',as_index=False).agg(roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),average_precision_mean=('average_precision','mean'),brier_mean=('brier','mean'),folds=('fold','count')))
    comparison.sort_values(['average_precision_mean','roc_auc_mean','brier_mean'],ascending=[False,False,True]).to_csv(OUT/'model_comparison.csv',index=False)
    best=sorted(model_results,key=lambda name:(-model_results[name]['average_precision_mean'],-model_results[name]['roc_auc_mean'],model_results[name]['brier_mean']))[0]
    metrics={'validation':f'GroupKFold by {block_size:g}-degree spatial blocks','selection_metric':'mean spatial-fold average precision; ties by ROC AUC then Brier score','best_model':best,'models':model_results,'positive_rate':float(y.mean()),**model_results[best]}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    log.info('Spatial CV selected %s: ROC AUC %.3f, average precision %.3f, Brier %.3f.',best,metrics['roc_auc_mean'],metrics['average_precision_mean'],metrics['brier_mean'])
    return metrics

def fit_and_explain(df,metrics,log,model_cfg=None):
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan); y=df.fire.astype(int)
    model_cfg=model_cfg or {}; name=metrics['best_model']; model=_make_candidate(name,int(model_cfg.get('random_state',42)))
    weights=compute_sample_weight(class_weight='balanced',y=y)
    model.fit(X,y,sample_weight=weights)
    sample_size=min(int(model_cfg.get('shap_sample_size',1500)),len(X)); explain_x=X.sample(n=sample_size,random_state=int(model_cfg.get('random_state',42)))
    background=X.sample(n=min(100,len(X)),random_state=21)
    y_explain=y.loc[explain_x.index]
    try:
        import shap
        if isinstance(model,RandomForestClassifier):
            explainer=shap.TreeExplainer(model,data=background,feature_perturbation='interventional',model_output='probability')
            values=explainer.shap_values(explain_x,check_additivity=False)
            if isinstance(values,list): values=values[-1]
            values=np.asarray(values)
            if values.ndim==3: values=values[:,:,1]
            values=np.asarray(values)
        else:
            explainer=shap.Explainer(model.predict_proba,background,algorithm='permutation')
            explanation=explainer(explain_x,max_evals=2*len(FEATURES)+1)
            values=np.asarray(explanation.values)
            if values.ndim==3: values=values[:,:,1]
        importance=pd.DataFrame({'feature':FEATURES,'importance':np.abs(values).mean(axis=0),'importance_std':np.abs(values).std(axis=0)})
        explanation_method='SHAP'
    except Exception as exc:
        from sklearn.inspection import permutation_importance
        log.warning('SHAP could not explain %s (%s); computing held-sample permutation importance instead.',name,type(exc).__name__)
        result=permutation_importance(model,explain_x,y_explain,n_repeats=3,random_state=int(model_cfg.get('random_state',42)),scoring='average_precision',n_jobs=-1)
        importance=pd.DataFrame({'feature':FEATURES,'importance':result.importances_mean,'importance_std':result.importances_std})
        explanation_method='Permutation importance (SHAP unavailable in this runtime)'
    importance['method']=explanation_method
    importance.sort_values('importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
    plt.figure(figsize=(10,7),dpi=180)
    plot_data=importance.sort_values('importance').tail(14)
    plt.barh(plot_data.feature,plot_data.importance,color='#188b91',alpha=.9)
    plt.xlabel('Mean absolute contribution' if explanation_method=='SHAP' else 'Average precision score decrease')
    plt.title(f'{explanation_method} global feature importance — {name}',loc='left',weight='bold')
    plt.tight_layout(); plt.savefig(FIG/'feature_importance.png',bbox_inches='tight',facecolor='white'); plt.close()
    metrics['explanation_method']=explanation_method
    metrics['features']=FEATURES
    metrics['feature_set']='ERA5, MODIS, location, and seasonal features' if len(FEATURES)>len(BASE_FEATURES) else 'ERA5, location, and seasonal features'
    metrics['positive_rate']=float(y.mean())
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    joblib.dump({'model':model,'features':FEATURES,'algorithm':name,'training_scope':'Nigeria-wide monthly fire occurrence, 2003-2016 at native 0.25-degree Atlas cells','selection_metric':'spatial cross-validation average precision','explanation_method':explanation_method},OUT/'wildfire_model.joblib')
    log.info('Selected %s from spatial cross-validation; leading predictors by %s: %s',name,explanation_method,', '.join(importance.sort_values('importance',ascending=False).feature.head(4)))
    return model

def lagos_boundary():
    try:
        from shapely.geometry import mapping
        polygons=load_state_polygons(); return mapping(polygons['LAGOS']),'Nigeria state polygon from segebee/logbon72 gist'
    except Exception: return None,'buffered Lagos-area extent'

def make_outputs(model,df,log):
    geometry,boundary_source=lagos_boundary()
    if not geometry: raise ValueError('Lagos boundary is required to generate a state-clipped output map.')
    from shapely.geometry import shape
    from shapely.geometry import mapping as shapely_mapping
    from shapely.geometry import box
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Polygon as PolygonPatch
    geom=shape(geometry)
    cell_coords=df[['latitude','longitude']].drop_duplicates()
    lagos_cells=cell_coords[[geom.covers(Point(float(row.longitude),float(row.latitude))) for row in cell_coords.itertuples()]]
    if lagos_cells.empty: raise ValueError('No native 0.25-degree training cell centers fall inside Lagos State.')
    local=df.merge(lagos_cells,on=['latitude','longitude'],how='inner',validate='many_to_one')
    monthly_features=[name for name in ['temp_c','rain_mm','rain_lag_mm','wind_kmh','humidity']+MODIS_FEATURES if name in local.columns]
    cell_monthly=local.groupby(['latitude','longitude','month'],as_index=False)[monthly_features].mean()
    cell_monthly['month_sin']=np.sin(2*np.pi*cell_monthly.month/12); cell_monthly['month_cos']=np.cos(2*np.pi*cell_monthly.month/12)
    cell_monthly['dryness']=cell_monthly.temp_c/(1+cell_monthly.rain_lag_mm.clip(lower=0))
    base=cell_monthly.dropna(subset=FEATURES).copy()
    if base.empty: raise ValueError('No complete Lagos monthly feature rows are available for prediction.')
    base['risk_probability']=model.predict_proba(base[FEATURES])[:,1]
    seasonal=base.groupby('month',as_index=False).risk_probability.mean()
    risk=base.groupby(['latitude','longitude'],as_index=False).risk_probability.mean()
    risk['relative_risk_percentile']=risk.risk_probability.rank(pct=True)*100
    risk['grid_resolution_degrees']=.25
    risk.to_csv(PROCESSED/'lagos_risk_grid.csv',index=False)
    geojson_features=[]
    for row in risk.itertuples():
        cell=box(row.longitude-.125,row.latitude-.125,row.longitude+.125,row.latitude+.125).intersection(geom)
        if cell.is_empty: continue
        geojson_features.append({'type':'Feature','geometry':shapely_mapping(cell),'properties':{'latitude':row.latitude,'longitude':row.longitude,'risk_score':row.risk_probability,'relative_percentile':row.relative_risk_percentile,'grid_resolution_degrees':.25}})
    (PROCESSED/'lagos_risk_grid.geojson').write_text(json.dumps({'type':'FeatureCollection','features':geojson_features}),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,7.5),dpi=180)
    norm=Normalize(vmin=0,vmax=100); cmap=plt.get_cmap('YlOrRd')
    for feature in geojson_features:
        cell=shape(feature['geometry']); color=cmap(norm(feature['properties']['relative_percentile']))
        polygons=[cell] if cell.geom_type=='Polygon' else list(cell.geoms)
        for polygon in polygons:
            if polygon.geom_type=='Polygon': ax.add_patch(PolygonPatch(np.asarray(polygon.exterior.coords),closed=True,facecolor=color,edgecolor='#243244',linewidth=.45))
    polys=[geom] if geom.geom_type=='Polygon' else list(geom.geoms)
    for poly in polys:
        x,y=poly.exterior.xy; ax.plot(x,y,color='#102033',linewidth=1.5)
    ax.set_xlim(LAGOS_BOUNDS[0],LAGOS_BOUNDS[2]); ax.set_ylim(LAGOS_BOUNDS[1],LAGOS_BOUNDS[3])
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_title('Lagos Wildfire Risk Screening\nNative 0.25° fire grid · 2003–2016 baseline',loc='left',fontsize=15,weight='bold',pad=12)
    data_note='fire observations, MODIS vegetation/temperature, and ERA5 weather' if len(FEATURES)>len(BASE_FEATURES) else 'fire observations and ERA5 weather (MODIS features not included in this run)'
    ax.text(.01,-.14,f'Cells are clipped to Lagos State. Colours rank the monthly mean model score within the displayed area.\nHistorical screening view from {data_note}.',transform=ax.transAxes,fontsize=8,color='#52616f')
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.78,pad=.03); cb.set_label('Relative risk percentile')
    ax.grid(color='#9aa7b2',alpha=.22); fig.tight_layout()
    fig.savefig(MAPS/'lagos_wildfire_risk.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),dpi=180)
    axes[0].plot(seasonal.month,seasonal.risk_probability,color='#e66b32',marker='o',lw=2.4); axes[0].set_xticks(range(1,13)); axes[0].set_xlabel('Month'); axes[0].set_ylabel('Mean model score'); axes[0].set_title('Lagos seasonal profile',weight='bold'); axes[0].grid(alpha=.25)
    importance=pd.read_csv(OUT/'feature_importance.csv').sort_values('importance')
    axes[1].barh(importance.feature,importance.importance,color='#188b91',alpha=.9); axes[1].set_title(importance.method.iloc[0]+' importance',weight='bold'); axes[1].set_xlabel('Mean absolute contribution' if importance.method.iloc[0]=='SHAP' else 'Average precision score decrease')
    fig.suptitle('Model explainability',x=.03,ha='left',fontsize=15,weight='bold'); fig.tight_layout(); fig.savefig(FIG/'model_explainability.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
    seasonal.to_csv(PROCESSED/'lagos_seasonal_profile.csv',index=False)
    log.info('Saved %d native 0.25-degree Lagos model cells clipped to state boundary; boundary source: %s.',len(risk),boundary_source)
    return risk,seasonal,boundary_source

def write_report(metrics,df,risk,seasonal,boundary_source):
    names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    sd=metrics['roc_auc_sd'] if metrics['roc_auc_sd'] is not None else 0
    lines=['# Lagos Wildfire Risk Screening','',
    '## Executive summary','',
    ('This project builds an interpretable spatial fire-occurrence screen clipped to Lagos State. It combines Global Fire Atlas fire observations with MODIS vegetation/thermal metrics and ERA5 reanalysis, trains on Nigeria-wide fire patterns, and evaluates generalisation by holding out spatial blocks.' if len(FEATURES)>len(BASE_FEATURES) else 'This baseline uses Global Fire Atlas fire observations and ERA5 reanalysis, trains on Nigeria-wide fire patterns, and evaluates generalisation by holding out spatial blocks. MODIS vegetation and thermal predictors remain to be added for the proposal-complete run.'),'',
    '**The Lagos map shows relative historical screening scores, not a real-time warning or a calibrated probability of an incident.** Satellite fire records can miss small burns and may include thermal anomalies. Local incident validation is required before operational use.','',
    '## Study design','',
    f'- Presentation area: Lagos State with nearby peri-urban context; boundary source: {boundary_source}.',
    '- Training region: Nigeria land cells within approximately 2.5–15°E and 4–14°N; only map cells whose centers fall within Lagos State are displayed.',
    '- Fire labels: Global Fire Atlas monthly fire-number rasters (2003–2016) at the native 0.25° modelling grid.',
    ('- Predictors: Open-Meteo ERA5 weather plus monthly MODIS NDVI, EVI, and day/night LST.' if len(FEATURES)>len(BASE_FEATURES) else '- Predictors: Open-Meteo ERA5 weather; MODIS predictors were not included in this baseline run.'),
    '- Validation: GroupKFold spatial cross-validation by 2° blocks; all candidate models use the same held-out folds.',
    f'- Selected model: {metrics["best_model"]}, chosen by mean spatial-fold average precision; {metrics.get("explanation_method", "model explanation pending")} summarizes fitted feature contributions.','',
    '## Validation results','',
    f'- Training records: {len(df):,}; fire-positive grid-month prevalence: {df.fire.mean():.2%}.',
    f'- Spatial ROC AUC: {metrics["roc_auc_mean"]:.3f} (fold SD {sd:.3f}).',
    f'- Spatial average precision: {metrics["average_precision_mean"]:.3f}; Brier score: {metrics["brier_mean"]:.3f}.',
    f'- At fixed threshold 0.50: precision {metrics["precision_at_0_5"]:.3f}, recall {metrics["recall_at_0_5"]:.3f}, F1 {metrics["f1_at_0_5"]:.3f}.',
    '- Read ROC AUC alongside average precision and Brier because fire months are rare; accuracy alone is not informative.','',
    '## Reading the Lagos map','',
    f'The map ranks {len(risk)} native 0.25° model cells by mean model score over calendar months. Each cell polygon is clipped to the Lagos State boundary, and percentiles are relative to the displayed cells. The surface represents a historical fire/climate baseline, not a future-day forecast.','',
    '### Seasonal profile','', '| Month | Mean model probability |','|---|---:|']
    for _,row in seasonal.iterrows(): lines.append(f'| {names[int(row.month)-1]} | {row.risk_probability:.4f} |')
    lines += ['', '## Model comparison','', '| Candidate | Spatial ROC AUC | Average precision | Brier score |','|---|---:|---:|---:|']
    for model_name, model_result in metrics['models'].items():
        lines.append(f"| {model_name} | {model_result['roc_auc_mean']:.3f} | {model_result['average_precision_mean']:.3f} | {model_result['brier_mean']:.3f} |")
    lines += ['', '## Sources','',
    '- Global Fire Atlas updated record (2002–2024): https://zenodo.org/records/14065246',
    '- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api',
    '- MODIS MOD13Q1 vegetation indices (NDVI/EVI): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1',
    '- MODIS MOD11A2 land surface temperature: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2',
    '- GeoBoundaries open boundaries: https://gist.github.com/segebee/ff4a2f75be560462c369','',
    '## Reproducibility','',
    'Run python run.py from the project root. Raw inputs, processed tables, model, figures, maps, metrics, report, and logs are stored under this project. The data archive is excluded from Git; its source URL and retrieval logic are recorded in the code and README.','']
    (ROOT/'reports'/'project_report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description='Train and map the Lagos wildfire screening model.')
    parser.add_argument('--without-modis',action='store_true',help='Train a real-data fire-and-weather baseline while MODIS Earth Engine features are unavailable.')
    parser.add_argument('--resume-features',action='store_true',help='Reuse the saved training table and spatial-CV metrics; skip data retrieval and model comparison.')
    args=parser.parse_args()
    log=setup_logging(); log.info('Starting Lagos wildfire risk screening.')
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))
    if args.resume_features:
        if not args.without_modis: raise ValueError('--resume-features currently requires --without-modis so feature columns are explicit.')
        global FEATURES
        FEATURES=BASE_FEATURES
        df=pd.read_csv(PROCESSED/'training_features.csv')
        metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        log.info('Resuming with %s saved training rows and saved spatial-CV model selection.',f'{len(df):,}')
    else:
        fire=load_fire_grid(log); weather=fetch_weather(fire,log); df=build_features(fire,weather,log,use_modis=not args.without_modis)
        metrics=spatial_cv(df,log,cfg.get('model',{}))
    model=fit_and_explain(df,metrics,log,cfg.get('model',{})); risk,seasonal,boundary=make_outputs(model,df,log)
    write_report(metrics,df,risk,seasonal,boundary)
    log.info('Pipeline complete. Outputs are in %s.',OUT)

if __name__=='__main__': main()
