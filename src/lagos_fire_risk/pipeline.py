from __future__ import annotations
import json, logging, math, re, sys, time, zipfile
from collections import defaultdict
from pathlib import Path
import joblib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, rasterio, requests
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data'/'raw'; PROCESSED=ROOT/'data'/'processed'; OUT=ROOT/'outputs'; FIG=OUT/'figures'; MAPS=OUT/'maps'; LOGS=OUT/'logs'
ATLAS_URL='https://zenodo.org/records/14065246/files/GeoTIFF_Qdeg_monthly_summaries.zip'
YEARS=range(2003,2017)
WEATHER_VARS='temperature_2m_mean,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean'
NGA_BOUNDS=(2.5,4.0,15.0,14.0)
LAGOS_BOUNDS=(2.65,6.25,4.05,6.95)
FEATURES=['latitude','longitude','month_sin','month_cos','temp_c','rain_mm','rain_lag_mm','wind_kmh','dryness']

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

def ensure_atlas(log):
    archive=RAW/'global_fire_atlas_monthly.zip'
    if not archive.exists() or archive.stat().st_size<1_000_000:
        log.info('Downloading Global Fire Atlas monthly summaries from Zenodo.')
        tmp=archive.with_suffix('.part')
        with requests.get(ATLAS_URL,stream=True,timeout=120,headers={'User-Agent':'lagos-wildfire-risk-capstone/1.0'}) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk: f.write(chunk)
        tmp.replace(archive)
    if not zipfile.is_zipfile(archive): raise ValueError('Global Fire Atlas archive is not a valid zip.')
    return archive

def atlas_month_rasters(archive):
    selected=[]
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.tif') or 'fire_number' not in name.lower(): continue
            m=re.search(r'_(20\d{2})(\d{2})\.tif$',name,re.I)
            if not m: continue
            year,month=int(m.group(1)),int(m.group(2))
            if year not in YEARS: continue
            selected.append((year,month,name))
        if not selected: raise ValueError('No 2003–2016 fire_number monthly rasters found in the Atlas archive.')
        extracted=RAW/'atlas_fire_number'; extracted.mkdir(exist_ok=True)
        paths=[]
        for year,month,name in sorted(selected):
            dst=extracted/Path(name).name
            if not dst.exists() or dst.stat().st_size==0:
                with zf.open(name) as src,dst.open('wb') as out: out.write(src.read())
            paths.append((year,month,dst))
    return paths

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
    files=atlas_month_rasters(ensure_atlas(log)); counts=defaultdict(float)
    polygons=load_state_polygons(); from shapely.ops import unary_union
    country=unary_union(list(polygons.values()))
    # Build the complete national 0.5-degree land grid first. Atlas nodata is zero, so zero must be represented as a negative label.
    land_cells=set()
    for lat in np.arange(4.25,14.01,.5):
        for lon in np.arange(2.75,15.01,.5):
            if country.covers(Point(float(lon),float(lat))):
                land_cells.add((round(float(lat),3),round(float(lon),3)))
    for year,month,path in files:
        log.info('Reading fire atlas %s',path.name)
        with rasterio.open(path) as src:
            from rasterio.windows import from_bounds
            bounds=from_bounds(*NGA_BOUNDS,transform=src.transform).round_offsets().round_lengths()
            arr=src.read(1,window=bounds,masked=True); tr=src.window_transform(bounds)
            rows,cols=np.indices(arr.shape); xs,ys=rasterio.transform.xy(tr,rows,cols,offset='center')
            lon=np.asarray(xs).reshape(arr.shape); lat=np.asarray(ys).reshape(arr.shape); val=np.asarray(arr.filled(0),dtype=float)
            mask=(~np.ma.getmaskarray(arr))&(val>0)&(lon>=NGA_BOUNDS[0])&(lon<=NGA_BOUNDS[2])&(lat>=NGA_BOUNDS[1])&(lat<=NGA_BOUNDS[3])
            for y,x,v in zip(lat[mask],lon[mask],val[mask]):
                key=(round(math.floor(y/.5)*.5+.25,3),round(math.floor(x/.5)*.5+.25,3))
                if key in land_cells: counts[(key[0],key[1],year,month)]+=float(v)
    rows=[]
    for lat,lon in sorted(land_cells):
        for year in YEARS:
            for month in range(1,13): rows.append({'latitude':lat,'longitude':lon,'year':year,'month':month,'ignitions':counts.get((lat,lon,year,month),0.)})
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError('No valid Nigeria land-grid cells intersect the fire archive.')
    df['fire']=(df.ignitions>0).astype('uint8')
    log.info('Loaded %s grid-month records across %s national land cells; positives=%s (%.2f%%).',f'{len(df):,}',df[['latitude','longitude']].drop_duplicates().shape[0],int(df.fire.sum()),df.fire.mean()*100)
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

def build_features(fire,weather,log):
    # Match each cell to its nearest regional climate anchor using a latitude-adjusted longitude distance.
    anchors={'southwest':(5.75,3.5),'central':(8.5,6.5),'northwest':(11.5,6.5),'northeast':(12.5,11.5)}
    cell=fire[['latitude','longitude']].drop_duplicates().copy()
    def closest(row):
        return min(anchors,key=lambda k:((row.latitude-anchors[k][0])**2+(math.cos(math.radians(row.latitude))*(row.longitude-anchors[k][1]))**2))
    cell['climate_zone']=cell.apply(closest,axis=1)
    df=fire.merge(cell,on=['latitude','longitude'],how='left',validate='many_to_one')
    df=df.merge(weather,on=['climate_zone','year','month'],how='left',validate='many_to_one')
    if df.temp_c.isna().mean()>.1: raise ValueError(f'Weather is missing for {df.temp_c.isna().mean():.1%} of records.')
    df=df.sort_values(['latitude','longitude','year','month']).reset_index(drop=True)
    df['rain_lag_mm']=df.groupby(['latitude','longitude']).rain_mm.shift(1).fillna(df.rain_mm)
    df['month_sin']=np.sin(2*np.pi*df.month/12); df['month_cos']=np.cos(2*np.pi*df.month/12)
    df['dryness']=df.temp_c/(1+df.rain_lag_mm.clip(lower=0)); df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['fire'])
    df.to_csv(PROCESSED/'training_features.csv',index=False); log.info('Saved %s complete feature rows.',f'{len(df):,}'); return df

def spatial_cv(df,log):
    df=df.copy(); groups=df.longitude.floordiv(2).astype(str)+'_'+df.latitude.floordiv(2).astype(str)
    n=min(5,groups.nunique())
    if n<3 or df.fire.nunique()!=2: raise ValueError(f'Insufficient classes or spatial blocks for spatial validation (groups={groups.nunique()}, classes={df.fire.nunique()}).')
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan); y=df.fire.astype(int)
    splitter=GroupKFold(n_splits=n); predictions=np.full(len(df),np.nan); rows=[]
    for fold,(tr,te) in enumerate(splitter.split(X,y,groups=groups),1):
        if y.iloc[tr].nunique()<2 or y.iloc[te].nunique()<2:
            log.warning('Skipping fold %d because it contains only one class.',fold); continue
        model=RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',max_features=.8,n_jobs=-1,random_state=42)
        model.fit(X.iloc[tr],y.iloc[tr]); pred=model.predict_proba(X.iloc[te])[:,1]; predictions[te]=pred
        rows.append({'fold':fold,'train_rows':len(tr),'test_rows':len(te),'roc_auc':roc_auc_score(y.iloc[te],pred),'average_precision':average_precision_score(y.iloc[te],pred),'brier':brier_score_loss(y.iloc[te],pred)})
    valid=np.isfinite(predictions)
    if not valid.any(): raise ValueError('No spatial CV folds contained both classes.')
    fold_df=pd.DataFrame(rows); fold_df.to_csv(OUT/'spatial_validation_folds.csv',index=False)
    metrics={'validation':'GroupKFold by 2-degree spatial blocks','folds':int(len(fold_df)),'records':int(valid.sum()),'positive_rate':float(y.mean()),'roc_auc_mean':float(fold_df.roc_auc.mean()),'roc_auc_sd':float(fold_df.roc_auc.std(ddof=1)) if len(fold_df)>1 else None,'average_precision_mean':float(fold_df.average_precision.mean()),'brier_mean':float(fold_df.brier.mean()),'threshold':0.5,'precision_at_0_5':float(precision_score(y[valid],predictions[valid]>=.5,zero_division=0)),'recall_at_0_5':float(recall_score(y[valid],predictions[valid]>=.5,zero_division=0)),'f1_at_0_5':float(f1_score(y[valid],predictions[valid]>=.5,zero_division=0)),'confusion_matrix':confusion_matrix(y[valid],predictions[valid]>=.5).tolist()}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    log.info('Spatial CV ROC AUC %.3f (SD %s), average precision %.3f, Brier %.3f.',metrics['roc_auc_mean'],f'{metrics["roc_auc_sd"]:.3f}' if metrics['roc_auc_sd'] is not None else 'n/a',metrics['average_precision_mean'],metrics['brier_mean'])
    return metrics

def fit_and_explain(df,log):
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan); y=df.fire.astype(int)
    model=RandomForestClassifier(n_estimators=400,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',max_features=.8,n_jobs=-1,random_state=42)
    model.fit(X,y); joblib.dump({'model':model,'features':FEATURES,'training_scope':'Nigeria-wide monthly fire occurrence, 2003-2016'},OUT/'wildfire_model.joblib')
    # Model-agnostic held-out group importance; use impurity only when a spatial block lacks both labels.
    block=df.longitude.floordiv(2).astype(str)+'_'+df.latitude.floordiv(2).astype(str)
    hold=block==block.unique()[-1]
    if hold.sum()>100 and y[hold].nunique()==2:
        imp=permutation_importance(model,X.loc[hold],y.loc[hold],scoring='average_precision',n_repeats=6,random_state=42,n_jobs=-1)
        values=pd.DataFrame({'feature':FEATURES,'importance_mean':imp.importances_mean,'importance_sd':imp.importances_std})
    else:
        values=pd.DataFrame({'feature':FEATURES,'importance_mean':model.feature_importances_,'importance_sd':0.0})
    values.sort_values('importance_mean',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
    log.info('Leading predictors: %s',', '.join(values.sort_values('importance_mean',ascending=False).feature.head(4)))
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
    geom=shape(geometry)
    # Predict on the source Atlas 0.25-degree display grid inside Lagos. The classifier is trained on aggregated 0.5-degree cells, so these are spatially interpolated screening scores.
    lat0=math.floor(LAGOS_BOUNDS[1]*4)/4+.125; lon0=math.floor(LAGOS_BOUNDS[0]*4)/4+.125
    display=[]
    climate_anchors={'southwest':(5.75,3.5),'central':(8.5,6.5),'northwest':(11.5,6.5),'northeast':(12.5,11.5)}
    for lat in np.arange(lat0,LAGOS_BOUNDS[3],.25):
        for lon in np.arange(lon0,LAGOS_BOUNDS[2],.25):
            if geom.covers(Point(float(lon),float(lat))):
                zone=min(climate_anchors,key=lambda k:((lat-climate_anchors[k][0])**2+(math.cos(math.radians(lat))*(lon-climate_anchors[k][1]))**2))
                display.append({'latitude':float(lat),'longitude':float(lon),'climate_zone':zone})
    if not display: raise ValueError('No 0.25-degree display cells fall inside the Lagos state polygon.')
    points=pd.DataFrame(display)
    climate=(df.groupby(['climate_zone','month'],as_index=False)[['temp_c','rain_mm','rain_lag_mm','wind_kmh']].mean())
    base=points.merge(pd.DataFrame({'month':range(1,13)}),how='cross').merge(climate,on=['climate_zone','month'],how='left',validate='many_to_one')
    # Cross join above intentionally combines each Lagos grid cell with the 12-month climate normals for its nearest anchor.
    base['month_sin']=np.sin(2*np.pi*base.month/12); base['month_cos']=np.cos(2*np.pi*base.month/12)
    base['dryness']=base.temp_c/(1+base.rain_lag_mm.clip(lower=0))
    base['risk_probability']=model.predict_proba(base[FEATURES])[:,1]
    seasonal=base.groupby('month',as_index=False).risk_probability.mean()
    risk=base.groupby(['latitude','longitude'],as_index=False).risk_probability.mean()
    risk['relative_risk_percentile']=risk.risk_probability.rank(pct=True)*100
    risk['display_resolution_degrees']=.25
    risk.to_csv(PROCESSED/'lagos_risk_grid.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,7.5),dpi=180)
    sc=ax.scatter(risk.longitude,risk.latitude,c=risk.relative_risk_percentile,s=460,cmap='YlOrRd',vmin=0,vmax=100,marker='s',edgecolor='#243244',linewidth=.45)
    polys=[geom] if geom.geom_type=='Polygon' else list(geom.geoms)
    for poly in polys:
        x,y=poly.exterior.xy; ax.plot(x,y,color='#102033',linewidth=1.5)
    ax.set_xlim(LAGOS_BOUNDS[0],LAGOS_BOUNDS[2]); ax.set_ylim(LAGOS_BOUNDS[1],LAGOS_BOUNDS[3])
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.set_title('Lagos Wildfire Risk Screening\nRelative mean monthly score · 2003–2016 climate baseline',loc='left',fontsize=15,weight='bold',pad=12)
    ax.text(.01,-.14,'0.25° display grid; classifier trained at 0.5°. Colours are relative ranks within Lagos.\nHistorical screening view from Atlas ignitions and ERA5, not a live forecast or confirmed incident map.',transform=ax.transAxes,fontsize=8,color='#52616f')
    cb=fig.colorbar(sc,ax=ax,shrink=.78,pad=.03); cb.set_label('Relative risk percentile')
    ax.grid(color='#9aa7b2',alpha=.22); fig.tight_layout()
    fig.savefig(MAPS/'lagos_wildfire_risk.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
    imp=pd.read_csv(OUT/'feature_importance.csv').sort_values('importance_mean')
    fig,axes=plt.subplots(1,2,figsize=(13,5),dpi=180)
    axes[0].plot(seasonal.month,seasonal.risk_probability,color='#e66b32',marker='o',lw=2.4); axes[0].set_xticks(range(1,13)); axes[0].set_xlabel('Month'); axes[0].set_ylabel('Mean model score'); axes[0].set_title('Lagos seasonal profile',weight='bold'); axes[0].grid(alpha=.25)
    axes[1].barh(imp.feature,imp.importance_mean,xerr=imp.importance_sd,color='#188b91',alpha=.9); axes[1].set_title('Predictor importance',weight='bold'); axes[1].set_xlabel('Permutation importance (average precision)')
    fig.suptitle('Model explainability',x=.03,ha='left',fontsize=15,weight='bold'); fig.tight_layout(); fig.savefig(FIG/'model_explainability.png',bbox_inches='tight',facecolor='white'); plt.close(fig)
    seasonal.to_csv(PROCESSED/'lagos_seasonal_profile.csv',index=False)
    log.info('Saved %d 0.25-degree Lagos display cells and explainability figure; boundary source: %s.',len(risk),boundary_source)
    return risk,seasonal,boundary_source

def write_report(metrics,df,risk,seasonal,boundary_source):
    names=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    sd=metrics['roc_auc_sd'] if metrics['roc_auc_sd'] is not None else 0
    lines=['# Lagos Wildfire Risk Screening','',
    '## Executive summary','',
    'This project builds an interpretable spatial fire-occurrence screen for Lagos State and its peri-urban edge. It combines Global Fire Atlas ignition counts with ERA5 reanalysis, trains on Nigeria-wide fire patterns, and evaluates generalisation by holding out spatial blocks.','',
    '**The Lagos map shows relative historical screening scores, not a real-time warning or a calibrated probability of an incident.** Satellite fire records can miss small burns and may include thermal anomalies. Local incident validation is required before operational use.','',
    '## Study design','',
    f'- Presentation area: Lagos State with nearby peri-urban context; boundary source: {boundary_source}.',
    '- Training region: Nigeria land grid cells within approximately 2.75–15°E and 4.25–14°N.',
    '- Fire labels: Global Fire Atlas monthly fire-number rasters (2003–2016), aggregated to 0.5° modelling cells.',
    '- Weather predictors: Open-Meteo ERA5 daily temperature, precipitation, and wind summarized monthly from four regional anchors.',
    '- Validation: GroupKFold spatial cross-validation by 2° blocks; grid cells in a held-out block are excluded from training.',
    '- Model: class-weighted Random Forest; permutation importance and seasonal profiles support interpretation.','',
    '## Validation results','',
    f'- Training records: {len(df):,}; fire-positive grid-month prevalence: {df.fire.mean():.2%}.',
    f'- Spatial ROC AUC: {metrics["roc_auc_mean"]:.3f} (fold SD {sd:.3f}).',
    f'- Spatial average precision: {metrics["average_precision_mean"]:.3f}; Brier score: {metrics["brier_mean"]:.3f}.',
    f'- At fixed threshold 0.50: precision {metrics["precision_at_0_5"]:.3f}, recall {metrics["recall_at_0_5"]:.3f}, F1 {metrics["f1_at_0_5"]:.3f}.',
    '- Read ROC AUC alongside average precision and Brier because fire months are rare; accuracy alone is not informative.','',
    '## Reading the Lagos map','',
    f'The map ranks {len(risk)} model cells by mean model probability over the 12 calendar months. Percentiles are relative to the displayed Lagos-area grid. The surface represents a historical climate/fire-regime baseline, not a future-day forecast.','',
    '### Seasonal profile','', '| Month | Mean model probability |','|---|---:|']
    for _,row in seasonal.iterrows(): lines.append(f'| {names[int(row.month)-1]} | {row.risk_probability:.4f} |')
    lines += ['', '## Limitations and next steps','',
    '1. Compare satellite labels with geocoded Lagos State Fire and Rescue Service records; the current target is a satellite-derived fire proxy.',
    '2. Improve spatial detail with Sentinel-2 fuel/moisture and land-cover layers, settlement/road variables, and verified fire perimeters.',
    '3. Treat probabilities as relative until a time-based independent validation and local calibration are completed.',
    '4. The fitted baseline uses 2003–2016 fire history; a full recent archive is needed to update the model responsibly.','',
    '## Sources','',
    '- Global Fire Atlas updated record (2002–2024): https://zenodo.org/records/14065246',
    '- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api',
    '- GeoBoundaries open boundaries: https://gist.github.com/segebee/ff4a2f75be560462c369','',
    '## Reproducibility','',
    'Run python run.py from the project root. Raw inputs, processed tables, model, figures, maps, metrics, report, and logs are stored under this project. The data archive is excluded from Git; its source URL and retrieval logic are recorded in the code and README.','']
    (ROOT/'reports'/'project_report.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    log=setup_logging(); log.info('Starting Lagos wildfire risk screening.')
    fire=load_fire_grid(log); weather=fetch_weather(fire,log); df=build_features(fire,weather,log)
    metrics=spatial_cv(df,log); model=fit_and_explain(df,log); risk,seasonal,boundary=make_outputs(model,df,log)
    write_report(metrics,df,risk,seasonal,boundary)
    log.info('Pipeline complete. Outputs are in %s.',OUT)

if __name__=='__main__': main()
