# Lagos Wildfire Risk Screening

A geospatial machine-learning workflow for historical fire screening. The model is trained across Nigeria using real satellite fire observations, ERA5 weather, and MODIS vegetation and thermal summaries; its displayed grid cells are clipped to Lagos State.

## Data and model

- **Fire labels:** Global Fire Atlas monthly fire-number rasters, 2003–2016, aggregated at their native 0.25° modelling grid.
- **Vegetation:** MODIS MOD13Q1 monthly NDVI and EVI summaries.
- **Thermal conditions:** MODIS MOD11A2 monthly daytime and nighttime land-surface temperature.
- **Weather:** ERA5 reanalysis through Open-Meteo, including temperature, rainfall, wind, and humidity.
- **Model comparison:** Random Forest and HistGradientBoosting, evaluated on the same GroupKFold spatial folds.
- **Selection and interpretation:** select by mean spatial-fold average precision (ties by ROC AUC, then Brier score); explain the fitted model with SHAP when available, with permutation importance as a clearly labelled fallback.
- **Lagos output:** predict monthly climatological scores for model cells whose centers fall inside Lagos, clip each cell polygon to the state boundary, and export the map, CSV, and GeoJSON.

### Current saved model run

The current saved run joins real Global Fire Atlas monthly fire labels, ERA5 weather, and quality-masked MODIS NDVI, EVI, and day/night land-surface temperature. It compares Random Forest with HistGradientBoosting using five-fold GroupKFold validation by 2° geographic blocks. HistGradientBoosting was selected by mean spatial-fold average precision. On 191,976 complete Nigeria grid-month records at the native 0.25° grid, the saved spatial-CV results are ROC AUC 0.944 (fold SD 0.013), average precision 0.809, and Brier score 0.092. The model is trained nationally; its Lagos map shows mean monthly model scores for the three native grid-cell centers inside Lagos, with cell polygons clipped to the state boundary. These scores are relative historical screening outputs, not incident probabilities or operational forecasts.

An independent later-year check applies the frozen classifier to Lagos predictors from 2017–2024 and compares them with NASA VIIRS VNP14A1 active-fire detections. Earth Engine computes monthly MODIS features and VIIRS labels remotely for the three Lagos cells; only compact cell-month summaries are returned. Of 288 possible cell-month rows, 284 have complete predictors. This holdout gives ROC AUC 0.856, average precision 0.894, and Brier score 0.396 against a VIIRS-positive rate of 65.1%. It is reported separately from Global Fire Atlas spatial cross-validation and is not a service-verified incident evaluation.

Fire rasters can be read from the existing local archive. When that archive is absent, the pipeline reads only selected monthly TIFF members through HTTP byte ranges and opens them in memory; it does not save the full archive. ERA5 features are requested through Open-Meteo and cached as a small monthly table.

## Prepare MODIS inputs

The MODIS export uses Google Earth Engine and requires an authenticated Earth Engine-enabled Google Cloud project. Earth Engine calculates monthly summaries over the Nigeria grid; only the resulting feature table is exported, rather than the original satellite rasters.

```powershell
earthengine authenticate
python -m pip install -r requirements.txt
python -m src.lagos_fire_risk.modis_export --project YOUR_EARTH_ENGINE_PROJECT_ID
```

Download the completed CSV from Google Drive to `data/processed/modis_monthly.csv`. It contains monthly cell-level mean NDVI, EVI, daytime LST, and nighttime LST for 2003–2016. The export masks lower-quality MODIS observations and uses a common 0.25° Nigeria land grid.

## Run the wildfire workflow

```powershell
python run.py
```

After preparing the MODIS feature table, `python run.py` joins it to the fire and weather records, compares the candidate models on shared spatially held-out 2° blocks, fits the selected model, explains it, and regenerates the Lagos outputs and report.

To train the current fire-and-weather baseline without MODIS, run:

```powershell
python run.py --without-modis
```

To reuse the saved training table and model comparison while regenerating the fitted model and maps, add `--resume-features`.

To reproduce the independent later-period Lagos check, authenticate Earth Engine and run:

```powershell
python scripts/validate_lagos_viirs.py --project YOUR_EARTH_ENGINE_PROJECT_ID
```

The script applies the saved 2003–2016 model to 2017–2024 ERA5/MODIS predictors, compares scores with VIIRS VNP14A1 detections, and writes CSV, JSON, and Markdown outputs. Earth Engine computes the satellite summaries remotely; the script does not download full image collections.

## Browse saved results

```powershell
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

The Streamlit app presents saved metrics, model comparisons, the Lagos boundary-clipped map, model feature importance, and reproducibility artifacts. It does not fetch data or train models. The compact results snapshot and fitted model are tracked so the dashboard can open directly from a clone. Large source exports and the complete training table remain local and ignored. The updated, full-model presentation is [Lagos_Wildfire_Risk_Screening_Updated.pptx](presentation/Lagos_Wildfire_Risk_Screening_Updated.pptx); slide previews are in `presentation/previews/`.

## Main files

- `src/lagos_fire_risk/pipeline.py` — remote fire-archive reads, weather ingestion, feature construction, model comparison, spatial validation, explanations, and Lagos outputs.
- `src/lagos_fire_risk/modis_export.py` — Earth Engine export for monthly MODIS vegetation and thermal features.
- `scripts/validate_lagos_viirs.py` — independent 2017–2024 Lagos evaluation using remotely aggregated NASA VIIRS fire detections.
- `config.yaml` — study bounds, time span, native grid size, spatial folds, candidate models, and SHAP sample size.
- `PROJECT_KNOWLEDGE_BASE.md` — project methods, source provenance, current saved results, and reporting guidance.
- `docs/TECHNICAL_NOTES.md` — feature construction, spatial cross-validation, remote data reads, evaluation, and GIS terminology.
- `reports/project_report.md` — generated research summary, regenerated by `python run.py`.
- `reports/viirs_validation.md` — design and results of the independent later-period Lagos satellite holdout.

## Data sources

- [Global Fire Atlas](https://zenodo.org/records/14065246) — satellite-derived monthly fire-number observations.
- [MODIS MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1) — 16-day NDVI/EVI product aggregated monthly.
- [MODIS MOD11A2](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2) — 8-day land-surface temperature product aggregated monthly.
- [NASA VIIRS VNP14A1](https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP14A1) — independent 1 km daily active-fire product, aggregated monthly for the 2017–2024 Lagos holdout.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — ERA5 reanalysis summaries.
- [Nigeria state polygons](https://gist.github.com/segebee/ff4a2f75be560462c369) — national land grid and Lagos State boundary.
