# Lagos Wildfire Risk Screening

## Executive summary

This project builds an interpretable spatial fire-occurrence screen clipped to Lagos State. It combines Global Fire Atlas fire observations with MODIS vegetation/thermal metrics and ERA5 reanalysis, trains on Nigeria-wide fire patterns, and evaluates generalisation by holding out spatial blocks.

**The Lagos map shows relative historical screening scores, not a real-time warning or a calibrated probability of an incident.** Satellite fire records can miss small burns and may include thermal anomalies. Local incident validation is required before operational use.

## Study design

- Presentation area: Lagos State with nearby peri-urban context; boundary source: Nigeria state polygon from segebee/logbon72 gist.
- Training region: Nigeria land cells within approximately 2.5–15°E and 4–14°N; only map cells whose centers fall within Lagos State are displayed.
- Fire labels: Global Fire Atlas monthly fire-number rasters (2003–2016) at the native 0.25° modelling grid.
- Predictors: Open-Meteo ERA5 weather plus monthly MODIS NDVI, EVI, and day/night LST.
- Validation: GroupKFold spatial cross-validation by 2° blocks; all candidate models use the same held-out folds.
- Selected model: HistGradientBoosting, chosen by mean spatial-fold average precision; Permutation importance (SHAP unavailable in this runtime) summarizes fitted feature contributions.

## Validation results

- Training records: 191,976; fire-positive grid-month prevalence: 19.42%.
- Spatial ROC AUC: 0.944 (fold SD 0.013).
- Spatial average precision: 0.809; Brier score: 0.092.
- At fixed threshold 0.50: precision 0.613, recall 0.873, F1 0.721.
- Read ROC AUC alongside average precision and Brier because fire months are rare; accuracy alone is not informative.

## Reading the Lagos map

The map ranks 3 native 0.25° model cells by mean model score over calendar months. Each cell polygon is clipped to the Lagos State boundary, and percentiles are relative to the displayed cells. The surface represents a historical fire/climate baseline, not a future-day forecast.

### Seasonal profile

| Month | Mean model probability |
|---|---:|
| Jan | 0.7880 |
| Feb | 0.4144 |
| Mar | 0.2061 |
| Apr | 0.0430 |
| May | 0.0007 |
| Jun | 0.0002 |
| Jul | 0.0005 |
| Aug | 0.0015 |
| Sep | 0.0025 |
| Oct | 0.0049 |
| Nov | 0.3433 |
| Dec | 0.6986 |

## Model comparison

| Candidate | Spatial ROC AUC | Average precision | Brier score |
|---|---:|---:|---:|
| RandomForest | 0.941 | 0.783 | 0.092 |
| HistGradientBoosting | 0.944 | 0.809 | 0.092 |

## Independent Lagos VIIRS holdout

- Later evaluation period: 2017–2024; 284 complete cell-months across three Lagos cells.
- NASA VIIRS-positive cell-month rate: 65.14%.
- ROC AUC: 0.856; average precision: 0.894; Brier score: 0.396.
- At 0.50 threshold: precision 0.938, recall 0.324.
This later-period satellite-sensor holdout is reported separately from Global Fire Atlas spatial cross-validation and is not a fire-service incident audit.

## Sources

- Global Fire Atlas updated record (2002–2024): https://zenodo.org/records/14065246
- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api
- MODIS MOD13Q1 vegetation indices (NDVI/EVI): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
- MODIS MOD11A2 land surface temperature: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2
- NASA VIIRS VNP14A1 active fire: https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP14A1
- GeoBoundaries open boundaries: https://gist.github.com/segebee/ff4a2f75be560462c369

## Reproducibility

Run python run.py from the project root. Raw inputs, processed tables, model, figures, maps, metrics, report, and logs are stored under this project. The data archive is excluded from Git; its source URL and retrieval logic are recorded in the code and README.
