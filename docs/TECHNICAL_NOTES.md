# Technical notes: Lagos wildfire risk screening

This document explains the data engineering, model, validation and GIS terms behind the project. The concise project summary is in the [README](../README.md), while run configuration is in [`config.yaml`](../config.yaml).

## Study table and target construction

The unit of analysis is a **grid-cell-month** on the Global Fire Atlas native 0.25-degree grid. The script `src/lagos_fire_risk/pipeline.py` opens the archive remotely with HTTP byte-range reads when no local archive exists, selects only the 168 `fire_number` TIFF members for 2003–2016, opens each member in memory, and reads only the Nigeria window. Zero counts are explicitly included for Nigeria land cells/months with no mapped fire. The classifier target is binary: `fire = 1` when the monthly fire-number count is positive, otherwise `fire = 0`.

Coordinates in `data/processed/lagos_risk_grid.csv` identify native grid cells. A 0.25-degree cell is several tens of kilometres wide and varies in ground area with latitude; it is not a neighbourhood-scale fire map. Cell polygons shown for Lagos are intersected with the state boundary for display.

## Predictor construction

The saved proposal-complete model combines three groups of predictors:

1. **Location and season:** cell latitude/longitude and cyclic month encodings (`sin(2π month/12)`, `cos(2π month/12)`). Cyclic encoding avoids a numerical discontinuity between December and January.
2. **ERA5 weather via Open-Meteo:** monthly mean 2 m temperature, total precipitation, monthly mean of daily maximum wind speed, and monthly mean relative humidity. Four climate anchors approximate Nigeria's broad north–south/east–west gradients; each cell is assigned to the nearest anchor with longitude distance adjusted by cosine(latitude).
3. **MODIS Earth Engine aggregates:** monthly NDVI/EVI from `MODIS/061/MOD13Q1` and daytime/nighttime land-surface temperature from `MODIS/061/MOD11A2`, with product scale factors and QA masks applied. The export summarizes values over the 0.25-degree analysis cells at 1 km aggregation scale.

`rain_lag_mm` is the previous month’s precipitation per grid cell (the first month falls back to current rainfall). `dryness = temp_c / (1 + max(rain_lag_mm, 0))` is a derived heuristic predictor. Rows with missing required features are removed after weather/MODIS joins; join cardinality and duplicate keys are validated.

## Models and spatial validation

The candidates are scikit-learn `RandomForestClassifier` (250 trees, max depth 12) and `HistGradientBoostingClassifier` (200 boosting iterations, learning rate 0.08, up to 31 leaves). Both use seed 42 and balanced per-record sample weights. Model input columns are kept in a serialized feature list with the fitted estimator to preserve inference order.

Validation groups records by 2-degree geographic blocks and uses `GroupKFold` (five folds when enough groups are available). All observations in a block stay together in a fold; this makes validation harder and more informative than randomly splitting neighbouring cell-month rows. Scores for each model are ROC AUC, average precision, and Brier score per fold. Selection uses mean average precision; ties prefer higher ROC AUC and then lower Brier score. Thresholded precision/recall/F1 and confusion matrices use 0.50 and pooled out-of-fold predictions.

The saved run has 191,976 complete Nigeria grid-month records, five spatial folds, and selects HistGradientBoosting. Its reported spatial-fold means are ROC AUC 0.944 (fold SD 0.013), average precision 0.809, and Brier score 0.092. The saved `outputs/metrics.json` is the source for the dashboard and report.

## Lagos score surface

After final fitting on all 2003–2016 records, the model scores 12-month climatological predictor averages for the three native grid cells whose centers are inside Lagos. Monthly predictor values are averaged within each cell/month across the historical years; the model score is then attached to the cell geometry. `relative_risk_percentile` ranks those displayed cells against each other. It is a ranking of historical model scores, not an incident probability, calibrated risk or forecast. Fiona/GeoPandas clipping intersects cell polygons with the Lagos state geometry; it does not increase the model's spatial resolution.

Feature explanation uses SHAP for a sampled subset when available. Permutation importance (average-precision decrease) is the labelled fallback. Neither measures causal effects.

## Independent VIIRS time/sensor check

`scripts/validate_lagos_viirs.py` keeps the fitted 2003–2016 classifier frozen, creates monthly MODIS predictors remotely in Earth Engine for 2017–2024, and labels cell-months from NASA VIIRS VNP14A1 `FireMask` confidence classes 8–9. Weather comes from Open-Meteo. Earth Engine returns only compact summaries for the three model cells instead of imagery collections. 284 of 288 possible rows have all required predictors. The recorded scores are ROC AUC 0.856, average precision 0.894, and Brier score 0.396 with a 65.1% positive rate.

This is an independent in-time and sensor-product comparison, but only three large Lagos cells contribute. VIIRS active-fire detections are a satellite proxy, not ground-truthed fire-service incidents. The high positive rate and Brier score should be read alongside AUC/AP; these metrics do not establish calibrated operational probabilities.

## GIS, provenance and saved artifacts

- **CRS:** coordinate reference system; rasters and vector features are transformed to matching coordinate spaces before overlay/aggregation.
- **Raster window:** a selected rectangular subset of a raster. The remote ZIP reader uses byte ranges to fetch just requested compressed TIFF members without persisting the full archive.
- **Native resolution:** the Atlas target grid is 0.25 degrees. MODIS pixels are aggregated to those cells; no downscaling of the fire label is implied.
- **Out-of-fold prediction:** a prediction made for rows in a fold that was excluded from that fold's fitting. These are the records used for validation metrics.
- **Brier score:** mean squared error between predicted probability and binary outcome; lower is better, and it reflects both ranking/calibration.
- **Calibration:** agreement between predicted probability levels and observed frequency. Current dashboards call the Lagos output a score, and do not present it as a calibrated probability.
- **Temporal holdout:** evaluation on later years not used to fit the historical model; in this case it also changes target sensor product from Global Fire Atlas to VIIRS.

`outputs/metrics.json`, `outputs/spatial_validation_folds.csv`, `outputs/model_comparison.csv`, `outputs/viirs_validation_metrics.json`, the compact Lagos grid tables, saved model, logs, report and figures are linked data products. The dashboard reads these saved products; it does not run Earth Engine or retrain on page load.

## Checks and CI

`scripts/verify_outputs.py` recalculates basic confusion-matrix and VIIRS metrics from the saved CSVs and checks row counts, periods, unique cell-month keys, positive counts and finite probability ranges. The GitHub Actions workflow installs the dashboard/metric dependencies, runs that verifier, and runs pytest. The app smoke test uses Streamlit's `AppTest` to render the saved-results dashboard and checks tabs and expected metrics.
