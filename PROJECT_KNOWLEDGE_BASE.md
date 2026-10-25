# Project Knowledge Base: Lagos Wildfire Risk Screening

This document is the detailed reference for explaining, reporting, and presenting this project. It describes the implementation and saved results in the project folder. If a slide, an older report, or a generated figure conflicts with the code or saved data described here, verify the underlying code and data before repeating the claim.

## 1. Project at a glance

**Project name:** Lagos Wildfire Risk Screening  
**Project type:** SIWES geospatial machine learning capstone  
**Proposal reference:** Project 3, “Wildfire Risk Prediction via Multi-Source Data Fusion,” from the supplied SIWES proposal  
**Study area:** Lagos State, Nigeria  
**Historical fire-label period:** 2003–2016  
**Canonical project folder:** `C:\Projects\lagos-wildfire-risk-screening`

The full source pipeline combines real Global Fire Atlas fire observations, monthly MODIS vegetation and thermal summaries, and ERA5 weather. It compares Random Forest and HistGradientBoosting with geographically separated folds, selects by spatial-fold average precision, explains the selected model, and clips native-grid model cells to Lagos State. Training remains national so the model has a larger set of historical fire examples; the mapped output is restricted to Lagos.

**Current implementation status:** a new real-data baseline has been trained on 199,584 Nigeria grid-month records at the native 0.25° grid, using Global Fire Atlas fire labels and ERA5 weather. Five-fold spatial validation selected HistGradientBoosting (ROC AUC 0.936, average precision 0.771, Brier score 0.097). Its Lagos output covers three grid-cell centers. The current baseline excludes MODIS because Earth Engine credentials are not present in this workspace. The code includes a MODIS exporter for the full proposal feature set. The presentation deck and slide previews are still from an earlier run and have not been regenerated from this baseline.

The phrase “risk screening” is used as a project label. The map is not an operational fire warning, a forecast, a confirmed incident map, or a property-level risk assessment. Its scores are model outputs based on coarse historical inputs.

## 2. Why this project

Vegetation fires can affect settlements, infrastructure, health, and ecosystems. Lagos is a useful presentation area because it combines dense urban development with peri-urban and vegetated areas. A geospatial model provides a way to bring together fire history, climate, location, and maps in one analysis.

The practical question is:

> Where do historical satellite fire patterns and broad monthly climate conditions indicate comparatively higher vegetation-fire occurrence scores within the Lagos display area?

This question is deliberately narrower than asking where the next fire will occur. The available inputs do not include recent, local fire-service incident records, detailed fuels, current weather forecasts, or verified fire perimeters. The project therefore demonstrates a screening method; it does not establish a deployable public-safety service.

## 3. What the project does

1. Retrieves Global Fire Atlas monthly fire-number rasters for 2003–2016 and aggregates observations on the native 0.25-degree model grid.
2. Creates a Nigeria land-cell/month table, including zero-fire months as negative examples.
3. Retrieves ERA5 weather summaries from Open-Meteo and, for the full run, joins the monthly MODIS NDVI/EVI/LST table exported from Earth Engine.
4. Builds seasonal, rainfall, dryness, wind, temperature, humidity, location, and (when MODIS is enabled) vegetation and thermal features.
5. Compares Random Forest and HistGradientBoosting with shared GroupKFold spatial blocks, selecting by mean average precision.
6. Fits the selected model, writes SHAP explanations when available or a labelled permutation-importance fallback, and predicts monthly scores only for model cells whose centers fall inside the Lagos boundary.
7. Clips each 0.25-degree output cell polygon to Lagos State and saves map, CSV, GeoJSON, metrics, model-comparison tables, report, and run log.

## 4. Data sources and their roles

### Global Fire Atlas

- **Dataset:** Global Fire Atlas monthly fire-number GeoTIFF summaries.
- **Source record:** [Global Fire Atlas on Zenodo](https://zenodo.org/records/14065246).
- **Download used by the code:** `https://zenodo.org/records/14065246/files/GeoTIFF_Qdeg_monthly_summaries.zip`.
- **Time used:** 2003–2016 inclusive, or 168 calendar months per grid cell.
- **Source resolution:** 0.25 degrees.
- **Model resolution:** 0.25 degrees, matching the source raster grid.
- **Role:** Provides the satellite-derived fire-number observations used to make a binary monthly target.

The target is `fire = 1` when the aggregated monthly fire-number value is greater than zero; it is `fire = 0` otherwise. It is a satellite-derived proxy for fire occurrence. It is not a complete inventory of fires or a record confirmed by a local fire service. Small fires may be missed, and thermal detections may not always correspond to the intended vegetation-fire event.

### Weather reanalysis

- **Provider/API:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), using ERA5 reanalysis.
- **Weather variables retrieved daily:** mean 2 m temperature, precipitation sum, maximum 10 m wind speed, and mean relative humidity.
- **Weather features used by the model:** temperature, precipitation, wind speed, humidity, and derived rainfall features.
- **Sampling locations:** four regional anchors: southwest `(5.75°N, 3.5°E)`, central `(8.5°N, 6.5°E)`, northwest `(11.5°N, 6.5°E)`, and northeast `(12.5°N, 11.5°E)`.
- **Assignment:** each 0.25-degree cell is assigned to its nearest anchor using latitude-adjusted longitude distance.
- **Role:** Provides broad monthly climate conditions at much lower cost and query volume than requesting weather separately for every grid cell.

The weather anchors are a coarse approximation. They do not reproduce local Lagos microclimates or cell-level weather variation.

### Nigeria state boundaries

- **Source used:** [Public Nigeria state polygon data on Gist](https://gist.github.com/segebee/ff4a2f75be560462c369).
- **Role:** Defines Nigeria land cells for the training table and clips the Lagos display grid to the Lagos state polygon.
- **Coordinate handling:** The code interprets the source coordinate pairs as latitude/longitude and converts them to the longitude/latitude order expected by the geometry library.

The boundary file is downloaded and cached as `data/raw/nigeria_state_polygons.json` when needed. For formal publication, confirm the source’s licensing, provenance, and boundary accuracy.

## 5. Study area, sample, and spatial scales

The current training grid is built from Nigeria state polygons. Candidate cell centers are spaced every 0.25 degrees, approximately across 2.625–14.875°E and 4.125–13.875°N; a center is included when it is covered by the union of state polygons. Exact grid and row counts are generated by the current data run.

The current saved baseline uses 1,188 national cells and 199,584 cell-month records, with 38,861 fire-positive rows (19.47%). Lagos output uses three 0.25-degree model-cell centers; each displayed polygon is clipped to the Lagos State boundary.

## 6. Feature construction

The current model uses these features, in the order defined in `src/lagos_fire_risk/pipeline.py`:

| Feature | Meaning |
|---|---|
| `latitude` | Center latitude of the 0.25-degree training cell. |
| `longitude` | Center longitude of the 0.25-degree training cell. |
| `month_sin` | Sine encoding of calendar month to represent season cyclically. |
| `month_cos` | Cosine encoding of calendar month to represent season cyclically. |
| `temp_c` | Monthly average of daily mean 2 m temperature, in °C. |
| `rain_mm` | Monthly total precipitation, in mm. |
| `rain_lag_mm` | Previous month’s rainfall for the same cell; for the first available month it is set to that month’s rainfall. |
| `wind_kmh` | Monthly average of daily maximum 10 m wind speed, in km/h. |
| `humidity` | Monthly average relative humidity, in percent. |
| `dryness` | Derived index `temp_c / (1 + rain_lag_mm)`. Larger values indicate warmer conditions with less prior-month rainfall under this simple formula. |
| `ndvi` | Monthly mean MODIS NDVI from MOD13Q1, scaled by 0.0001 before aggregation. |
| `evi` | Monthly mean MODIS EVI from MOD13Q1, scaled by 0.0001 before aggregation. |
| `lst_day_c` | Monthly mean daytime MODIS land-surface temperature from MOD11A2, converted to °C. |
| `lst_night_c` | Monthly mean nighttime MODIS land-surface temperature from MOD11A2, converted to °C. |

MODIS vegetation and thermal features are aggregated to the 0.25-degree cell/month scale. Their source products have different native resolutions and observation cadences; the Earth Engine exporter applies quality masks, monthly compositing, and cell-level mean reduction.

## 7. Model and validation

### Classifier

The current code compares scikit-learn `RandomForestClassifier` and `HistGradientBoostingClassifier`. Both use balanced sample weights and are evaluated with the same spatial folds. The candidate with the highest mean spatial-fold average precision is selected; ties are resolved by ROC AUC and then Brier score. The selected estimator is fit on the full feature table and saved with its algorithm name, feature list, and training-scope metadata at `outputs/wildfire_model.joblib`.

### Spatial cross-validation

The code assigns each cell to a group using configurable 2-degree longitude and latitude blocks, then uses five-fold `GroupKFold`. All records from a spatial group are held together, so the same group’s cells/months do not appear in both a fold’s training and test partitions. Both candidate models use identical folds.

This is spatial validation within the historical Nigeria dataset. It is **not** a time-based test on later years, an independent Lagos incident validation, or a calibration test. The final Lagos estimates are a geographic transfer beyond the validation result; the Nigeria-wide metrics do not prove local Lagos accuracy.

### Saved validation results

The current saved results are the fire-and-ERA5 baseline. Run `python run.py` after placing the exported MODIS CSV at `data/processed/modis_monthly.csv` to train the full feature-set model. AUC, average precision, and Brier score below are arithmetic means across five spatial folds.

| Metric | Result | Plain-language reading |
|---|---:|---|
| ROC AUC | 0.9359 (fold SD 0.0129) | Ability to rank a positive row above a negative row across thresholds. |
| Average precision | 0.7709 | Precision-recall summary; positive prevalence is 0.1947. |
| Brier score | 0.0971 | Mean squared probability error; lower is better. |
| Precision at 0.50 | 0.6039 | Of rows called positive at this threshold, about 60% were positive in pooled held-out rows. |
| Recall at 0.50 | 0.8576 | Of positive held-out rows, about 86% were identified at this threshold. |
| F1 at 0.50 | 0.7087 | Harmonic mean of precision and recall at that threshold. |

Pooled confusion matrix at threshold 0.50, with actual classes as rows and predicted classes as columns:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 138,861 | 21,862 |
| Actual 1 | 5,532 | 33,329 |

These metrics describe historical fire-proxy classification on spatially held-out Nigeria blocks. They should not be reported as the expected accuracy of an operational Lagos warning system.

## 8. Lagos outputs and interpretation

The current saved fire-and-weather baseline regenerates `data/processed/lagos_risk_grid.csv` and a boundary-clipped `data/processed/lagos_risk_grid.geojson` on the same 0.25-degree grid as training:

| Cell center (latitude, longitude) | Mean 12-month model score | Relative percentile |
|---|---:|---:|
| (6.625, 3.375) | 0.2439 | 100 |
| (6.625, 3.625) | 0.2363 | 50 |
| (6.625, 3.875) | 0.2363 | 50 |

The table above is from the current 0.25-degree fire-and-ERA5 baseline. Percentiles rank Lagos cell scores against one another. The score is not a calibrated probability of a future fire. In this project, the word “risk” on the map is shorthand for a relative model score.

The seasonal profile below is from the current baseline; `data/processed/lagos_seasonal_profile.csv` is regenerated across current Lagos model cells from ERA5 weather features:

| Month | Model score |
|---|---:|
| January | 0.8845 |
| February | 0.7626 |
| March | 0.3830 |
| April | 0.0228 |
| May | 0.00003 |
| June | 0.00007 |
| July | 0.00171 |
| August | 0.00007 |
| September | 0.00014 |
| October | 0.00054 |
| November | 0.1659 |
| December | 0.6444 |

This seasonal curve is generated by the fitted model under the available climate averages. It is a modelled pattern, not an observed Lagos monthly incident rate. It should be checked against local incident history before drawing operational or causal conclusions.

## 9. Explainability

The current run saves predictor importance in `outputs/feature_importance.csv` and the combined seasonal/importance chart in `outputs/figures/model_explainability.png`. The current local environment could not import SHAP, so this baseline uses explicitly labelled permutation importance. A full run uses SHAP when the runtime supports it. Neither method establishes that a predictor causes fire occurrence.

## 10. Reproducing the project

From the project root, with Python 3.10 or newer:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

For the proposal-complete model, authenticate to Earth Engine, start the MODIS export with `python -m src.lagos_fire_risk.modis_export --project YOUR_EARTH_ENGINE_PROJECT_ID`, and save the completed Drive CSV as `data/processed/modis_monthly.csv`. Then run `python run.py`. The pipeline reads only the required Global Fire Atlas monthly raster members (using an existing local archive or HTTP byte ranges), retrieves ERA5 weather for four regional locations, joins the MODIS table, compares models, and regenerates feature rows, cross-validation tables, the selected estimator, explanation output, Lagos map and GeoJSON, report, and log. ERA5 summaries and source rasters are cached locally.

To refresh only the Lagos score/map and report from existing saved inputs and model, run:

```powershell
python refresh_outputs.py
```

To browse the current saved results in the Streamlit dashboard, run:

```powershell
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

The dashboard is read-only with respect to the analysis: it displays saved metrics, the map, explanation charts, fold results, artifact availability, and recent logs. It does not rerun the model or make API calls. Run `python run.py` separately to regenerate project outputs.

The MODIS export requires an Earth Engine-enabled Google Cloud project and Google account authentication. The other sources are retrieved from their public endpoints. Package versions in `requirements.txt` use minimum version bounds rather than a fully locked environment, so exact bit-for-bit reproduction across future software releases is not guaranteed.

## 11. Project file guide

| Path | Purpose |
|---|---|
| `run.py` | Small entry point that runs the full pipeline. |
| `app.py` | Read-only Streamlit dashboard for browsing saved metrics, map, explanations, files, and logs. |
| `src/lagos_fire_risk/pipeline.py` | Data retrieval, preparation, feature engineering, model, validation, figures, map, and report generation. |
| `src/lagos_fire_risk/modis_export.py` | Earth Engine export of monthly MODIS NDVI, EVI, daytime LST, and nighttime LST by grid cell. |
| `refresh_outputs.py` | Rebuilds Lagos outputs and report from existing cached features and saved model. |
| `config.yaml` | Project area, grid resolution, data years, random seed, spatial block size, and fold count. |
| `requirements.txt` | Python dependency minimums. |
| `requirements-dashboard.txt` | Minimal dependencies for the read-only dashboard. |
| `.streamlit/config.toml` | Dashboard theme and headless server setting. |
| `data/raw/` | Downloaded Atlas archive, extracted monthly rasters, and state polygon file. |
| `data/processed/` | Weather cache, MODIS CSV, model input table, Lagos grid scores/GeoJSON, and monthly score profile. |
| `outputs/metrics.json` | Machine-readable validation summary. |
| `outputs/spatial_validation_folds.csv` | Per-fold validation metrics and sample counts. |
| `outputs/model_comparison.csv` | Spatial validation summary for candidate models. |
| `outputs/feature_importance.csv` | SHAP or explicitly labelled permutation importance for the selected model. |
| `outputs/wildfire_model.joblib` | Final fitted classifier and metadata. |
| `outputs/maps/lagos_wildfire_risk.png` | Lagos screening map. |
| `outputs/figures/model_explainability.png` | Lagos seasonal score profile and selected-model feature-importance summary. |
| `data/processed/lagos_risk_grid.geojson` | Model cell polygons intersected with the Lagos State boundary. |
| `outputs/logs/pipeline.log` | Timestamped run log. |
| `outputs/presentation_previews/` | Nine PNG slide previews. |
| `reports/project_report.md` | Concise written project report. |
| `reports/presentation_outline.md` | Initial outline for presenting the project. |
| `presentation/Lagos_Wildfire_Risk_Screening.pptx` | Nine-slide presentation deck. |

Raw data, processed data, and generated `outputs/` are stored locally in the project folder and ignored by Git to avoid putting large generated files into source history. The README, source, requirements, report, outline, and presentation are available to Git.

## 12. Limitations to state clearly in a report or presentation

1. **Historical period:** fire labels end in 2016; this project does not use current fire detections.
2. **Proxy target:** satellite-derived fire-number values are not a complete, locally verified incident register.
3. **Grid scale:** fire labels use the Global Fire Atlas native 0.25-degree grid; each cell is a regional model unit.
4. **Lagos display:** scores are shown only for native grid-cell centers inside the Lagos boundary, and cell polygons are clipped to that boundary.
5. **Climate sampling:** the current baseline uses four regional ERA5 anchors; the proposal-complete feature set also adds MODIS vegetation and thermal summaries aggregated to the model grid.
6. **Geographic transfer:** Nigeria spatial cross-validation does not directly validate the Lagos map or demonstrate local calibration.
7. **No temporal holdout:** the saved evaluation does not test whether a model trained on earlier years transfers to later years.
8. **Model explanation:** SHAP values describe fitted model behavior and are not causal estimates.
9. **No causal claim:** predictor importance and seasonal scores describe model associations, not causes.
10. **No operational use:** do not use these outputs for dispatch, evacuation, insurance, property decisions, or public alerts without updated inputs, local validation, calibration, and expert review.

## 13. Recommended next steps

1. Authenticate an Earth Engine-enabled Cloud project, export the real MODIS monthly table, and run the full multi-source model comparison.
2. Obtain geocoded fire incident records from the Lagos State Fire and Rescue Service, with suitable permission and metadata, and document their time coverage and completeness.
3. Add a time-based evaluation that tests later years while preserving spatial separation where sample sizes allow.
4. Add verified perimeters and local land-cover, fuel, topography, roads, and settlement context when suitable data are available.
5. Calibrate model scores on independent Lagos incident records before describing them as probabilities.
6. Record exact dependency versions, source retrieval dates, data checksums, and model run identifiers for a formal research release.

## 14. Suggested factual narrative for later reports or slides

> The current saved baseline trains HistGradientBoosting on Global Fire Atlas monthly fire labels and ERA5 weather for Nigeria's native 0.25-degree grid, compares it with Random Forest using five spatial folds, and maps the selected model's relative historical scores to Lagos. The proposal-complete feature set adds MODIS NDVI, EVI, and day/night land-surface temperature after the Earth Engine export is prepared. Cite the current `outputs/metrics.json` and state which feature set produced the results.

This narrative can be shortened for an abstract or adapted for slides, but keep the scale and validation limits attached to the results.

## 15. Instructions for a future chatbot

When using a chatbot to prepare a report or slides, provide this file plus any requested output artifacts. Ask it to:

- Treat this file as the project reference and check claims against the underlying saved metrics and code when needed.
- Separate measured results, interpretation, assumptions, and recommendations.
- Preserve the distinction between a satellite fire proxy and a confirmed fire-service incident.
- Describe Lagos outputs as native 0.25-degree model cells clipped to the Lagos State boundary.
- State that the risk scores are relative historical model scores, not calibrated probabilities or forecasts.
- Cite the Global Fire Atlas, MODIS, Open-Meteo, and boundary sources when using their data.
- Never invent local fire counts, model results, causes, data sources, or operational performance.

Suggested prompt:

> Use `PROJECT_KNOWLEDGE_BASE.md` as the factual reference for this project. Draft [a report section / slide outline / speaker notes] for [audience and length]. Describe the current MODIS-enabled workflow and use regenerated metrics only. Keep measured results separate from interpretation. Cite the listed sources and flag any claim that cannot be verified from the project files.
