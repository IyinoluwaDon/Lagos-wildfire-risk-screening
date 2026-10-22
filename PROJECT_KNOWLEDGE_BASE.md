# Project Knowledge Base: Lagos Wildfire Risk Screening

This document is the detailed reference for explaining, reporting, and presenting this project. It describes the implementation and saved results in the project folder. If a slide, an older report, or a generated figure conflicts with the code or saved data described here, verify the underlying code and data before repeating the claim.

## 1. Project at a glance

**Project name:** Lagos Wildfire Risk Screening  
**Project type:** SIWES geospatial machine learning capstone  
**Proposal reference:** Project 3, “Wildfire Risk Prediction via Multi-Source Data Fusion,” from the supplied SIWES proposal  
**Study area:** Lagos State, Nigeria  
**Historical fire-label period:** 2003–2016  
**Canonical project folder:** `C:\Projects\lagos-wildfire-risk-screening`

The project combines historical satellite-derived fire-number data with monthly climate summaries. It trains a classifier on a Nigeria-wide land grid, evaluates it using geographically separated folds, and scores a small grid inside Lagos State. Its purpose is to demonstrate a reproducible geospatial ML workflow and produce a **historical, exploratory screening map**.

The phrase “risk screening” is used as a project label. The map is not an operational fire warning, a forecast, a confirmed incident map, or a property-level risk assessment. Its scores are model outputs based on coarse historical inputs.

## 2. Why this project

Vegetation fires can affect settlements, infrastructure, health, and ecosystems. Lagos is a useful presentation area because it combines dense urban development with peri-urban and vegetated areas. A geospatial model provides a way to bring together fire history, climate, location, and maps in one analysis.

The practical question is:

> Where do historical satellite fire patterns and broad monthly climate conditions indicate comparatively higher vegetation-fire occurrence scores within the Lagos display area?

This question is deliberately narrower than asking where the next fire will occur. The available inputs do not include recent, local fire-service incident records, detailed fuels, current weather forecasts, or verified fire perimeters. The project therefore demonstrates a screening method; it does not establish a deployable public-safety service.

## 3. What the project does

1. Retrieves the public Global Fire Atlas monthly fire-number raster archive if it is not already present.
2. Reads monthly rasters for 2003–2016 and aggregates source pixels into 0.5-degree training cells.
3. Creates a complete Nigeria land-cell/month table so months with no raster detections are represented as negative examples.
4. Retrieves ERA5 reanalysis weather summaries through the Open-Meteo Historical Weather API, using four regional locations, then caches the results.
5. Builds time, rainfall, dryness, wind, temperature, and location features.
6. Evaluates a class-weighted Random Forest with five spatially grouped folds using 2-degree blocks.
7. Fits a final model on the full historical training table and saves it.
8. Scores three 0.25-degree display-cell centers that fall inside the available Lagos state polygon, using 12-month climate averages.
9. Saves a map, model-explanation figure, CSV tables, metrics, a report, and a run log.

## 4. Data sources and their roles

### Global Fire Atlas

- **Dataset:** Global Fire Atlas monthly fire-number GeoTIFF summaries.
- **Source record:** [Global Fire Atlas on Zenodo](https://zenodo.org/records/14065246).
- **Download used by the code:** `https://zenodo.org/records/14065246/files/GeoTIFF_Qdeg_monthly_summaries.zip`.
- **Time used:** 2003–2016 inclusive, or 168 calendar months per grid cell.
- **Source resolution:** 0.25 degrees.
- **Model resolution:** 0.5 degrees. Source values are summed into the enclosing model cell.
- **Role:** Provides the satellite-derived fire-number observations used to make a binary monthly target.

The target is `fire = 1` when the aggregated monthly fire-number value is greater than zero; it is `fire = 0` otherwise. It is a satellite-derived proxy for fire occurrence. It is not a complete inventory of fires or a record confirmed by a local fire service. Small fires may be missed, and thermal detections may not always correspond to the intended vegetation-fire event.

### Weather reanalysis

- **Provider/API:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), using ERA5 reanalysis.
- **Weather variables retrieved daily:** mean 2 m temperature, precipitation sum, maximum 10 m wind speed, and mean relative humidity.
- **Weather features used by the model:** temperature, precipitation, wind speed, and derived rainfall features. Humidity is retrieved and cached, but it is **not** in the model feature list.
- **Sampling locations:** four regional anchors: southwest `(5.75°N, 3.5°E)`, central `(8.5°N, 6.5°E)`, northwest `(11.5°N, 6.5°E)`, and northeast `(12.5°N, 11.5°E)`.
- **Assignment:** each 0.5-degree cell is assigned to its nearest anchor using latitude-adjusted longitude distance.
- **Role:** Provides broad monthly climate conditions at much lower cost and query volume than requesting weather separately for every grid cell.

The weather anchors are a coarse approximation. They do not reproduce local Lagos microclimates or cell-level weather variation.

### Nigeria state boundaries

- **Source used:** [Public Nigeria state polygon data on Gist](https://gist.github.com/segebee/ff4a2f75be560462c369).
- **Role:** Defines Nigeria land cells for the training table and clips the Lagos display grid to the Lagos state polygon.
- **Coordinate handling:** The code interprets the source coordinate pairs as latitude/longitude and converts them to the longitude/latitude order expected by the geometry library.

The boundary file is downloaded and cached as `data/raw/nigeria_state_polygons.json` when needed. For formal publication, confirm the source’s licensing, provenance, and boundary accuracy.

## 5. Study area, sample, and spatial scales

The training grid is built from Nigeria state polygons. Candidate cell centers are spaced every 0.5 degrees, approximately across 2.75–15°E and 4.25–14°N; a center is included when it is covered by the union of state polygons.

- **Nigeria training land cells:** 301.
- **Time steps per cell:** 168 months (January 2003 through December 2016).
- **Training records:** 50,568 cell-month rows (`301 × 168`).
- **Fire-positive rate:** 29.10% of rows.
- **Lagos display cells:** 3 cell centers at 0.25-degree spacing.

The Lagos display grid is finer than the model grid. The final classifier is trained at 0.5 degrees, then evaluated at 0.25-degree cell centers inside Lagos. Those three values are **interpolated spatial screening estimates from the coarser model**, not observations or independently validated 0.25-degree predictions. The very small number of displayed cells also limits how much spatial ranking can be inferred within Lagos.

## 6. Feature construction

The model uses the following nine features, in the order defined in `src/lagos_fire_risk/pipeline.py`:

| Feature | Meaning |
|---|---|
| `latitude` | Center latitude of the 0.5-degree training cell. |
| `longitude` | Center longitude of the 0.5-degree training cell. |
| `month_sin` | Sine encoding of calendar month to represent season cyclically. |
| `month_cos` | Cosine encoding of calendar month to represent season cyclically. |
| `temp_c` | Monthly average of daily mean 2 m temperature, in °C. |
| `rain_mm` | Monthly total precipitation, in mm. |
| `rain_lag_mm` | Previous month’s rainfall for the same cell; for the first available month it is set to that month’s rainfall. |
| `wind_kmh` | Monthly average of daily maximum 10 m wind speed, in km/h. |
| `dryness` | Derived index `temp_c / (1 + rain_lag_mm)`. Larger values indicate warmer conditions with less prior-month rainfall under this simple formula. |

Humidity is downloaded as part of the weather response but is not used as a predictor. The project does not include land cover, vegetation fuel, roads, population, settlement, slope, or local fire response data.

## 7. Model and validation

### Classifier

The final model is a scikit-learn `RandomForestClassifier` with 400 trees, maximum depth 12, minimum leaf size 3, `class_weight='balanced_subsample'`, `max_features=0.8`, all available CPU cores, and random seed 42. It is fit on all 50,568 training rows and saved at `outputs/wildfire_model.joblib` along with the feature names and training-scope metadata.

### Spatial cross-validation

The code assigns each cell to a group using its 2-degree longitude and latitude blocks, then uses five-fold `GroupKFold`. All records from a spatial group are held together, so the same group’s cells/months do not appear in both a fold’s training and test partitions. This is a stronger geographic transfer check than a random row split.

This is spatial validation within the historical Nigeria dataset. It is **not** a time-based test on later years, an independent Lagos incident validation, or a calibration test. The final Lagos estimates are a geographic transfer beyond the validation result; the Nigeria-wide metrics do not prove local Lagos accuracy.

### Saved validation results

The AUC, average precision, and Brier score below are arithmetic means of the five fold metrics. Precision, recall, F1, and the confusion matrix use pooled out-of-fold predictions at threshold 0.50.

| Metric | Result | Plain-language reading |
|---|---:|---|
| ROC AUC | 0.9346 (fold SD 0.0285) | Ability to rank a positive row above a negative row across thresholds. |
| Average precision | 0.8570 | Precision-recall summary; compare it with the 0.291 positive prevalence. |
| Brier score | 0.0916 | Mean squared probability error; lower is better. |
| Precision at 0.50 | 0.7646 | Of rows called positive at this threshold, about 76% were positive in the pooled held-out rows. |
| Recall at 0.50 | 0.8213 | Of positive held-out rows, about 82% were identified at this threshold. |
| F1 at 0.50 | 0.7919 | Harmonic mean of precision and recall at that threshold. |

Pooled confusion matrix at threshold 0.50, with actual classes as rows and predicted classes as columns:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 32,129 | 3,722 |
| Actual 1 | 2,630 | 12,087 |

These metrics describe historical fire-proxy classification on spatially held-out Nigeria blocks. They should not be reported as the expected accuracy of an operational Lagos warning system.

## 8. Lagos outputs and interpretation

The saved grid is `data/processed/lagos_risk_grid.csv`:

| Cell center (latitude, longitude) | Mean 12-month model score | Relative percentile |
|---|---:|---:|
| (6.625, 3.375) | 0.2439 | 100 |
| (6.625, 3.625) | 0.2363 | 50 |
| (6.625, 3.875) | 0.2363 | 50 |

The score is the mean of the model’s 12 monthly scores at each cell. Percentiles rank the three displayed scores against one another. Tied scores receive the same percentile under the ranking implementation. The score is not a calibrated probability of a future fire. In this project, the word “risk” on the map is shorthand for a relative model score.

The saved seasonal profile is `data/processed/lagos_seasonal_profile.csv`. It is the average model score across the three display cells for each month:

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

## 9. Explainability: what is and is not established

The saved file `outputs/feature_importance.csv` reports permutation importance using average precision, with six repeats. Its leading values are:

| Feature | Mean importance | Repeat SD |
|---|---:|---:|
| `month_cos` | 0.0753 | 0.0133 |
| `rain_mm` | 0.0504 | 0.0060 |
| `wind_kmh` | 0.0087 | 0.0014 |
| `temp_c` | 0.0085 | 0.0014 |
| `dryness` | 0.0044 | 0.0003 |

**Important implementation caveat:** the code fits the final 400-tree model on the full dataset, then calculates permutation importance using a spatial block drawn from that same dataset. Because the model has already seen that block during fitting, the importance calculation is not an independent held-out explanation and may be optimistic. Use the values as exploratory model diagnostics only. Do not describe them as validated causal drivers. A stronger future analysis should fit on training blocks and calculate permutation importance on a block held out from that fit, or aggregate fold-specific held-out importances.

The figure `outputs/figures/model_explainability.png` combines that importance table with the seasonal model profile. Permutation importance measures how model performance changes when a feature is shuffled; it does not show that the feature causes fires. Correlated features can also share or mask importance.

## 10. Reproducing the project

From the project root, with Python 3.10 or newer:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

The first run downloads the Global Fire Atlas ZIP (about 155 MB) and queries Open-Meteo for four locations for each year from 2003 through 2016. The weather response is cached in `data/processed/era5_monthly.csv`; the archive and extracted rasters are retained in `data/raw/`. Subsequent runs reuse those files when present. A normal run regenerates the processed feature table, cross-validation results, final model, importance values, Lagos map, explainability figure, report, and log.

To refresh only the Lagos score/map and report from existing saved inputs and model, run:

```powershell
python refresh_outputs.py
```

To browse the current saved results in the Streamlit dashboard, run:

```powershell
streamlit run app.py
```

The dashboard is read-only with respect to the analysis: it displays saved metrics, the map, explanation charts, fold results, artifact availability, and recent logs. It does not rerun the model or make API calls. Run `python run.py` separately to regenerate project outputs.

The project has no required Earth Engine account, API token, or user-provided data file. Network access is needed on a first run when source data are absent. Package versions in `requirements.txt` use minimum version bounds rather than a fully locked environment, so exact bit-for-bit reproduction across future software releases is not guaranteed.

## 11. Project file guide

| Path | Purpose |
|---|---|
| `run.py` | Small entry point that runs the full pipeline. |
| `app.py` | Read-only Streamlit dashboard for browsing saved metrics, map, explanations, files, and logs. |
| `src/lagos_fire_risk/pipeline.py` | Data retrieval, preparation, feature engineering, model, validation, figures, map, and report generation. |
| `refresh_outputs.py` | Rebuilds Lagos outputs and report from existing cached features and saved model. |
| `config.yaml` | Project area, grid resolution, data years, random seed, spatial block size, and fold count. |
| `requirements.txt` | Python dependency minimums. |
| `.streamlit/config.toml` | Dashboard theme and headless server setting. |
| `data/raw/` | Downloaded Atlas archive, extracted monthly rasters, and state polygon file. |
| `data/processed/` | Weather cache, model input table, Lagos grid scores, and monthly score profile. |
| `outputs/metrics.json` | Machine-readable validation summary. |
| `outputs/spatial_validation_folds.csv` | Per-fold validation metrics and sample counts. |
| `outputs/feature_importance.csv` | Saved feature-importance values and repeat variability. |
| `outputs/wildfire_model.joblib` | Final fitted classifier and metadata. |
| `outputs/maps/lagos_wildfire_risk.png` | Lagos screening map. |
| `outputs/figures/model_explainability.png` | Seasonal score profile and feature importance figure. |
| `outputs/logs/pipeline.log` | Timestamped run log. |
| `outputs/presentation_previews/` | Nine PNG slide previews. |
| `reports/project_report.md` | Concise written project report. |
| `reports/presentation_outline.md` | Initial outline for presenting the project. |
| `presentation/Lagos_Wildfire_Risk_Screening.pptx` | Nine-slide presentation deck. |

Raw data, processed data, and generated `outputs/` are stored locally in the project folder and ignored by Git to avoid putting large generated files into source history. The README, source, requirements, report, outline, and presentation are available to Git.

## 12. Limitations to state clearly in a report or presentation

1. **Historical period:** fire labels end in 2016; this project does not use current fire detections.
2. **Proxy target:** satellite-derived fire-number values are not a complete, locally verified incident register.
3. **Coarse model scale:** the training cells are 0.5 degrees. The three Lagos display cells at 0.25 degrees do not create true 0.25-degree model resolution.
4. **Sparse Lagos display:** only three cell centers fall inside the state polygon at the chosen grid alignment, so detailed within-state rankings are not supportable.
5. **Coarse climate assignment:** four regional anchors cannot represent local weather differences across Nigeria or Lagos.
6. **Geographic transfer:** Nigeria spatial cross-validation does not directly validate the Lagos map or demonstrate local calibration.
7. **No temporal holdout:** the saved evaluation does not test whether a model trained on earlier years transfers to later years.
8. **Explanability limitation:** the current permutation importance block was included in the final model fit; importance is exploratory until recalculated with an independent spatial holdout.
9. **No causal claim:** predictor importance and seasonal scores describe model associations, not causes.
10. **No operational use:** do not use these outputs for dispatch, evacuation, insurance, property decisions, or public alerts without updated inputs, local validation, calibration, and expert review.

## 13. Recommended next steps

1. Obtain geocoded fire incident records from the Lagos State Fire and Rescue Service, with suitable permission and metadata, and document their time coverage and completeness.
2. Add a true time-based evaluation, such as training on earlier years and evaluating later years, while preserving spatial separation where sample sizes allow.
3. Recalculate permutation importance from spatially held-out predictions or fold-specific models.
4. Add current and suitable satellite fire products, verified fire perimeters, land cover, vegetation/fuel condition, topography, roads, and settlement context.
5. Improve Lagos spatial resolution only when inputs and local validation can support it; compare any finer grid against independent incident data.
6. Calibrate probabilities on an independent local dataset before describing any output as a probability.
7. Record exact dependency versions, source retrieval dates, data checksums, and model run identifiers for a formal research release.

## 14. Suggested factual narrative for later reports or slides

> This SIWES capstone implements a reproducible geospatial machine-learning workflow for historical wildfire screening around Lagos. It combines Global Fire Atlas monthly fire-number rasters from 2003–2016 with monthly ERA5 reanalysis summaries, creates 50,568 grid-month records across 301 Nigeria land cells, and evaluates a class-weighted Random Forest using five spatially grouped folds. The mean spatial ROC AUC is 0.935 (fold SD 0.029), average precision is 0.857, and Brier score is 0.092. The model produces a Lagos screening map for three 0.25-degree display cells using a classifier trained at 0.5 degrees. These are exploratory relative historical scores, not local incident probabilities or forecasts. The main next steps are independent Lagos incident validation, a time-based evaluation, improved spatial inputs, and held-out explainability analysis.

This narrative can be shortened for an abstract or adapted for slides, but keep the scale and validation limits attached to the results.

## 15. Instructions for a future chatbot

When using a chatbot to prepare a report or slides, provide this file plus any requested output artifacts. Ask it to:

- Treat this file as the project reference and check claims against the underlying saved metrics and code when needed.
- Separate measured results, interpretation, assumptions, and recommendations.
- Preserve the distinction between a satellite fire proxy and a confirmed fire-service incident.
- Describe the Lagos display as three quarter-degree cell centers scored by a half-degree-trained model.
- State that the risk scores are relative historical model scores, not calibrated probabilities or forecasts.
- Disclose that the present feature importance calculation is not independent of the final model fit.
- Cite the Global Fire Atlas, Open-Meteo, and boundary sources when using their data.
- Never invent local fire counts, model results, causes, data sources, or operational performance.

Suggested prompt:

> Use `PROJECT_KNOWLEDGE_BASE.md` as the factual reference for this project. Draft [a report section / slide outline / speaker notes] for [audience and length]. Keep verified results separate from interpretation and next steps. Preserve all caveats about the satellite proxy, the three-cell Lagos display, coarse input resolution, lack of time-based and local incident validation, and exploratory permutation importance. Cite the listed sources and flag any claim that cannot be verified from the project files.
