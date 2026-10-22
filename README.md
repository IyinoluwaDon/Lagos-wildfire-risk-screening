# Lagos Wildfire Risk Screening

A reproducible geospatial ML capstone built from Proposal Project 3. It fuses historical fire ignitions and weather reanalysis, performs spatial validation, explains the fitted model, and produces a Lagos screening map, report, metrics, and saved estimator.

## One-command run

1. Use Python 3.10 or newer.
2. From the project folder:

   powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python run.py

No Earth Engine account or API token is required. First run downloads the public Global Fire Atlas monthly archive (about 155 MB) and queries ERA5 via Open-Meteo. Weather is cached locally.

## View the saved results

Install the dashboard dependencies, then launch the read-only dashboard from the project folder:

```powershell
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

The dashboard displays saved validation metrics, the Lagos screening map, seasonal scores, predictor importance, fold results, artifact availability, and recent logs. It does not download data or rerun the model. Use `python run.py` to refresh the pipeline outputs.

## Study design

- Presentation area: Lagos State.
- Training region: Nigeria land grid. Weather features are assigned to the nearest of four ERA5 anchors.
- Fire labels: Global Fire Atlas monthly fire-number rasters, aggregated from 0.25° to 0.5° cells, 2003–2016.
- Weather: ERA5 daily temperature, rainfall, and wind summarized by month.
- Validation: spatial GroupKFold using 2° blocks; neighbouring grid cells do not cross between training and test blocks.
- Classifier: class-weighted Random Forest. Permutation importance and seasonal profiles support interpretation.

## Stored artifacts

- data/raw: source archive and extracted GeoTIFFs.
- data/processed: monthly weather cache, training table, Lagos grid, and seasonal profile.
- outputs/maps: presentation-ready Lagos map.
- outputs/figures: model explanation figure.
- outputs/metrics.json and outputs/spatial_validation_folds.csv: validation results.
- outputs/wildfire_model.joblib: serialized model.
- outputs/logs/pipeline.log: run log.
- reports/project_report.md and reports/presentation_outline.md: findings and presentation structure.
- PROJECT_KNOWLEDGE_BASE.md: full project rationale, data, method, results, limitations, and guidance for future report/slide drafting.
- presentation/Lagos_Wildfire_Risk_Screening.pptx: editable presentation deck; preview images are saved in outputs/presentation_previews.

## Sources and attribution

- Global Fire Atlas updated record: https://zenodo.org/records/14065246
- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api
- Nigeria state polygons (Gist): https://gist.github.com/segebee/ff4a2f75be560462c369

## How to interpret the map

The map displays mean modelled historical fire-occurrence scores by coarse grid cell, ranked as percentiles within the Lagos display area. These are relative screening values, not calibrated probabilities of an incident, operational warnings, or property-level risk. Satellite-derived labels are proxies and need validation against local incident records.

## Known limits

The fitted label period is 2003–2016 and the modelling grid is 0.5 degrees. Use local records and newer satellite data to refresh it. The model is exploratory until locally validated.
