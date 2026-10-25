# Lagos Wildfire Risk Screening

> These saved results describe the earlier 0.5° ERA5-only Random Forest baseline. The current source pipeline adds monthly MODIS NDVI/EVI/LST features, compares Random Forest with HistGradientBoosting, uses the native 0.25° fire grid, and clips output cells to Lagos. Run the MODIS export and `python run.py` to regenerate this report with current results.

## Executive summary

This project builds an interpretable spatial fire-occurrence screen for Lagos State and its peri-urban edge. It combines Global Fire Atlas ignition counts with ERA5 reanalysis, trains on Nigeria-wide fire patterns, and evaluates generalisation by holding out spatial blocks.

**The Lagos map shows relative historical screening scores, not a real-time warning or a calibrated probability of an incident.** Satellite fire records can miss small burns and may include thermal anomalies. Local incident validation is required before operational use.

## Study design

- Presentation area: Lagos State with nearby peri-urban context; boundary source: Nigeria state polygon data from a public Gist.
- Training region: Nigeria land grid cells within approximately 2.75–15°E and 4.25–14°N.
- Fire labels: Global Fire Atlas monthly fire-number rasters (2003–2016), aggregated to 0.5° modelling cells.
- Weather predictors: Open-Meteo ERA5 daily temperature, precipitation, and wind summarized monthly from four regional anchors.
- Validation: GroupKFold spatial cross-validation by 2° blocks; grid cells in a held-out block are excluded from training.
- Model: class-weighted Random Forest; permutation importance and seasonal profiles support interpretation.

## Validation results

- Training records: 50,568; fire-positive grid-month prevalence: 29.10%.
- Spatial ROC AUC: 0.935 (fold SD 0.029).
- Spatial average precision: 0.857; Brier score: 0.092.
- At fixed threshold 0.50: precision 0.765, recall 0.821, F1 0.792.
- Read ROC AUC alongside average precision and Brier because fire months are rare; accuracy alone is not informative.

## Reading the Lagos map

The map ranks 3 model cells by mean model probability over the 12 calendar months. Percentiles are relative to the displayed Lagos-area grid. The surface represents a historical climate/fire-regime baseline, not a future-day forecast.

### Seasonal profile

| Month | Mean model probability |
|---|---:|
| Jan | 0.8845 |
| Feb | 0.7626 |
| Mar | 0.3830 |
| Apr | 0.0228 |
| May | 0.0000 |
| Jun | 0.0001 |
| Jul | 0.0017 |
| Aug | 0.0001 |
| Sep | 0.0001 |
| Oct | 0.0005 |
| Nov | 0.1659 |
| Dec | 0.6444 |

## Limitations and next steps

1. Compare satellite labels with geocoded Lagos State Fire and Rescue Service records; the current target is a satellite-derived fire proxy.
2. Improve spatial detail with Sentinel-2 fuel/moisture and land-cover layers, settlement/road variables, and verified fire perimeters.
3. Treat probabilities as relative until a time-based independent validation and local calibration are completed.
4. The fitted baseline uses 2003–2016 fire history; a full recent archive is needed to update the model responsibly.

## Sources

- Global Fire Atlas updated record (2002–2024): https://zenodo.org/records/14065246
- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api
- Nigeria state polygon data (Gist): https://gist.github.com/segebee/ff4a2f75be560462c369

## Reproducibility

Run python run.py from the project root. Raw inputs, processed tables, model, figures, maps, metrics, report, and logs are stored under this project. The data archive is excluded from Git; its source URL and retrieval logic are recorded in the code and README.
