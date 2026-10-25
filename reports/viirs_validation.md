# Independent Lagos VIIRS validation

## Design

The HistGradientBoosting classifier was trained on Global Fire Atlas monthly fire labels from 2003–2016. This later-period evaluation applies the frozen model to Lagos predictors for 2017–2024 and compares each cell-month score with NASA VIIRS VNP14A1 active-fire detections. VIIRS confidence classes 8–9 are positive detections. MODIS and VIIRS summaries were aggregated remotely in Earth Engine; only the three Lagos model cells and monthly summaries were returned.

## Results

- Cell-month records: 284 across 3 Lagos cells.
- VIIRS fire-positive cell-months: 185 (65.14%).
- ROC AUC: 0.8563472563472565.
- Average precision: 0.8944812650267525.
- Brier score: 0.3962.
- At threshold 0.50: precision 0.938, recall 0.324.

## Source data

- NASA VIIRS VNP14A1 Earth Engine Data Catalog: https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP14A1
- MODIS MOD13Q1 and MOD11A2: documented in the project README.
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api

The result evaluates later-year independent-sensor agreement for only three coarse Lagos grid cells. It should be read as a local satellite-proxy holdout and kept separate from service-verified incident evaluation.
