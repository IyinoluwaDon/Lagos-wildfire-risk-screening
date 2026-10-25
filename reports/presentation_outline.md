# Presentation Outline

The updated deck reflects the full MODIS + ERA5 + Global Fire Atlas model run.

## Slide 1 — Lagos Wildfire Risk Screening
SIWES geospatial ML capstone; Nigeria-wide HistGradientBoosting model with a Lagos map on the native 0.25° grid.

## Slide 2 — Study question and scope
Historical satellite-observed monthly fire occurrence; train and spatially validate across Nigeria, then summarize the Lagos cell scores.

## Slide 3 — Data inputs
Global Fire Atlas monthly fire-number rasters (2003–2016), ERA5 reanalysis, MODIS MOD13Q1 NDVI/EVI, MODIS MOD11A2 day/night LST, and open Nigeria state polygons.

## Slide 4 — Model workflow
Join monthly 0.25° predictors; engineer seasonal and lagged-rain variables; fit HistGradientBoosting; evaluate five GroupKFold folds grouped by 2° blocks; score native cells clipped to Lagos.

## Slide 5 — Spatial validation
191,976 complete Nigeria grid-month records; ROC AUC 0.944 ± 0.013, average precision 0.809, Brier 0.092; fixed-threshold precision 0.613, recall 0.873, F1 0.721.

## Slide 6 — Lagos screening map
Updated full-model map, three native 0.25° cell centers, mean cell scores 0.240, 0.233, and 0.153.

## Slide 7 — Model explanation
Full-model monthly Lagos score profile and permutation importance; seasonality, EVI, latitude, and longitude are the leading predictors.

## Slide 8 — Study scope and interpretation
Summarize historical labels, spatial scale, source fusion, Nigeria geographic-block validation, and the later Lagos VIIRS holdout.

## Slide 9 — Reproduce and extend
Run `python run.py`; saved outputs include the model, validation records, map, feature importance, report, and Streamlit app. The later 2017–2024 VIIRS holdout is complete; local incident records are the next evaluation source.
