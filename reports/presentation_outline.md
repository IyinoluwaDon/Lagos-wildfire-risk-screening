# Presentation Outline

## Slide 1 — Lagos Wildfire Risk Screening
SIWES geospatial ML capstone. Identify where historical fire regimes and climate indicate higher vegetation-fire potential around Lagos.

## Slide 2 — Problem and Lagos context
Urban expansion meets vegetation and peri-urban open space. This is a regional screening question; operational use requires local incident records.

## Slide 3 — Data
Global Fire Atlas monthly fire-number rasters (2003–2016) and ERA5 reanalysis. Nigeria-wide training provides fire-regime context for Lagos inference; weather is assigned from four regional anchors.

## Slide 4 — Data preparation
Aggregate source pixels to a 0.5-degree grid; align monthly fire labels and weather; engineer seasonality, previous-month rainfall, and dryness.

## Slide 5 — Model and validation
Class-weighted Random Forest. GroupKFold by 2-degree spatial blocks to measure transfer to unseen regions.

## Slide 6 — Results
Show spatial ROC AUC, average precision, Brier score, and fold variability. Explain why accuracy is misleading for rare events.

## Slide 7 — Lagos screening map
Show outputs/maps/lagos_wildfire_risk.png. Interpret colour as relative rank within the map only.

## Slide 8 — Model explanation
Show outputs/figures/model_explainability.png. Discuss predictor importance and seasonal profile.

## Slide 9 — Limitations and next steps
Historical label period ends in 2016; satellite-derived fire proxy; coarse grid. Validate against Lagos fire-service records and add land cover and newer imagery.

## Slide 10 — Demonstration
Run python run.py and show generated maps, metrics, explanations, model artifact, and logs.
