# Lagos Wildfire Risk Screening

A geospatial machine-learning study of historical fire occurrence across Nigeria, with a Lagos and peri-urban screening map. The fitted classifier learns national fire/climate patterns and summarizes their historical monthly scores over Lagos.

## Data and model

The target is monthly fire presence derived from Global Fire Atlas fire-number rasters for 2003-2016, aggregated to 0.5-degree modelling cells. Predictors combine spatial location with monthly Open-Meteo ERA5 temperature, precipitation, wind, and humidity summaries. A class-weighted Random Forest is evaluated with five-fold GroupKFold cross-validation that holds out 2-degree spatial blocks; the saved final estimator uses 400 trees.

The recorded spatial validation covers **50,568 grid-month records** (29.10% fire-positive): mean ROC AUC **0.935** (fold SD 0.029), average precision **0.857**, Brier score **0.092**, and at threshold 0.50 precision **0.765**, recall **0.821**, F1 **0.792**. These are spatial-CV metrics for the satellite-derived fire label. The Lagos visualization reports relative mean monthly screening scores from the historical 2003-2016 baseline.

## Run

```sh
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

`run.py` rebuilds the data/model/output pipeline from the documented public sources. `scripts/verify_outputs.py` checks saved metrics and seasonal summaries. `src/lagos_fire_risk/pipeline.py` contains the ingestion, feature construction, spatial validation, Random Forest training, and mapping workflow; `reports/` contains the study report.
