# Saved-output verification

Run `python scripts/verify_outputs.py` to check confusion-matrix totals, positive prevalence, precision/recall/F1 consistency and the complete 12-month seasonal score series. This is a read-only artifact check, not retraining or independent validation. The full pipeline fetches external historical inputs and can be expensive. Spatial cross-validation is already recorded in outputs/metrics.json; interpretation remains historical screening rather than live warning. Feature importance uses a block included in final fitting and is exploratory.
