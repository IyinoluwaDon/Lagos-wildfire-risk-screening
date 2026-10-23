# Lagos Historical Fire-Proxy Screening

A spatial screening dashboard that summarizes historical satellite fire detections and related saved analyses for Lagos. It is a retrospective proxy-risk exploration, not a live wildfire forecast or emergency alert.

## What is included

- Streamlit dashboard over saved spatial outputs
- Data/output consistency validation
- Documented study design, sources and interpretation limits

## Run locally

```sh
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

## Verify

Use `python scripts/verify_outputs.py` to verify the saved confusion-matrix records, derived metrics and seasonal output consistency.

## Repository map

`app.py` serves the dashboard. `src/` and `scripts/` contain analysis and output checks. `reports/` and `presentation/` hold saved results. `VALIDATION.md` documents checks and assumptions.

## Evidence and limitations

The saved-output verifier checks 50,568 confusion-matrix records, derived metrics and twelve seasonal months. The dashboard renders from stored results.

Fire-proxy records can be affected by detection limits, cloud cover and location uncertainty. No local incident validation or independent temporal evaluation establishes predictive skill. The saved analysis is historical and is not an operational warning service.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.
