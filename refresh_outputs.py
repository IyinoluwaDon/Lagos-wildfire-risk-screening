import json
import joblib
import pandas as pd
from src.lagos_fire_risk.pipeline import setup_logging, make_outputs, write_report, PROCESSED, OUT

log=setup_logging()
artifact=joblib.load(OUT/'wildfire_model.joblib')
model=artifact['model']
df=pd.read_csv(PROCESSED/'training_features.csv')
metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
risk,seasonal,boundary=make_outputs(model,df,log)
write_report(metrics,df,risk,seasonal,boundary)
