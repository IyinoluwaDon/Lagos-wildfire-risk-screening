"""Read-only consistency checks for the historical saved screening outputs."""
from pathlib import Path
import json
import csv
import math
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score

def verify(root):
    metrics=json.loads((root/"outputs/metrics.json").read_text())
    matrix=metrics["confusion_matrix"]
    assert sum(sum(row) for row in matrix)==metrics["records"],"Confusion matrix count mismatch"
    tn,fp=matrix[0];fn,tp=matrix[1]
    expected={"precision_at_0_5":tp/(tp+fp),"recall_at_0_5":tp/(tp+fn),"f1_at_0_5":2*tp/(2*tp+fp+fn),"positive_rate":(tp+fn)/metrics["records"]}
    for name,value in expected.items():assert math.isclose(metrics[name],value,rel_tol=1e-10),name
    for name in ["roc_auc_mean","average_precision_mean","brier_mean"]:assert 0<=metrics[name]<=1,name
    with (root/"data/processed/lagos_seasonal_profile.csv").open() as f:rows=list(csv.DictReader(f))
    assert sorted(int(row["month"]) for row in rows)==list(range(1,13))
    assert all(math.isfinite(float(row["risk_probability"])) and 0<=float(row["risk_probability"])<=1 for row in rows)
    viirs=json.loads((root/"outputs/viirs_validation_metrics.json").read_text())
    holdout=pd.read_csv(root/"data/processed/lagos_viirs_validation.csv")
    required={"latitude","longitude","year","month","viirs_fire","predicted_probability","ndvi","evi","lst_day_c","lst_night_c","temp_c","rain_mm","rain_lag_mm","wind_kmh","humidity","dryness","month_sin","month_cos"}
    assert required.issubset(holdout.columns),"VIIRS holdout is missing model or target columns"
    assert len(holdout)==viirs["records"]
    assert not holdout.duplicated(["latitude","longitude","year","month"]).any()
    assert holdout.year.min()==2017 and holdout.year.max()==2024
    assert holdout[["latitude","longitude"]].drop_duplicates().shape[0]==3
    assert holdout["viirs_fire"].nunique()==2
    assert holdout["viirs_fire"].sum()==viirs["positive_records"]
    assert math.isclose(holdout["viirs_fire"].mean(),viirs["positive_rate"],rel_tol=1e-10)
    y=holdout["viirs_fire"].astype(int); probability=holdout["predicted_probability"].astype(float); predicted=probability>=0.5
    assert probability.map(math.isfinite).all() and probability.between(0,1).all()
    assert math.isclose(roc_auc_score(y,probability),viirs["roc_auc"],rel_tol=1e-10)
    assert math.isclose(average_precision_score(y,probability),viirs["average_precision"],rel_tol=1e-10)
    assert math.isclose(brier_score_loss(y,probability),viirs["brier_score"],rel_tol=1e-10)
    assert confusion_matrix(y,predicted,labels=[0,1]).tolist()==viirs["confusion_matrix_at_0_5"]
    assert math.isclose(precision_score(y,predicted,zero_division=0),viirs["precision_at_0_5"],rel_tol=1e-10)
    assert math.isclose(recall_score(y,predicted,zero_division=0),viirs["recall_at_0_5"],rel_tol=1e-10)
    assert (root/"reports/viirs_validation.md").is_file()
    return {"records":metrics["records"],"folds":metrics["folds"],"seasonal_months":len(rows),"viirs_holdout_records":len(holdout),"viirs_holdout_years":"2017–2024","checks":"passed"}

if __name__=="__main__":print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
