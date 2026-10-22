"""Read-only consistency checks for the historical saved screening outputs."""
from pathlib import Path
import json
import csv
import math

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
    return {"records":metrics["records"],"folds":metrics["folds"],"seasonal_months":len(rows),"checks":"passed"}

if __name__=="__main__":print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
