"""Evaluate the saved historical model against later independent VIIRS fire detections.

This is a Lagos-only temporal/sensor holdout: the classifier is trained on
Global Fire Atlas labels from 2003–2016, while the evaluation target is the
NASA VIIRS VNP14A1 active-fire product for 2017–2024. Earth Engine computes
monthly MODIS predictors and VIIRS labels remotely; only the 3 × 96 monthly
cell summaries are returned to this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import ee
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lagos_fire_risk.pipeline import OUT, PROCESSED


START_YEAR = 2017
END_YEAR = 2024
CELL_DEGREES = 0.25
WEATHER_ANCHOR = (5.75, 3.5)
WEATHER_VARIABLES = (
    "temperature_2m_mean,precipitation_sum,wind_speed_10m_max,"
    "relative_humidity_2m_mean"
)


def _gee_grid(ee_module, risk: pd.DataFrame):
    features = []
    for row in risk.itertuples():
        half = CELL_DEGREES / 2
        geometry = ee_module.Geometry.Rectangle(
            [row.longitude - half, row.latitude - half,
             row.longitude + half, row.latitude + half],
            proj="EPSG:4326",
            geodesic=False,
        )
        features.append(ee_module.Feature(geometry, {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
        }))
    return ee_module.FeatureCollection(features)


def _earth_engine_rows(project: str, risk: pd.DataFrame) -> pd.DataFrame:
    ee.Initialize(project=project)
    grid = _gee_grid(ee, risk)
    fire_collection = ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
    vegetation = ee.ImageCollection("MODIS/061/MOD13Q1")
    temperature = ee.ImageCollection("MODIS/061/MOD11A2")
    monthly = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, "month")

            def prepare_vi(image):
                qa = image.select("DetailedQA")
                good = qa.bitwiseAnd(3).lte(1)
                return image.select(["NDVI", "EVI"], ["ndvi", "evi"]).multiply(0.0001).updateMask(good)

            def prepare_lst(image):
                day_good = image.select("QC_Day").bitwiseAnd(3).lte(1)
                night_good = image.select("QC_Night").bitwiseAnd(3).lte(1)
                day = image.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst_day_c").updateMask(day_good)
                night = image.select("LST_Night_1km").multiply(0.02).subtract(273.15).rename("lst_night_c").updateMask(night_good)
                return day.addBands(night)

            vi = vegetation.filterDate(start, end).map(prepare_vi).mean()
            lst = temperature.filterDate(start, end).map(prepare_lst).mean()
            predictors = vi.addBands(lst).reduceRegions(
                collection=grid,
                reducer=ee.Reducer.mean(),
                scale=1000,
                tileScale=2,
            )

            active = (
                fire_collection.filterDate(start, end)
                .select("FireMask")
                .map(lambda image: image.gte(8).rename("active"))
                .max()
            )
            labels = active.reduceRegions(
                collection=grid,
                reducer=ee.Reducer.sum(),
                scale=1000,
                tileScale=2,
            )

            def combine(feature):
                matching = labels.filter(
                    ee.Filter.And(
                        ee.Filter.eq("latitude", feature.get("latitude")),
                        ee.Filter.eq("longitude", feature.get("longitude")),
                    )
                ).first()
                fire_pixels = ee.Number(ee.Algorithms.If(matching, matching.get("sum"), 0))
                return feature.set({
                    "year": year,
                    "month": month,
                    "viirs_active_fire_pixels": fire_pixels,
                    "viirs_fire": fire_pixels.gt(0).int(),
                })

            monthly.append(predictors.map(combine))

    collection = ee.FeatureCollection(monthly).flatten()
    response = collection.getInfo()
    records = [feature["properties"] for feature in response["features"]]
    result = pd.DataFrame.from_records(records)
    expected = len(risk) * ((END_YEAR - START_YEAR + 1) * 12)
    if len(result) != expected:
        raise ValueError(f"Expected {expected} VIIRS/MODIS cell-month rows; received {len(result)}.")
    if result.duplicated(["latitude", "longitude", "year", "month"]).any():
        raise ValueError("Earth Engine returned duplicate cell/month rows.")
    return result


def _weather_rows() -> pd.DataFrame:
    params = {
        "latitude": WEATHER_ANCHOR[0],
        "longitude": WEATHER_ANCHOR[1],
        "start_date": f"{START_YEAR - 1}-12-01",
        "end_date": f"{END_YEAR}-12-31",
        "daily": WEATHER_VARIABLES,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=180,
        headers={"User-Agent": "lagos-wildfire-risk-capstone/1.0 research"},
    )
    response.raise_for_status()
    daily = pd.DataFrame(response.json()["daily"])
    daily["date"] = pd.to_datetime(daily.pop("time"), errors="raise")
    daily["year"] = daily.date.dt.year
    daily["month"] = daily.date.dt.month
    return daily.groupby(["year", "month"], as_index=False).agg(
        temp_c=("temperature_2m_mean", "mean"),
        rain_mm=("precipitation_sum", "sum"),
        wind_kmh=("wind_speed_10m_max", "mean"),
        humidity=("relative_humidity_2m_mean", "mean"),
    )


def _score(viirs: pd.DataFrame, model_artifact: dict) -> pd.DataFrame:
    weather = _weather_rows()
    frame = viirs.merge(weather, on=["year", "month"], how="left", validate="many_to_one")
    frame = frame.sort_values(["latitude", "longitude", "year", "month"]).reset_index(drop=True)
    frame["rain_lag_mm"] = frame.groupby(["latitude", "longitude"]).rain_mm.shift(1)
    first_month = frame.rain_lag_mm.isna()
    december_2016 = weather.loc[(weather.year == START_YEAR - 1) & (weather.month == 12), "rain_mm"]
    if december_2016.empty:
        raise ValueError("Open-Meteo did not return December 2016 rainfall for the initial lag feature.")
    frame.loc[first_month, "rain_lag_mm"] = float(december_2016.iloc[0])
    frame["month_sin"] = np.sin(2 * np.pi * frame.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.month / 12)
    frame["dryness"] = frame.temp_c / (1 + frame.rain_lag_mm.clip(lower=0))
    features = model_artifact["features"]
    complete = frame.dropna(subset=features + ["viirs_fire"]).copy()
    if complete.empty:
        raise ValueError("No complete VIIRS validation rows remain after joining MODIS and ERA5.")
    complete["predicted_probability"] = model_artifact["model"].predict_proba(complete[features])[:, 1]
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Earth Engine-enabled Google Cloud project ID")
    args = parser.parse_args()
    risk = pd.read_csv(PROCESSED / "lagos_risk_grid.csv")
    artifact = joblib.load(OUT / "wildfire_model.joblib")
    validation = _score(_earth_engine_rows(args.project, risk), artifact)

    labels = validation.viirs_fire.astype(int)
    predictions = validation.predicted_probability
    binary = predictions >= 0.5
    matrix = confusion_matrix(labels, binary, labels=[0, 1]).tolist()
    metrics = {
        "evaluation": "Lagos 2017–2024 later-period, independent-sensor holdout",
        "training_labels": "Global Fire Atlas 2003–2016",
        "evaluation_labels": "NASA VIIRS VNP14A1 nominal/high confidence active fire detections (FireMask >= 8)",
        "model": artifact["algorithm"],
        "records": int(len(validation)),
        "positive_records": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, predictions)) if labels.nunique() == 2 else None,
        "average_precision": float(average_precision_score(labels, predictions)) if labels.sum() else None,
        "brier_score": float(brier_score_loss(labels, predictions)),
        "precision_at_0_5": float(precision_score(labels, binary, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, binary, zero_division=0)),
        "confusion_matrix_at_0_5": matrix,
        "interpretation": "This compares future-year model scores with an independent satellite sensor in three Lagos grid cells; it is a local satellite-proxy evaluation, not a fire-service incident audit.",
    }
    output = PROCESSED / "lagos_viirs_validation.csv"
    validation.to_csv(output, index=False)
    metrics_path = OUT / "viirs_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "# Independent Lagos VIIRS validation",
        "",
        "## Design",
        "",
        "The HistGradientBoosting classifier was trained on Global Fire Atlas monthly fire labels from 2003–2016. This later-period evaluation applies the frozen model to Lagos predictors for 2017–2024 and compares each cell-month score with NASA VIIRS VNP14A1 active-fire detections. VIIRS confidence classes 8–9 are positive detections. MODIS and VIIRS summaries were aggregated remotely in Earth Engine; only the three Lagos model cells and monthly summaries were returned.",
        "",
        "## Results",
        "",
        f"- Cell-month records: {metrics['records']:,} across {validation[['latitude', 'longitude']].drop_duplicates().shape[0]} Lagos cells.",
        f"- VIIRS fire-positive cell-months: {metrics['positive_records']:,} ({metrics['positive_rate']:.2%}).",
        f"- ROC AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'not defined (only one label class)' }.",
        f"- Average precision: {metrics['average_precision'] if metrics['average_precision'] is not None else 'not defined (no positive labels)' }.",
        f"- Brier score: {metrics['brier_score']:.4f}.",
        f"- At threshold 0.50: precision {metrics['precision_at_0_5']:.3f}, recall {metrics['recall_at_0_5']:.3f}.",
        "",
        "## Source data",
        "",
        "- NASA VIIRS VNP14A1 Earth Engine Data Catalog: https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP14A1",
        "- MODIS MOD13Q1 and MOD11A2: documented in the project README.",
        "- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api",
        "",
        "The result evaluates later-year independent-sensor agreement for only three coarse Lagos grid cells. It should be read as a local satellite-proxy holdout and kept separate from service-verified incident evaluation.",
    ]
    report_path = ROOT / "reports" / "viirs_validation.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "csv": str(output), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
