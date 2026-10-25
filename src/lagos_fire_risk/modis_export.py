"""Export monthly MODIS vegetation and land-surface temperature features from Earth Engine.

Authenticate with ``earthengine authenticate`` first. The export creates a CSV in
Google Drive; download it to ``data/processed/modis_monthly.csv`` before running
the full fire-risk pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import Point, box, mapping
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "raw" / "nigeria_state_polygons.json"
YEARS = range(2003, 2017)
GRID_DEGREES = 0.25
BOUNDS = (2.5, 4.0, 15.0, 14.0)
SELECTORS = ["cell_id", "latitude", "longitude", "year", "month", "ndvi", "evi", "lst_day_c", "lst_night_c"]


def _grid_features(ee):
    if not STATE_PATH.exists():
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://gist.githubusercontent.com/segebee/ff4a2f75be560462c369/raw/nigeria_state_polygons.json"
        response = requests.get(url, timeout=60, headers={"User-Agent": "lagos-wildfire-risk-capstone/1.0 research"})
        response.raise_for_status()
        STATE_PATH.write_bytes(response.content)
    from shapely.geometry import Polygon

    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    polygons = [Polygon([(lon, lat) for lat, lon in coords]) for coords in raw.values() if len(coords) >= 4]
    country = unary_union(polygons)
    features = []
    step = GRID_DEGREES
    lat0, lat1 = BOUNDS[1], BOUNDS[3]
    lon0, lon1 = BOUNDS[0], BOUNDS[2]
    for lat in np.arange(lat0 + step / 2, lat1, step):
        for lon in np.arange(lon0 + step / 2, lon1, step):
            center = Point(float(lon), float(lat))
            if not country.covers(center):
                continue
            cell = box(lon - step / 2, lat - step / 2, lon + step / 2, lat + step / 2).intersection(country)
            if cell.is_empty:
                continue
            features.append(ee.Feature(
                ee.Geometry(mapping(cell)),
                {"cell_id": f"{lat:.3f}_{lon:.3f}", "latitude": round(float(lat), 3), "longitude": round(float(lon), 3)},
            ))
    if not features:
        raise ValueError("No Nigeria grid cells were created for MODIS aggregation.")
    return ee.FeatureCollection(features)


def _monthly_collection(ee, grid):
    vegetation = ee.ImageCollection("MODIS/061/MOD13Q1")
    temperature = ee.ImageCollection("MODIS/061/MOD11A2")
    monthly = []
    for year in YEARS:
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
            composite = vi.addBands(lst)
            stats = composite.reduceRegions(
                collection=grid,
                reducer=ee.Reducer.mean(),
                scale=1000,
                tileScale=4,
            )
            stats = stats.map(lambda feature: feature.set({"year": year, "month": month}))
            monthly.append(stats)

    return ee.FeatureCollection(monthly).flatten()


def start_export(project: str, description: str, drive_folder: str) -> str:
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Install project requirements to use the MODIS Earth Engine exporter.") from exc
    ee.Initialize(project=project)
    grid = _grid_features(ee)
    collection = _monthly_collection(ee, grid)
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=drive_folder,
        fileNamePrefix=description,
        fileFormat="CSV",
        selectors=SELECTORS,
    )
    task.start()
    return task.id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Earth Engine-enabled Google Cloud project ID")
    parser.add_argument("--description", default="lagos_wildfire_modis_monthly_2003_2016")
    parser.add_argument("--drive-folder", default="Lagos_Wildfire_MODIS")
    args = parser.parse_args()
    task_id = start_export(args.project, args.description, args.drive_folder)
    print(f"Earth Engine export started: {task_id}")
    print(f"When it completes, download {args.description}.csv to data/processed/modis_monthly.csv")


if __name__ == "__main__":
    main()
