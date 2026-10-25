"""Read-only Streamlit dashboard for the saved Lagos wildfire screening outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"


st.set_page_config(
    page_title="Lagos Wildfire Risk Screening",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def format_metric(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


metrics = read_json(OUTPUTS / "metrics.json")
viirs_metrics = read_json(OUTPUTS / "viirs_validation_metrics.json")
risk_grid = read_csv(PROCESSED / "lagos_risk_grid.csv")
seasonal = read_csv(PROCESSED / "lagos_seasonal_profile.csv")
importance = read_csv(OUTPUTS / "feature_importance.csv")
comparison = read_csv(OUTPUTS / "model_comparison.csv")
folds = read_csv(OUTPUTS / "spatial_validation_folds.csv")

with st.sidebar:
    st.title("Project 3")
    st.caption("Lagos Wildfire Risk Screening")
    st.markdown("---")
    st.markdown("**Study period**  \n2003–2016")
    st.markdown("**Training grid**  \n0.25° across Nigeria")
    st.markdown("**Validation**  \n5 spatial folds · 2° blocks")
    st.markdown("**Independent holdout**  \n2017–2024 Lagos VIIRS")
    st.markdown("---")
    st.caption("This dashboard reads saved project outputs. It does not run the model or fetch data.")
    if st.button("Refresh saved outputs", width="stretch"):
        st.rerun()

st.title("Lagos Wildfire Risk Screening")
st.markdown(
    "A presentation-ready view of the historical geospatial ML project, its saved results, "
    "and the files needed to reproduce it."
)
st.warning(
    "Exploratory historical screening only. Map scores are relative model outputs, "
    "not calibrated incident probabilities, live forecasts, or operational warnings."
)

if metrics is None:
    st.error("Saved validation metrics were not found or could not be read.")
    st.code("python run.py", language="powershell")
    st.stop()

if "features" not in metrics:
    st.warning(
        "These saved metrics predate the current feature-set metadata. "
        "Run the model pipeline to regenerate the current saved results."
    )

overview, map_tab, explanation_tab, files_tab = st.tabs(
    ["Overview", "Lagos map", "Model comparison & importance", "Data & run log"]
)

with overview:
    st.subheader("Spatial validation")
    st.markdown(f"**Selected model:** {metrics.get('best_model', 'Earlier Random Forest baseline')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ROC AUC",
        format_metric(metrics.get("roc_auc_mean")),
        border=True,
        help="Mean ROC AUC across held-out spatial folds. Higher is better.",
    )
    c1.caption(f"Fold SD {format_metric(metrics.get('roc_auc_sd'))}")
    c2.metric(
        "Average precision",
        format_metric(metrics.get("average_precision_mean")),
        border=True,
        help="Precision-recall summary; compare with the positive rate below.",
    )
    c3.metric(
        "Brier score",
        format_metric(metrics.get("brier_mean")),
        border=True,
        help="Mean squared probability error across folds. Lower is better.",
    )
    c4.metric("Fire-positive rows", f"{metrics.get('positive_rate', 0):.1%}", border=True)

    st.markdown("### Study at a glance")
    a, b, c = st.columns(3)
    a.metric("Grid-month records", f"{metrics.get('records', 0):,}", border=True)
    a.caption("Out-of-fold grid-month records")
    b.metric("Spatial folds", str(metrics.get("folds", "—")), border=True)
    b.caption("Groups are 2° geographic blocks")
    c.metric("Lagos display cells", str(len(risk_grid) if risk_grid is not None else 0), border=True)
    c.caption("Native 0.25° model cells, clipped to Lagos State")

    if comparison is not None and not comparison.empty:
        st.markdown("### Candidate model comparison")
        st.dataframe(comparison, hide_index=True, width="stretch")

    st.markdown("### Threshold summary")
    t1, t2, t3 = st.columns(3)
    t1.metric("Precision at 0.50", format_metric(metrics.get("precision_at_0_5")), border=True)
    t2.metric("Recall at 0.50", format_metric(metrics.get("recall_at_0_5")), border=True)
    t3.metric("F1 at 0.50", format_metric(metrics.get("f1_at_0_5")), border=True)
    st.caption(
        "These results describe satellite fire-proxy classification on spatially held-out "
        "Nigeria blocks. They do not measure local Lagos incident accuracy."
    )

    with st.expander("How to read the validation metrics"):
        st.markdown(
            "- **ROC AUC** measures ranking quality across thresholds.\n"
            "- **Average precision** summarizes the precision-recall curve; the training positive rate is shown above.\n"
            "- **Brier score** measures squared probability error; smaller values are better.\n"
            "- Precision, recall, and F1 use a fixed 0.50 threshold and pooled out-of-fold predictions."
        )

    if viirs_metrics is not None:
        st.subheader("Independent Lagos holdout · 2017–2024")
        st.caption(
            "The frozen model was applied to later-year Lagos predictors and compared "
            "with NASA VIIRS active-fire detections. This sensor/time holdout is separate "
            "from the Nigeria spatial cross-validation above."
        )
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("VIIRS ROC AUC", format_metric(viirs_metrics.get("roc_auc")), border=True)
        v2.metric("VIIRS average precision", format_metric(viirs_metrics.get("average_precision")), border=True)
        v3.metric("VIIRS Brier score", format_metric(viirs_metrics.get("brier_score")), border=True)
        v4.metric("Fire-positive cell-months", f"{viirs_metrics.get('positive_rate', 0):.1%}", border=True)
        st.caption(
            f"{viirs_metrics.get('records', 0):,} complete monthly records across three Lagos grid cells. "
            "The target is a VIIRS satellite detection proxy, not a fire-service incident audit."
        )

with map_tab:
    st.subheader("Historical Lagos screening map")
    map_path = OUTPUTS / "maps" / "lagos_wildfire_risk.png"
    if map_path.is_file():
        st.image(
            str(map_path),
            width="stretch",
            alt="Lagos map of historical wildfire screening scores for model cells clipped to the state boundary.",
        )
    else:
        st.info("The map is not available yet. Run the project pipeline to create it.")

    if risk_grid is not None and not risk_grid.empty:
        shown = risk_grid.rename(
            columns={
                "latitude": "Latitude",
                "longitude": "Longitude",
                "risk_probability": "Mean model score",
                "relative_risk_percentile": "Relative percentile",
                "grid_resolution_degrees": "Model grid (degrees)",
            }
        ).copy()
        if "Mean model score" in shown:
            shown["Mean model score"] = shown["Mean model score"].map(lambda x: f"{x:.4f}")
        if "Relative percentile" in shown:
            shown["Relative percentile"] = shown["Relative percentile"].map(lambda x: f"{x:.0f}")
        st.markdown("### Displayed cells")
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            alt="Model scores and relative ranks for Lagos grid cells.",
        )
    predictor_text = metrics.get("feature_set", "the saved model feature set")
    st.caption(
        f"Each score uses {predictor_text} on the 0.25° model grid. "
        "Cell polygons are clipped to the Lagos State boundary; relative ranks are calculated over the displayed Lagos cells."
    )

with explanation_tab:
    st.subheader("Seasonal model profile")
    if seasonal is not None and not seasonal.empty:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        season_view = seasonal.copy()
        season_view["Month"] = season_view["month"].astype(int).map(lambda m: month_names[m - 1])
        season_view = season_view.set_index("Month")[["risk_probability"]].rename(
            columns={"risk_probability": "Mean model score"}
        )
        st.line_chart(
            season_view,
            y="Mean model score",
            x_label="Month",
            y_label="Mean model score",
            alt="Monthly average model score across Lagos grid cells.",
        )
    else:
        st.info("The saved seasonal profile is not available yet.")

    st.subheader("Model predictor importance")
    if importance is not None and not importance.empty:
        importance_method = str(importance["method"].iloc[0])
        method_label = "SHAP" if importance_method == "SHAP" else "Permutation"
        metric_label = "Mean absolute SHAP contribution" if method_label == "SHAP" else "Average precision score decrease"
        imp_view = importance.sort_values("importance", ascending=True).tail(14).copy()
        imp_view = imp_view.set_index("feature")[["importance"]].rename(columns={"importance": metric_label})
        st.bar_chart(
            imp_view,
            x_label=metric_label,
            y_label="Feature",
            horizontal=True,
            alt=f"{method_label} feature importance for the selected model.",
        )
        st.dataframe(
            importance.sort_values("importance", ascending=False),
            hide_index=True,
            width="stretch",
            alt=f"{method_label} feature importance by predictor.",
        )
    else:
        st.info("The saved importance table is not available yet.")
    st.caption(f"{method_label if importance is not None and not importance.empty else 'Model importance'} describes fitted model behavior, not causal effects.")

    if folds is not None and not folds.empty:
        st.subheader("Fold-by-fold validation")
        st.dataframe(
            folds,
            hide_index=True,
            width="stretch",
            alt="Validation metrics and record counts for each spatial fold.",
        )

with files_tab:
    st.subheader("Saved project artifacts")
    artifact_rows = [
        ("Lagos map", "outputs/maps/lagos_wildfire_risk.png"),
        ("Model explanation figure", "outputs/figures/model_explainability.png"),
        ("Saved estimator", "outputs/wildfire_model.joblib"),
        ("Validation metrics", "outputs/metrics.json"),
        ("Spatial validation folds", "outputs/spatial_validation_folds.csv"),
        ("Model comparison", "outputs/model_comparison.csv"),
        ("Model feature importance", "outputs/feature_importance.csv"),
        ("Independent VIIRS metrics", "outputs/viirs_validation_metrics.json"),
        ("Independent VIIRS cell-month results", "data/processed/lagos_viirs_validation.csv"),
        ("Training features", "data/processed/training_features.csv"),
        ("Lagos grid scores", "data/processed/lagos_risk_grid.csv"),
        ("Lagos clipped grid GeoJSON", "data/processed/lagos_risk_grid.geojson"),
        ("MODIS monthly feature export", "data/processed/modis_monthly.csv"),
        ("Seasonal profile", "data/processed/lagos_seasonal_profile.csv"),
        ("Updated presentation", "presentation/Lagos_Wildfire_Risk_Screening_Updated.pptx"),
        ("Project knowledge base", "PROJECT_KNOWLEDGE_BASE.md"),
        ("Project report", "reports/project_report.md"),
        ("VIIRS validation report", "reports/viirs_validation.md"),
    ]
    status = pd.DataFrame(
        [
            {"Artifact": label, "Path": rel_path, "Available": (ROOT / rel_path).is_file()}
            for label, rel_path in artifact_rows
        ]
    )
    st.dataframe(
        status,
        hide_index=True,
        width="stretch",
        alt="Availability of saved project outputs and reports.",
    )

    st.subheader("Recent pipeline log")
    log_path = OUTPUTS / "logs" / "pipeline.log"
    if log_path.is_file():
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            st.code("\n".join(tail), language="text", wrap_lines=True)
        except OSError as exc:
            st.error(f"Could not read the saved log: {exc}")
    else:
        st.info("No pipeline log is available yet. Run `python run.py` to generate one.")

    st.subheader("Data sources")
    st.markdown(
        "- [Global Fire Atlas](https://zenodo.org/records/14065246) · monthly fire-number rasters, 2003–2016\n"
        "- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · ERA5 reanalysis\n"
        "- [MODIS MOD13Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1) · NDVI and EVI\n"
        "- [MODIS MOD11A2](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2) · land-surface temperature\n"
        "- [NASA VIIRS VNP14A1](https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP14A1) · independent active-fire holdout, 2017–2024\n"
        "- [Nigeria state polygons](https://gist.github.com/segebee/ff4a2f75be560462c369) · training land mask and Lagos outline"
    )

st.divider()
st.caption(
    "Lagos Wildfire Risk Screening · SIWES Project 3 · Reproducible outputs are stored in this project folder."
)
