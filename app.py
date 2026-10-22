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
    page_icon="🌿",
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
risk_grid = read_csv(PROCESSED / "lagos_risk_grid.csv")
seasonal = read_csv(PROCESSED / "lagos_seasonal_profile.csv")
importance = read_csv(OUTPUTS / "feature_importance.csv")
folds = read_csv(OUTPUTS / "spatial_validation_folds.csv")

with st.sidebar:
    st.title("Project 3")
    st.caption("Lagos Wildfire Risk Screening")
    st.markdown("---")
    st.markdown("**Study period**  \n2003–2016")
    st.markdown("**Training grid**  \n0.5° across Nigeria")
    st.markdown("**Validation**  \n5 spatial folds · 2° blocks")
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

overview, map_tab, explanation_tab, files_tab = st.tabs(
    ["Overview", "Lagos map", "Model explanation", "Data & run log"]
)

with overview:
    st.subheader("Spatial validation")
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
    a.caption("301 Nigeria land cells × 168 months")
    b.metric("Spatial folds", str(metrics.get("folds", "—")), border=True)
    b.caption("Groups are 2° geographic blocks")
    c.metric("Lagos display cells", str(len(risk_grid) if risk_grid is not None else 0), border=True)
    c.caption("0.25° display centers; model trained at 0.5°")

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
            "- **Average precision** summarizes the precision-recall curve; the baseline positive rate is shown above.\n"
            "- **Brier score** measures squared probability error; smaller values are better.\n"
            "- Precision, recall, and F1 use a fixed 0.50 threshold and pooled out-of-fold predictions."
        )

with map_tab:
    st.subheader("Historical Lagos screening map")
    map_path = OUTPUTS / "maps" / "lagos_wildfire_risk.png"
    if map_path.is_file():
        st.image(
            str(map_path),
            width="stretch",
            alt="Lagos map of relative historical wildfire screening scores for three displayed cells.",
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
                "display_resolution_degrees": "Display grid (degrees)",
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
            alt="Model scores and relative ranks for the three Lagos display cells.",
        )
    st.caption(
        "Only three 0.25° cell centers fall inside the current Lagos polygon/grid alignment. "
        "Their values are spatially interpolated from a 0.5°-trained model. Percentiles are "
        "relative to these displayed cells and are not calibrated probabilities."
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
            alt="Monthly model score profile, highest in January and lowest from May through October.",
        )
    else:
        st.info("The saved seasonal profile is not available yet.")

    st.subheader("Predictor importance")
    if importance is not None and not importance.empty:
        imp_view = importance.sort_values("importance_mean", ascending=True).tail(9).copy()
        imp_view = imp_view.set_index("feature")[["importance_mean"]].rename(
            columns={"importance_mean": "Mean permutation importance"}
        )
        st.bar_chart(
            imp_view,
            x_label="Mean permutation importance",
            y_label="Feature",
            horizontal=True,
            alt="Exploratory feature importance is highest for month seasonality and rainfall.",
        )
        st.dataframe(
            importance.sort_values("importance_mean", ascending=False),
            hide_index=True,
            width="stretch",
            alt="Mean permutation importance and repeat variability for model features.",
        )
    else:
        st.info("The saved importance table is not available yet.")
    st.info(
        "Interpretation caveat: the saved importance block was included when the final model "
        "was fitted. Treat these values as exploratory diagnostics, not independent validation "
        "or causal evidence."
    )

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
        ("Feature importance", "outputs/feature_importance.csv"),
        ("Training features", "data/processed/training_features.csv"),
        ("Lagos grid scores", "data/processed/lagos_risk_grid.csv"),
        ("Seasonal profile", "data/processed/lagos_seasonal_profile.csv"),
        ("Presentation", "presentation/Lagos_Wildfire_Risk_Screening.pptx"),
        ("Project knowledge base", "PROJECT_KNOWLEDGE_BASE.md"),
        ("Project report", "reports/project_report.md"),
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
        "- [Nigeria state polygons](https://gist.github.com/segebee/ff4a2f75be560462c369) · training land mask and Lagos outline"
    )

st.divider()
st.caption(
    "Lagos Wildfire Risk Screening · SIWES Project 3 · Reproducible outputs are stored in this project folder."
)
