"""
Spare Part Demand Forecasting - V2 Dashboard

Final v2 dashboard for the 2023-2026 dataset.
Run from:
    D:/Diploma_work/diploma/diploma_v2

Command:
    streamlit run app_v2.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# 1. PAGE CONFIG AND PATHS
# ============================================================

st.set_page_config(
    page_title="Spare Part Demand Forecasting V2",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
XLSX = BASE / "results" / "xlsx"
CHARTS = BASE / "results" / "charts"
MODELS = BASE / "models"


# ============================================================
# 2. STYLE
# ============================================================

st.markdown(
    """
<style>
    .block-container { padding-top: 1.4rem; }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .defence-box {
        background: #12355b;
        color: white;
        padding: 1.2rem 1.4rem;
        border-radius: 8px;
        line-height: 1.6;
        font-size: 1.05rem;
        margin: 0.8rem 0 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #f0ad4e;
        color: #333;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin: 0.7rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        color: #1f3d2b;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin: 0.7rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. HELPERS
# ============================================================

def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def show_chart(path: Path, caption: str = ""):
    if file_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart not found: {path}")


def read_excel_first_available(path: Path, possible_sheets: list[str]) -> pd.DataFrame | None:
    if not file_exists(path):
        return None

    try:
        excel = pd.ExcelFile(path)
        for sheet in possible_sheets:
            if sheet in excel.sheet_names:
                return pd.read_excel(path, sheet_name=sheet)
        return pd.read_excel(path, sheet_name=excel.sheet_names[0])
    except Exception:
        return None


def safe_metric(df: pd.DataFrame | None, metric_name: str, default="-"):
    if df is None or "metric" not in df.columns or "value" not in df.columns:
        return default
    row = df[df["metric"].astype(str).str.lower() == metric_name.lower()]
    if row.empty:
        return default
    return row["value"].iloc[0]


def make_search_text(df: pd.DataFrame) -> pd.Series:
    part = df["Part Number"].astype(str) if "Part Number" in df.columns else ""
    desc = df["description"].astype(str) if "description" in df.columns else ""
    return (part + " " + desc).str.lower()


def format_number(value):
    try:
        value = float(value)
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    except Exception:
        return str(value)


# ============================================================
# 4. DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_discovery_report():
    path = XLSX / "03_combined_data_discovery_report.xlsx"
    return {
        "business": read_excel_first_available(path, ["business_summary"]),
        "date": read_excel_first_available(path, ["date_summary"]),
        "quantity": read_excel_first_available(path, ["quantity_summary"]),
        "forecastable": read_excel_first_available(path, ["forecastable_summary"]),
    }


@st.cache_data(show_spinner=False)
def load_monthly_report_summary():
    path = XLSX / "04_monthly_demand_table_v2_report.xlsx"
    return read_excel_first_available(path, ["summary", "monthly_summary"])


@st.cache_data(show_spinner=False)
def load_feature_data():
    path = XLSX / "06b_feature_engineered_v4_no_lag24.csv"
    if not file_exists(path):
        return None

    df = pd.read_csv(path, low_memory=False)
    if "Month Date" in df.columns:
        df["Month Date"] = pd.to_datetime(df["Month Date"], errors="coerce")
    if "Month" not in df.columns and "Month Date" in df.columns:
        df["Month"] = df["Month Date"].dt.to_period("M").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_feature_summary():
    path = XLSX / "06b_feature_engineering_v4_no_lag24_report.xlsx"
    return read_excel_first_available(path, ["summary", "feature_summary"])


@st.cache_data(show_spinner=False)
def load_baseline_metrics():
    path = XLSX / "07_baseline_forecasting_v2_report.xlsx"
    return read_excel_first_available(path, ["baseline_metrics", "metrics"])


@st.cache_data(show_spinner=False)
def load_xgboost_report():
    path = XLSX / "08_xgboost_log_target_v2_report.xlsx"
    return {
        "metrics": read_excel_first_available(path, ["xgboost_metrics"]),
        "comparison": read_excel_first_available(path, ["comparison_metrics", "all_model_metrics"]),
        "predictions": read_excel_first_available(path, ["test_predictions"]),
        "importance": read_excel_first_available(path, ["feature_importance"]),
    }


@st.cache_data(show_spinner=False)
def load_shap_report():
    path = XLSX / "09_shap_explainability_v2_report.xlsx"
    return {
        "global": read_excel_first_available(path, ["shap_global_importance", "global_feature_importance"]),
        "example": read_excel_first_available(path, ["example_prediction", "example_explained"]),
        "contrib": read_excel_first_available(path, ["example_feature_contributions", "feature_contributions"]),
    }


@st.cache_data(show_spinner=False)
def load_final_report():
    path = XLSX / "10_final_model_selection_v2_report.xlsx"
    return {
        "metrics": read_excel_first_available(path, ["all_model_metrics", "comparison_metrics"]),
        "decision": read_excel_first_available(path, ["final_decision"]),
    }


@st.cache_data(show_spinner=False)
def load_future_forecasts():
    path = XLSX / "11_future_forecast_v2.xlsx"
    if not file_exists(path):
        return None
    try:
        return pd.read_excel(path, sheet_name="future_forecasts")
    except Exception:
        return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def load_model_metadata():
    path = MODELS / "xgboost_log_target_v2_metadata.json"
    if not file_exists(path):
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ============================================================
# 5. STATIC PROJECT FACTS
# ============================================================

PROJECT_FACTS = {
    "raw_rows": "376,633",
    "unique_parts": "35,161",
    "months": "37",
    "forecastable_parts": "1,001",
    "model_ready_rows": "24,024",
    "train_rows": "16,016",
    "validation_rows": "3,003",
    "test_rows": "5,005",
    "test_months": "2025-09 to 2026-01",
    "production_model": "pred_lag_1",
    "production_mae": "22.83",
    "xgboost_mae": "25.52",
}


# ============================================================
# 6. SIDEBAR
# ============================================================

PAGES = [
    "Project Overview",
    "Data and Features",
    "Model Comparison",
    "Production Forecast",
    "XGBoost and SHAP",
    "Final Conclusion",
]

with st.sidebar:
    st.markdown("## Demand Forecasting V2")
    st.markdown("Automotive spare parts")
    st.caption("2023-2026 final version")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("This app loads saved reports, charts, forecasts, and model metadata. It does not retrain models.")


# ============================================================
# 7. PAGE: PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":
    st.title("Explainable Demand Forecasting System for Automotive Spare Parts")
    st.markdown("Final v2 dashboard using 2023-2026 transaction data.")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Rows", PROJECT_FACTS["raw_rows"])
    c2.metric("Unique Parts", PROJECT_FACTS["unique_parts"])
    c3.metric("Months", PROJECT_FACTS["months"])
    c4.metric("Forecastable Parts", PROJECT_FACTS["forecastable_parts"])
    c5.metric("Final Model", PROJECT_FACTS["production_model"])

    st.markdown(
        """
<div class="defence-box">
This project does not claim that complex machine learning always wins. It builds a complete
forecasting pipeline, compares simple benchmark methods against XGBoost, and selects the final
production method using time-based test evidence.
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Business Problem")
        st.markdown(
            """
Automotive spare-parts demand is difficult because there are many products, irregular sales,
returns, occasional bulk orders, and different demand behaviour across parts.

Wrong forecasts create two business risks:

- Stockouts: the part is not available when the customer needs it.
- Overstock: money is locked in slow-moving inventory.
"""
        )

        st.subheader("Target Variable")
        st.info("net_qty = sold_qty - return_qty")
        st.markdown(
            """
The model predicts monthly net demand per part. Returns are subtracted because returned items
reduce true business demand.
"""
        )

    with right:
        st.subheader("V2 Improvement Over V1")
        st.markdown(
            """
| Area | V1 | V2 |
|---|---:|---:|
| Raw rows | 249,137 | 376,633 |
| Date coverage | 25 months | 37 months |
| Forecastable parts | 651 | 1,001 |
| Feature rows | 8,463 | 24,024 |
| Final production model | lag-1 | lag-1 |

V2 is stronger because it adds 2023 history, supports yearly lag features, and tests the final
decision on more forecastable parts.
"""
        )

        st.subheader("Final Decision")
        st.success(
            "pred_lag_1 is selected as the production model because it achieved the lowest MAE "
            "on the final time-based test set."
        )
        st.warning(
            "XGBoost is retained as an explainable ML model with SHAP, but it is not the best "
            "production method by MAE."
        )


# ============================================================
# 8. PAGE: DATA AND FEATURES
# ============================================================

elif page == "Data and Features":
    st.title("Data Preparation and Feature Engineering")
    st.markdown("How raw `.xlsb` transaction data became a model-ready forecasting dataset.")
    st.divider()

    discovery = load_discovery_report()
    feature_summary = load_feature_summary()
    monthly_summary = load_monthly_report_summary()
    feature_df = load_feature_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", PROJECT_FACTS["raw_rows"])
    c2.metric("Forecastable Parts", PROJECT_FACTS["forecastable_parts"])
    c3.metric("Model Rows", PROJECT_FACTS["model_ready_rows"])
    c4.metric("Feature Count", safe_metric(feature_summary, "feature_count", "38"))

    st.subheader("Data Pipeline")
    st.markdown(
        """
```text
DB_31_2023.xlsb + DB_31_24&25.xlsb
-> combine files
-> convert Excel serial dates
-> clean and standardise columns
-> aggregate transactions by Month + Part Number
-> select forecastable parts
-> create lag, rolling, calendar, intermittent, and part-level features
```
"""
    )

    tab1, tab2, tab3 = st.tabs(["Discovery Summary", "Feature Split", "Feature Data Sample"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.subheader("Business Summary")
            if discovery["business"] is not None:
                st.dataframe(discovery["business"], use_container_width=True, hide_index=True)
            else:
                st.info("Business summary file was not found.")

        with right:
            st.subheader("Quantity Summary")
            if discovery["quantity"] is not None:
                st.dataframe(discovery["quantity"], use_container_width=True, hide_index=True)
            else:
                st.info("Quantity summary file was not found.")

        st.subheader("Monthly Demand Table Summary")
        if monthly_summary is not None:
            st.dataframe(monthly_summary, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Train / Validation / Test Split")
        st.markdown(
            """
| Split | Months | Rows |
|---|---|---:|
| Train | 2024-02 to 2025-05 | 16,016 |
| Validation | 2025-06 to 2025-08 | 3,003 |
| Test | 2025-09 to 2026-01 | 5,005 |

The split is time-based, not random. This is important because forecasting must predict future
months from past months.
"""
        )

        if feature_summary is not None:
            st.dataframe(feature_summary, use_container_width=True, hide_index=True)

        st.subheader("Feature Groups")
        st.markdown(
            """
| Group | Examples |
|---|---|
| Lag features | lag_1, lag_2, lag_3, lag_6, lag_12, lag_13 |
| Rolling features | rolling_mean_3, rolling_mean_6, rolling_mean_12 |
| Calendar features | month_number, quarter, month_sin, month_cos |
| Return features | return_qty_lag_1, return_ratio_lag_1 |
| Intermittent-demand features | sales_months_last_6, months_since_last_sale |
| Safe part-level features | part_train_avg_qty, part_train_median_qty |
"""
        )

    with tab3:
        if feature_df is None:
            st.error("Feature engineered CSV not found.")
        else:
            important_cols = [
                "Part Number", "Month", "net_qty", "lag_1", "lag_12", "rolling_mean_3",
                "rolling_mean_12", "part_train_avg_qty", "part_train_median_qty", "data_split",
                "description", "fr",
            ]
            cols = [c for c in important_cols if c in feature_df.columns]
            st.dataframe(feature_df[cols].head(200), use_container_width=True, hide_index=True)


# ============================================================
# 9. PAGE: MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":
    st.title("Model Comparison and Final Selection")
    st.markdown("The final model is chosen by test-set MAE, not by model complexity.")
    st.divider()

    final_report = load_final_report()
    metrics = final_report["metrics"]
    decision = final_report["decision"]

    if metrics is None:
        metrics = pd.DataFrame(
            [
                {"model": "pred_lag_1", "MAE": 22.828172, "RMSE": 222.045448, "sMAPE": 96.832225},
                {"model": "pred_xgboost_log_v2", "MAE": 25.524848, "RMSE": 353.393542, "sMAPE": 94.054433},
                {"model": "pred_rolling_mean_3", "MAE": 27.056210, "RMSE": 341.325523, "sMAPE": 90.602719},
                {"model": "pred_part_train_median", "MAE": 36.717782, "RMSE": 487.274604, "sMAPE": 96.294472},
                {"model": "pred_rolling_mean_6", "MAE": 36.992607, "RMSE": 510.593792, "sMAPE": 90.576537},
                {"model": "pred_part_train_avg", "MAE": 40.687995, "RMSE": 541.603275, "sMAPE": 95.977249},
            ]
        )

    metrics = metrics.copy()
    if "model" in metrics.columns:
        metrics = metrics.sort_values("MAE", ascending=True)
        best_model = metrics.iloc[0]["model"]
        best_mae = metrics.iloc[0]["MAE"]
    else:
        best_model = "pred_lag_1"
        best_mae = 22.828172

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Production Model", str(best_model))
    c2.metric("Best MAE", f"{float(best_mae):.2f}")
    c3.metric("Test Period", PROJECT_FACTS["test_months"])

    st.markdown(
        """
<div class="success-box">
Final production decision: pred_lag_1. It achieved the lowest MAE on the final time-based test set.
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("All Model Metrics")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        show_chart(
            CHARTS / "final_model_selection_v2" / "final_model_selection_v2_mae_comparison.png",
            "Final model comparison by MAE",
        )
    with right:
        show_chart(
            CHARTS / "final_model_selection_v2" / "final_model_selection_v2_rmse_comparison.png",
            "Final model comparison by RMSE",
        )

    st.subheader("Decision Record")
    if decision is not None:
        st.dataframe(decision, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
| Field | Value |
|---|---|
| final_production_model | pred_lag_1 |
| selection_metric | MAE |
| explainable_ml_model | pred_xgboost_log_v2 |
| xgboost_role | Explainability with SHAP, not production winner |
"""
        )

    st.markdown(
        """
<div class="warning-box">
Important: the XGBoost model is not a failure. It is a useful explainable ML model, but the
business production forecast should use the method that gives the lowest test error.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 10. PAGE: PRODUCTION FORECAST
# ============================================================

elif page == "Production Forecast":
    st.title("Production Forecast")
    st.markdown("Future monthly forecasts using the selected production model: `pred_lag_1`.")
    st.divider()

    future_df = load_future_forecasts()
    feature_df = load_feature_data()

    if future_df is None:
        st.error("Future forecast file not found: results/xlsx/11_future_forecast_v2.xlsx")
        st.stop()

    future_df = future_df.copy()
    future_df["search_text"] = make_search_text(future_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecastable Parts", f"{future_df['Part Number'].nunique():,}")
    c2.metric("Forecast Rows", f"{len(future_df):,}")
    c3.metric("Forecast Horizon", f"{future_df['forecast_step'].max()} months")

    st.markdown(
        """
<div class="warning-box">
The first future month is the strongest forecast because it uses the latest real month as lag_1.
Later months are recursive, meaning they depend on earlier predictions.
</div>
""",
        unsafe_allow_html=True,
    )

    search = st.text_input("Search by part number or description", "")
    if search.strip():
        matched = future_df[future_df["search_text"].str.contains(search.strip().lower(), na=False)]
        part_options = sorted(matched["Part Number"].unique())
        if not part_options:
            st.warning("No matching parts found. Showing all parts.")
            part_options = sorted(future_df["Part Number"].unique())
    else:
        part_options = sorted(future_df["Part Number"].unique())

    selected_part = st.selectbox("Select part", part_options)
    part_future = future_df[future_df["Part Number"] == selected_part].copy()

    description = part_future["description"].iloc[0] if "description" in part_future.columns else ""
    st.subheader(f"{selected_part} - {description}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=part_future["forecast_month"],
            y=part_future["predicted_net_qty"],
            text=part_future["predicted_net_qty"].round(1),
            textposition="outside",
            name="Future forecast",
        )
    )
    fig.update_layout(
        title="Production forecast using lag-1",
        xaxis_title="Forecast month",
        yaxis_title="Predicted net quantity",
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    show_cols = [
        "forecast_month", "forecast_step", "recommended_model",
        "predicted_net_qty", "forecast_warning",
    ]
    show_cols = [c for c in show_cols if c in part_future.columns]
    st.dataframe(part_future[show_cols], use_container_width=True, hide_index=True)

    if feature_df is not None:
        st.subheader("Historical Demand")
        hist = feature_df[feature_df["Part Number"] == selected_part].copy()
        if not hist.empty:
            hist = hist.sort_values("Month Date")
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Scatter(
                    x=hist["Month"],
                    y=hist["net_qty"],
                    mode="lines+markers",
                    name="Actual net_qty",
                )
            )
            fig_hist.update_layout(
                title="Historical monthly net demand",
                xaxis_title="Month",
                yaxis_title="Net quantity",
                height=420,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            hist_cols = [
                "Month", "net_qty", "sold_qty", "return_qty", "transaction_count",
                "description", "fr", "data_split",
            ]
            hist_cols = [c for c in hist_cols if c in hist.columns]
            with st.expander("Historical table"):
                st.dataframe(hist[hist_cols], use_container_width=True, hide_index=True)


# ============================================================
# 11. PAGE: XGBOOST AND SHAP
# ============================================================

elif page == "XGBoost and SHAP":
    st.title("XGBoost and SHAP Explainability")
    st.markdown("The ML model is kept for explanation and comparison, not as the production winner.")
    st.divider()

    xgb_report = load_xgboost_report()
    shap_report = load_shap_report()
    metadata = load_model_metadata()

    st.markdown(
        """
<div class="warning-box">
SHAP explains the XGBoost model trained on log1p(net_qty). It does not explain the lag-1
production forecast, because lag-1 is a simple benchmark formula.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost MAE", PROJECT_FACTS["xgboost_mae"])
    c2.metric("Production MAE", PROJECT_FACTS["production_mae"])
    c3.metric("Feature Count", metadata.get("feature_count", "38"))

    tab1, tab2, tab3 = st.tabs(["XGBoost Results", "SHAP Charts", "SHAP Tables"])

    with tab1:
        st.subheader("XGBoost Model Metrics")
        if xgb_report["metrics"] is not None:
            st.dataframe(xgb_report["metrics"], use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            show_chart(
                CHARTS / "xgboost_v2" / "xgboost_v2_model_comparison_mae.png",
                "XGBoost compared with baseline methods",
            )
        with right:
            show_chart(
                CHARTS / "xgboost_v2" / "xgboost_v2_feature_importance.png",
                "XGBoost feature importance",
            )

        show_chart(
            CHARTS / "xgboost_v2" / "xgboost_v2_example_part_prediction.png",
            "Example XGBoost prediction against actual demand",
        )

        if metadata:
            with st.expander("Saved model metadata"):
                st.json(metadata)

    with tab2:
        st.subheader("Global SHAP Explanation")
        left, right = st.columns(2)
        with left:
            show_chart(
                CHARTS / "shap_v2" / "shap_v2_global_feature_importance.png",
                "Mean absolute SHAP value",
            )
        with right:
            show_chart(
                CHARTS / "shap_v2" / "shap_v2_summary_plot.png",
                "SHAP summary plot",
            )

        st.subheader("Example Local Explanation")
        show_chart(
            CHARTS / "shap_v2" / "shap_v2_waterfall_example.png",
            "Waterfall explanation for one prediction",
        )

    with tab3:
        st.subheader("Top SHAP Features")
        if shap_report["global"] is not None:
            st.dataframe(shap_report["global"], use_container_width=True, hide_index=True)
        else:
            st.info("SHAP global importance table not found.")

        st.subheader("Example Prediction Explained")
        if shap_report["example"] is not None:
            st.dataframe(shap_report["example"], use_container_width=True, hide_index=True)

        st.subheader("Example Feature Contributions")
        if shap_report["contrib"] is not None:
            st.dataframe(shap_report["contrib"], use_container_width=True, hide_index=True)


# ============================================================
# 12. PAGE: FINAL CONCLUSION
# ============================================================

elif page == "Final Conclusion":
    st.title("Final Conclusion")
    st.markdown("What the project proves and how it should be used.")
    st.divider()

    st.markdown(
        """
<div class="defence-box">
Final defence message: I built a complete demand-forecasting decision system. The final
production method is lag-1 because it achieved the lowest MAE on future test months. XGBoost
is retained as an explainable machine-learning model using SHAP.
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Final Findings")
        st.markdown(
            """
1. The project used 376,633 transaction rows from 2023-2026.
2. 1,001 forecastable parts were selected using business rules.
3. A time-based split was used, not a random split.
4. The final production method is `pred_lag_1`.
5. XGBoost was close but did not beat lag-1 by MAE.
6. SHAP was used to explain the XGBoost model.
7. Streamlit presents forecasts, model comparison, and explainability outputs.
"""
        )

        st.subheader("Recommended Business Use")
        st.success(
            "Use the production lag-1 forecast for short-term monthly inventory planning, "
            "especially the next month."
        )
        st.warning(
            "Refresh the data and rerun the pipeline monthly. Do not assume the current model "
            "will stay best forever."
        )

    with right:
        st.subheader("Limitations")
        st.markdown(
            """
- Multi-step future forecasts become weaker because later months are recursive.
- The dataset does not include stock levels, promotions, supplier delays, or prices as external drivers.
- Some demand is intermittent, meaning many parts have zero or unstable monthly demand.
- XGBoost underpredicts some extreme high-demand months.
"""
        )

        st.subheader("Future Improvements")
        st.markdown(
            """
- Add more years of history.
- Add stock availability and lost sales data.
- Add price, promotion, season, and service campaign variables.
- Evaluate specialised intermittent-demand models.
- Automate monthly refresh and monitoring.
"""
        )

    st.divider()
    st.subheader("Submission Summary")
    st.markdown(
        """
| Component | Final Status |
|---|---|
| Data pipeline | Complete |
| Feature engineering | Complete |
| Baseline forecasting | Complete |
| XGBoost model | Saved as `.pkl` |
| SHAP explainability | Complete |
| Future forecast file | Complete |
| Streamlit v2 dashboard | Complete |
| Final production model | `pred_lag_1` |
"""
    )

    st.caption("Dashboard v2. Built for the final 2023-2026 project version.")
