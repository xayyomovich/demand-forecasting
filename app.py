"""
Spare Part Demand Forecasting — Diploma Project Dashboard
Author: Diploma student | Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spare Part Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
XLSX   = BASE / "outputs" / "xlsx"
CHARTS = BASE / "outputs" / "charts"
MODELS = BASE / "models"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 0.85rem; }
    .defence-box {
        background: linear-gradient(135deg, #1f4e79 0%, #2d6a9f 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        font-size: 1.15rem; font-weight: 500; line-height: 1.7;
        margin: 1rem 0;
    }
    .honesty-box {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 1rem 1.25rem; border-radius: 6px; margin: 0.75rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_feature_data():
    path = XLSX / "06_feature_engineering_v3_lag12_all651_report.xlsx"
    if not path.exists():
        return None
    return {
        "model_ready": pd.read_excel(path, sheet_name="model_ready_data"),
        "train":       pd.read_excel(path, sheet_name="train_data"),
        "test":        pd.read_excel(path, sheet_name="test_data"),
        "parts":       pd.read_excel(path, sheet_name="forecastable_parts"),
        "summary":     pd.read_excel(path, sheet_name="feature_summary"),
    }


@st.cache_data(show_spinner=False)
def load_hybrid_data():
    path = XLSX / "10b_hybrid_validation_test_split_report.xlsx"
    if not path.exists():
        return None
    return {
        "test_predictions": pd.read_excel(path, sheet_name="test_predictions"),
        "test_metrics":     pd.read_excel(path, sheet_name="test_comparison_metrics"),
        "best_model":       pd.read_excel(path, sheet_name="best_model_per_part"),
        "model_counts":     pd.read_excel(path, sheet_name="model_selection_counts"),
        "split_summary":    pd.read_excel(path, sheet_name="split_summary"),
    }


@st.cache_data(show_spinner=False)
def load_production_forecasts():
    path = XLSX / "11_future_forecasts.xlsx"
    if not path.exists():
        return None
    return pd.read_excel(path, sheet_name="future_forecasts")


@st.cache_data(show_spinner=False)
def load_xgboost_forecasts():
    path = XLSX / "12_xgboost_future_forecasts.xlsx"
    if not path.exists():
        return None
    return pd.read_excel(path, sheet_name="xgboost_future_forecasts")


@st.cache_data(show_spinner=False)
def load_shap_sample():
    path = XLSX / "12_xgboost_shap_input_sample.xlsx"
    if not path.exists():
        return None
    return pd.read_excel(path, sheet_name="shap_input_sample")


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load XGBoost model, feature list, and metadata. Cached as resource (not data)."""
    model_path = MODELS / "xgboost_log_lag12_all651.pkl"
    feat_path  = MODELS / "xgboost_log_lag12_all651_features.json"
    meta_path  = MODELS / "xgboost_log_lag12_all651_metadata.json"
    if not all(p.exists() for p in [model_path, feat_path, meta_path]):
        return None, None, None
    model    = joblib.load(model_path)
    features = json.loads(feat_path.read_text(encoding="utf-8"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, features, metadata


def show_chart(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"Chart not found: `{path.name}`")


# ── Hard-coded final test metrics ──────────────────────────────────────────────
FINAL_METRICS = pd.DataFrame([
    {"Model": "pred_lag_1",                "MAE": 36.50,  "RMSE": 283.87,  "sMAPE": 89.09,  "Note": "✅ Best MAE"},
    {"Model": "pred_rolling_mean_3",       "MAE": 42.45,  "RMSE": 426.89,  "sMAPE": 78.17,  "Note": ""},
    {"Model": "hybrid_validation_selected","MAE": 51.47,  "RMSE": 554.19,  "sMAPE": 79.65,  "Note": ""},
    {"Model": "pred_xgboost_log_lag12",    "MAE": 53.72,  "RMSE": 544.80,  "sMAPE": 84.05,  "Note": "ML model"},
    {"Model": "pred_part_train_avg",       "MAE": 74.03,  "RMSE": 993.74,  "sMAPE": 80.72,  "Note": ""},
])


# ── Sidebar navigation ─────────────────────────────────────────────────────────
PAGES = [
    "🏠  Project Overview",
    "📊  Data Discovery",
    "🔍  Outlier Analysis",
    "⚙️  Feature Engineering",
    "📈  Model Comparison",
    "🏭  Production Forecast",
    "🤖  XGBoost + SHAP",
    "✅  Final Conclusion",
]

with st.sidebar:
    st.markdown("## 📦 Demand Forecasting")
    st.markdown("**Diploma Project**")
    st.markdown("Automotive Spare Parts")
    st.markdown("---")
    page = st.radio("Navigate to:", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Loads from saved Excel + model files.\nNo automatic retraining.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("Spare Part Demand Forecasting System")
    st.markdown("**Diploma project · Monthly demand forecasting for automotive spare parts**")
    st.divider()

    feat = load_feature_data()

    n_parts = 651
    n_months = "~36 months"
    if feat:
        summary = feat["summary"]
        r = summary[summary["metric"] == "selected_parts"]
        if not r.empty:
            n_parts = int(r["value"].iloc[0])
        r2 = summary[summary["metric"] == "all_months"]
        if not r2.empty:
            n_months = f"{r2['value'].iloc[0]} months"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw Transactions", "249,137")
    c2.metric("Raw Columns", "31")
    c3.metric("Forecastable Parts", f"{n_parts}")
    c4.metric("Date Coverage", n_months)
    c5.metric("Production Method", "Lag-1")

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Business Goal")
        st.markdown("""
Forecast **monthly demand per spare part** to help an automotive dealership:

- Reduce stockouts on fast-moving consumables
- Avoid over-ordering slow-moving parts
- Provide data-driven, explainable inventory decisions

**Source data**: Transaction-level sales & return records from the dealer management system (2024–2025).
        """)

        st.subheader("Target Variable")
        st.info("""
**`net_qty = sold_quantity − returned_quantity`**

Captures true net demand. Returns (warranty replacements, excess stock) are subtracted
so the model learns real consumption, not just gross sales volume.
        """)

        st.subheader("Production Forecasting Method")
        st.success("""
**Recommended method: `pred_lag_1`**

After evaluating all models on the held-out test set across all 651 parts,
`pred_lag_1` (predict next month = last month's actual demand) achieved the lowest MAE.

The XGBoost ML model is included as an explainable ML experiment and supports SHAP analysis,
but it did not outperform lag-1 on the final all-parts test set.
        """)

    with right:
        st.subheader("Project Pipeline")
        pipeline = [
            ("01",  "Load & validate raw data"),
            ("02–03", "Data discovery & monthly demand aggregation"),
            ("04–05", "Outlier detection & EDA visualisations"),
            ("06",  "Feature engineering — lags, rolling stats, calendar"),
            ("07",  "Baseline models — lag-1, rolling mean, part average"),
            ("08e", "XGBoost with log-target & lag-12 on all 651 parts"),
            ("09",  "SHAP explainability analysis"),
            ("10b", "Hybrid per-part model selection & test evaluation"),
            ("11",  "Production lag-1 future forecasts (all 651 parts)"),
            ("12",  "XGBoost future forecasts & SHAP input sample"),
        ]
        for step, desc in pipeline:
            st.markdown(f"**Step {step}** — {desc}")

        st.subheader("Part Selection Criteria")
        st.markdown("""
| Criterion | Threshold |
|-----------|-----------|
| Active months | ≥ 18 |
| Total transactions | ≥ 50 |
| Total sold quantity | > 0 |

→ **651 forecastable parts** selected from full raw data.
        """)

    if feat:
        with st.expander("Browse forecastable parts"):
            df = feat["parts"]
            cols = [c for c in ["Part Number", "description", "fr", "active_months",
                                  "total_transactions", "avg_monthly_net_qty"] if c in df.columns]
            st.dataframe(df[cols].head(100), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("Data Discovery")
    st.markdown("Key patterns from the raw transaction data before any modelling.")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Monthly Trends", "Sold vs Returned", "Top Parts", "Franchise & Branch"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Monthly Net Quantity Trend")
            show_chart(CHARTS / "eda" / "monthly_net_quantity_trend.png")
        with c2:
            st.subheader("Monthly Sales Value Trend")
            show_chart(CHARTS / "eda" / "monthly_sales_value_trend.png")
        st.subheader("Unique Parts Active per Month")
        show_chart(CHARTS / "eda" / "unique_parts_by_month.png")

    with tab2:
        st.subheader("Monthly Sold vs Returned Quantity")
        show_chart(CHARTS / "eda" / "monthly_sold_vs_returned_quantity.png")
        st.markdown("""
The gap between sold and returned quantity defines `net_qty` — our forecast target.
Returns are not errors — they are real business events (warranty, overstocking) that reduce effective demand.
Modelling `net_qty` gives a more honest demand signal than raw gross sales.
        """)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top 20 Parts by Net Quantity")
            show_chart(CHARTS / "eda" / "top_20_parts_by_net_quantity.png")
        with c2:
            st.subheader("Top 20 Parts by Transaction Count")
            show_chart(CHARTS / "eda" / "top_20_parts_by_transaction_count.png")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Net Quantity by Franchise")
            show_chart(CHARTS / "eda" / "net_quantity_by_franchise.png")
        with c2:
            st.subheader("Net Quantity by Branch")
            show_chart(CHARTS / "eda" / "net_quantity_by_branch.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — OUTLIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("Outlier Analysis")
    st.markdown("Identifying extreme demand values — and deciding what to do with them.")
    st.divider()

    st.info("""
**Key Decision: Outliers were NOT blindly removed.**

In spare parts sales, a very large order (fleet purchase, bulk dealership supply) is a
**real business event**, not a data error. Removing it would cause the model to
systematically underestimate demand for high-volume customers.

Instead: outliers were documented with a 99th-percentile flag
(`is_high_demand_month_analysis_only`) kept separate from model training features.
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transaction Quantity Distribution")
        show_chart(CHARTS / "outliers" / "transaction_qty_distribution.png")
        st.subheader("Distribution — Zoomed")
        show_chart(CHARTS / "outliers" / "transaction_qty_distribution_zoomed.png")
    with c2:
        st.subheader("Top 20 Transaction Outliers")
        show_chart(CHARTS / "outliers" / "top_20_transaction_qty_outliers.png")

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Monthly Net Quantity Distribution")
        show_chart(CHARTS / "outliers" / "monthly_net_qty_distribution.png")
    with c4:
        st.subheader("Top 20 Monthly Demand Outliers")
        show_chart(CHARTS / "outliers" / "top_20_monthly_demand_outliers.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("Feature Engineering")
    st.markdown("Time-aware features computed from raw monthly demand. All use *past data only*.")
    st.divider()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Lag Features")
        st.markdown("""
| Feature | Description |
|---------|-------------|
| `lag_1` | Net quantity previous month |
| `lag_2` | Net quantity 2 months ago |
| `lag_3` | Net quantity 3 months ago |
| `lag_12` | Net quantity same month last year |

`lag_12` captures annual seasonality — requires ≥ 13 months of history,
hence the ≥ 18 active months selection threshold.
        """)

        st.subheader("Rolling Statistics")
        st.markdown("""
| Feature | Description |
|---------|-------------|
| `rolling_mean_3` | Mean demand over previous 3 months |
| `rolling_mean_6` | Mean demand over previous 6 months |
| `rolling_std_3` | Demand std over previous 3 months |

Computed as `shift(1).rolling(n)` — no future data leaks into the window.
        """)

        st.subheader("Return Features")
        st.markdown("""
| Feature | Description |
|---------|-------------|
| `return_ratio_lag_1` | Return rate in previous month |
| `sold_qty_lag_1` | Gross sold qty previous month |
| `return_qty_lag_1` | Return qty previous month |
| `transaction_count_lag_1` | Transaction count previous month |
        """)

    with c2:
        st.subheader("Calendar Features")
        st.markdown("""
| Feature | Description |
|---------|-------------|
| `month_sin` | sin(2π × month / 12) |
| `month_cos` | cos(2π × month / 12) |
| `year` | Calendar year |
| `month_number` | Month (1–12) |
| `quarter` | Quarter (1–4) |

Month encoded as sine/cosine so December (12) and January (1) are numerically adjacent.
        """)

        st.subheader("Train / Validation / Test Split")
        st.markdown("""
| Set | Content | Purpose |
|-----|---------|---------|
| **Train** | All months except last 8 | Model learning |
| **Validation** | Months −8 to −5 | Per-part model selection |
| **Test** | Last 5 months | Final evaluation (held out) |

Strictly time-based. No random shuffling — future months can never appear in training.
        """)

        feat = load_feature_data()
        if feat:
            st.dataframe(feat["summary"], use_container_width=True, hide_index=True)

        with st.expander("Technical note — final model training"):
            st.markdown("""
**Evaluation vs deployment training**

The train/test split was used to measure model accuracy honestly on held-out data.

After evaluation, the saved XGBoost `.pkl` model was **retrained on all available
model-ready historical data** so it can use maximum information for future forecasting.

This is standard practice: evaluation uses held-out test data to measure accuracy,
but the final deployed model should learn from all available history.
The `.pkl` file therefore reflects full historical training, not just the train split.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("Model Comparison")
    st.markdown("All forecasting methods evaluated on the same held-out test set (last 5 months, all 651 parts).")
    st.divider()

    # ── Final metrics table ────────────────────────────────────────────────────
    st.subheader("Final Test Set Metrics — All 651 Parts")

    styled = FINAL_METRICS.style.apply(
        lambda row: ["background-color: #d4edda; font-weight: bold" if row["Note"] == "✅ Best MAE"
                     else "" for _ in row],
        axis=1,
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.warning("""
**Honest finding**: `pred_lag_1` achieved the **lowest MAE** across all 651 parts.

The XGBoost model is a real trained ML model and supports SHAP explainability, but on the
final all-651-parts test it did not outperform the simple lag-1 baseline. This is expected
behaviour when the dataset has many low-volume, high-noise parts where recent history
is the only reliable signal.
    """)

    # ── Interactive bar chart ──────────────────────────────────────────────────
    fig = px.bar(
        FINAL_METRICS, x="Model", y="MAE",
        color="Note",
        color_discrete_map={"✅ Best MAE": "#28a745", "ML model": "#007bff", "": "#6c757d"},
        title="MAE by Model (lower = better)",
        labels={"MAE": "Mean Absolute Error"},
        text="MAE",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False, height=400, yaxis=dict(range=[0, 90]))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost vs Baselines — MAE Chart")
        show_chart(CHARTS / "xgboost_lag12_all651" / "model_comparison_mae.png")
        st.subheader("XGBoost Feature Importance")
        show_chart(CHARTS / "xgboost_lag12_all651" / "feature_importance.png")
        st.caption("`lag_1` and `rolling_mean_3` dominate — recent history is the strongest signal.")

    with col2:
        st.subheader("Hybrid Validation — Test MAE")
        show_chart(CHARTS / "hybrid_validation" / "test_mae_comparison.png")
        st.subheader("Best Model Selected per Part")
        show_chart(CHARTS / "hybrid_validation" / "model_selection_counts.png")

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("XGBoost — Example Part Prediction")
        show_chart(CHARTS / "xgboost_lag12_all651" / "example_part_prediction.png")
    with c4:
        st.subheader("Hybrid — Example Part Prediction")
        show_chart(CHARTS / "hybrid_validation" / "example_part_prediction.png")

    st.divider()
    st.subheader("Model Summary")
    st.markdown("""
| Model | Type | Strength |
|-------|------|----------|
| `pred_lag_1` | **Baseline** | ✅ Best MAE overall; strong when demand is stable |
| `pred_rolling_mean_3` | Baseline | Smooths noise; better for highly volatile parts |
| `pred_part_train_avg` | Baseline | Fallback when recent months are abnormal |
| `xgboost_log_lag12` | **ML** | Captures non-linear patterns & annual seasonality via SHAP |
| `hybrid_validation` | Per-part selection | Routes each part to its best model per validation MAE |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PRODUCTION FORECAST (LAG-1)
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("Production Forecast")
    st.markdown("Customer-ready demand forecasts using the validated production method.")
    st.divider()

    st.success("""
**Recommended production model: `pred_lag_1`**

**Reason**: `pred_lag_1` achieved the lowest MAE on the final all-651-parts test set (MAE = 36.50).

**Logic**: The next-month prediction is based on the latest known monthly demand.
Subsequent months use the previous predicted value as the lag (recursive forecasting).
    """)

    st.warning("""
**Forecast accuracy warning**: Forecasts after the first future month are recursive.
Each step uses the previous prediction as input. Accuracy decreases as the forecast horizon increases.
    """)

    prod_df = load_production_forecasts()
    if prod_df is None:
        st.error("Production forecast file not found: `outputs/xlsx/11_future_forecasts.xlsx`")
        st.stop()

    # ── Part selector ──────────────────────────────────────────────────────────
    st.subheader("Select a Part")

    search = st.text_input("Search by Part Number or description:", "", key="prod_search")

    if search.strip():
        mask = prod_df["search_text"].str.contains(search.strip().lower(), na=False)
        parts_found = prod_df[mask]["Part Number"].unique().tolist()
        if not parts_found:
            st.warning("No parts matched. Try a shorter search term.")
            parts_found = sorted(prod_df["Part Number"].unique())
    else:
        parts_found = sorted(prod_df["Part Number"].unique())

    selected_part = st.selectbox("Part Number:", parts_found, key="prod_part")

    if not selected_part:
        st.stop()

    part_df = prod_df[prod_df["Part Number"] == selected_part].copy()
    desc = part_df["description"].iloc[0] if "description" in part_df.columns else ""
    latest_real = part_df["latest_real_month"].iloc[0] if "latest_real_month" in part_df.columns else "N/A"

    st.markdown(f"### {selected_part} — {desc}")
    st.markdown(f"**Latest real data month**: `{latest_real}`")

    # ── Forecast table ─────────────────────────────────────────────────────────
    st.subheader("Future Demand Forecast")

    show_cols = [c for c in [
        "forecast_month", "forecast_step", "recommended_model",
        "recommended_prediction", "lag_1_used", "lag_2_used", "lag_3_used",
    ] if c in part_df.columns]

    st.dataframe(
        part_df[show_cols].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    # ── Forecast chart ─────────────────────────────────────────────────────────
    st.subheader("Forecast Chart")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=part_df["forecast_month"],
        y=part_df["recommended_prediction"],
        name="Lag-1 Forecast",
        marker_color=[
            "#28a745" if s == 1 else "#ffc107" if s <= 3 else "#dc3545"
            for s in part_df["forecast_step"]
        ],
        text=part_df["recommended_prediction"].round(1),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Monthly Demand Forecast — {selected_part}",
        xaxis_title="Forecast Month",
        yaxis_title="Predicted Net Quantity",
        height=400,
        margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Step 1 = most reliable  |  🟡 Steps 2–3 = moderate caution  |  🔴 Step 4+ = higher uncertainty")

    # ── Lag values used ────────────────────────────────────────────────────────
    with st.expander("Lag input values used for each forecast step"):
        lag_cols = [c for c in ["forecast_month", "forecast_step", "lag_1_used", "lag_2_used", "lag_3_used"]
                    if c in part_df.columns]
        st.dataframe(part_df[lag_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
        st.caption(
            "Step 1 uses real historical lag values. "
            "Later steps use predicted values as inputs — this is the source of recursive uncertainty."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — XGBOOST + SHAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("XGBoost + SHAP")
    st.markdown("Explainable machine-learning forecasting — why did the model predict this number?")
    st.divider()

    # Honesty note
    st.markdown("""<div class="honesty-box">
⚠️ <strong>Honesty note</strong>: XGBoost is included as the explainable ML model and supports SHAP analysis.
However, on the final all-651-parts test set, <code>lag_1</code> had better MAE (36.50 vs 53.72).
<code>lag_1</code> remains the recommended production model.
XGBoost's value here is <strong>explainability</strong>, not raw accuracy.
</div>""", unsafe_allow_html=True)

    # Load model artifacts
    with st.spinner("Loading XGBoost model…"):
        model, features, metadata = load_model_artifacts()

    if model is None:
        st.error("Model files not found in `models/` directory.")
        st.stop()

    xgb_df    = load_xgboost_forecasts()
    shap_df   = load_shap_sample()

    if xgb_df is None or shap_df is None:
        st.error("XGBoost forecast or SHAP sample files not found.")
        st.stop()

    st.markdown(f"""
**Model**: `{metadata.get('model_name', 'XGBoost')}`
**Target**: `{metadata.get('target', 'log1p(net_qty)')}`
**Trained on**: {metadata.get('training_rows', '?'):,} rows
**Features**: {metadata.get('feature_count', len(features))}
**Latest real month**: `{metadata.get('latest_real_month', 'N/A')}`
    """)

    st.divider()

    tab_forecast, tab_shap = st.tabs(["📈 XGBoost Forecast", "💡 SHAP Explanation"])

    # ── Tab 1: XGBoost Forecast ────────────────────────────────────────────────
    with tab_forecast:
        st.subheader("XGBoost Future Demand Forecast")

        search_xgb = st.text_input("Search by Part Number or description:", "", key="xgb_search")

        if search_xgb.strip():
            mask = xgb_df["search_text"].str.contains(search_xgb.strip().lower(), na=False)
            xgb_parts = xgb_df[mask]["Part Number"].unique().tolist()
            if not xgb_parts:
                st.warning("No parts matched.")
                xgb_parts = sorted(xgb_df["Part Number"].unique())
        else:
            xgb_parts = sorted(xgb_df["Part Number"].unique())

        sel_xgb = st.selectbox("Part Number:", xgb_parts, key="xgb_part")
        if not sel_xgb:
            st.stop()

        xgb_part_df = xgb_df[xgb_df["Part Number"] == sel_xgb].copy()
        xgb_desc = xgb_part_df["description"].iloc[0] if "description" in xgb_part_df.columns else ""
        xgb_latest = xgb_part_df["latest_real_month"].iloc[0] if "latest_real_month" in xgb_part_df.columns else "N/A"

        st.markdown(f"### {sel_xgb} — {xgb_desc}")
        st.markdown(f"**Latest real data month**: `{xgb_latest}`")

        # Forecast table
        show_xgb_cols = [c for c in [
            "forecast_month", "forecast_step", "xgboost_prediction", "prediction_log",
            "lag_1_used", "lag_2_used", "lag_3_used", "lag_12_used", "warning",
        ] if c in xgb_part_df.columns]

        st.dataframe(
            xgb_part_df[show_xgb_cols].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

        # Chart
        fig_xgb = go.Figure()
        fig_xgb.add_trace(go.Bar(
            x=xgb_part_df["forecast_month"],
            y=xgb_part_df["xgboost_prediction"],
            name="XGBoost Forecast",
            marker_color=[
                "#007bff" if s == 1 else "#6f42c1" if s <= 3 else "#e83e8c"
                for s in xgb_part_df["forecast_step"]
            ],
            text=xgb_part_df["xgboost_prediction"].round(1),
            textposition="outside",
        ))
        fig_xgb.update_layout(
            title=f"XGBoost Monthly Demand Forecast — {sel_xgb}",
            xaxis_title="Forecast Month", yaxis_title="Predicted Net Quantity",
            height=400, margin=dict(t=50),
        )
        st.plotly_chart(fig_xgb, use_container_width=True)

        if "warning" in xgb_part_df.columns:
            for _, row in xgb_part_df.iterrows():
                if row["forecast_step"] > 1 and "recursive" in str(row.get("warning", "")).lower():
                    st.caption("⚠️ Recursive forecast: later months depend on previous predictions, not real sales.")
                    break

        # Lag table
        with st.expander("Lag input values per forecast step"):
            lag_cols_xgb = [c for c in [
                "forecast_month", "forecast_step",
                "lag_1_used", "lag_2_used", "lag_3_used", "lag_12_used"
            ] if c in xgb_part_df.columns]
            st.dataframe(xgb_part_df[lag_cols_xgb].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    # ── Tab 2: SHAP Explanation ────────────────────────────────────────────────
    with tab_shap:
        st.subheader("SHAP — Why did the model predict this?")

        st.info("""
**SHAP (SHapley Additive exPlanations)** decomposes each XGBoost prediction into
individual feature contributions.

The XGBoost model was trained on `log1p(net_qty)`, so **SHAP values explain the
prediction on the log-demand scale**.

- **Positive SHAP value** → feature pushed the prediction higher
- **Negative SHAP value** → feature pushed the prediction lower
- **|SHAP value|** → size of the feature's impact on this specific prediction
        """)

        # Part selector for SHAP (only parts in shap_input_sample)
        shap_parts = sorted(shap_df["Part Number"].unique())

        search_shap = st.text_input("Search Part Number or description:", "", key="shap_search")
        if search_shap.strip():
            mask_s = (
                shap_df["Part Number"].str.contains(search_shap.strip(), case=False, na=False) |
                shap_df["description"].str.contains(search_shap.strip(), case=False, na=False)
            )
            shap_parts_f = shap_df[mask_s]["Part Number"].unique().tolist()
            shap_parts = shap_parts_f if shap_parts_f else shap_parts

        sel_shap_part = st.selectbox(
            f"Select Part ({len(shap_parts)} available in SHAP sample):",
            shap_parts, key="shap_part"
        )

        if not sel_shap_part:
            st.stop()

        part_shap_rows = shap_df[shap_df["Part Number"] == sel_shap_part].copy()
        part_shap_rows["Month"] = part_shap_rows["Month"].astype(str).str[:7]
        part_shap_rows = part_shap_rows.sort_values("Month")

        shap_desc = part_shap_rows["description"].iloc[0] if "description" in part_shap_rows.columns else ""
        st.markdown(f"**Part**: {sel_shap_part} — {shap_desc}")
        st.markdown(f"Available historical months in sample: **{len(part_shap_rows)}**")

        if part_shap_rows.empty:
            st.warning("No SHAP sample rows found for this part.")
            st.stop()

        # Month selector
        months_available = part_shap_rows["Month"].tolist()
        sel_month = st.selectbox("Select a historical month to explain:", months_available, key="shap_month")

        row_to_explain = part_shap_rows[part_shap_rows["Month"] == sel_month].iloc[[0]]
        X_explain = row_to_explain[features].fillna(0)

        actual_qty  = float(row_to_explain["net_qty"].iloc[0])
        log_pred    = float(model.predict(X_explain)[0])
        qty_pred    = float(np.expm1(log_pred))

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Actual net_qty", f"{actual_qty:.1f}")
        m2.metric("XGBoost Predicted (log)", f"{log_pred:.4f}")
        m3.metric("XGBoost Predicted (quantity)", f"{qty_pred:.1f}")

        st.divider()

        # ── SHAP computation ────────────────────────────────────────────────────
        with st.spinner("Computing SHAP values…"):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer(X_explain)

        sv        = shap_vals.values[0]
        base_val  = float(shap_vals.base_values[0])
        shap_sum  = float(sv.sum())

        st.markdown(f"""
**SHAP decomposition (log scale)**
Base value (avg model output): `{base_val:.4f}`
Sum of SHAP contributions: `{shap_sum:+.4f}`
Final log prediction: `{base_val + shap_sum:.4f}` → `{np.expm1(base_val + shap_sum):.1f}` units
        """)

        st.divider()

        # ── Waterfall plot ──────────────────────────────────────────────────────
        st.subheader("SHAP Waterfall Plot")

        try:
            plt.close("all")
            shap.plots.waterfall(shap_vals[0], max_display=14, show=False)
            fig_wf = plt.gcf()
            fig_wf.set_size_inches(10, 7)
            plt.tight_layout()
            st.pyplot(fig_wf, clear_figure=True)
            plt.close("all")
        except Exception as e:
            st.warning(f"Waterfall plot could not render ({e}). See feature table below.")

        st.caption(
            "Each bar shows how one feature moved the log-prediction up or down from the "
            f"base value ({base_val:.3f}). The final bar is the model's log-prediction."
        )

        # ── Feature contribution table ──────────────────────────────────────────
        st.subheader("Feature Contribution Table")

        contrib_df = pd.DataFrame({
            "Feature":       features,
            "Feature Value": X_explain.values[0],
            "SHAP Value":    sv,
            "|SHAP|":        np.abs(sv),
            "Direction":     ["↑ higher" if v > 0 else "↓ lower" for v in sv],
        }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)

        contrib_df["Feature Value"] = contrib_df["Feature Value"].round(4)
        contrib_df["SHAP Value"]    = contrib_df["SHAP Value"].round(4)
        contrib_df["|SHAP|"]        = contrib_df["|SHAP|"].round(4)

        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        # ── Plotly bar for top features ─────────────────────────────────────────
        top_n = contrib_df.head(12)
        fig_contrib = px.bar(
            top_n, x="SHAP Value", y="Feature", orientation="h",
            color="SHAP Value",
            color_continuous_scale=["#d62728", "#aaaaaa", "#2ca02c"],
            color_continuous_midpoint=0,
            title=f"Top Feature Contributions — {sel_shap_part} | {sel_month}",
            labels={"SHAP Value": "SHAP Value (log scale)", "Feature": ""},
        )
        fig_contrib.update_yaxes(autorange="reversed")
        fig_contrib.update_layout(height=420, margin=dict(l=150))
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.divider()

        # ── Global SHAP charts ──────────────────────────────────────────────────
        st.subheader("Global SHAP Importance (Pre-Computed)")

        c1, c2 = st.columns(2)
        with c1:
            show_chart(CHARTS / "shap" / "shap_global_feature_importance.png",
                       "Mean |SHAP| across all predictions")
        with c2:
            show_chart(CHARTS / "shap" / "shap_summary_plot.png",
                       "Each dot = one prediction; color = feature value")

        st.subheader("SHAP Waterfall — Pre-Computed Example")
        show_chart(CHARTS / "shap" / "shap_waterfall_example.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — FINAL CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.title("Final Conclusion")
    st.markdown("What was built, what was proven, and what value this system delivers.")
    st.divider()

    st.markdown("""<div class="defence-box">
"I did not just build an AI model. I built a forecasting evaluation system
and proved which method works best for this business data."
</div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Core Findings")
        st.markdown("""
**1. The final production forecasting method is `pred_lag_1`.**

After evaluating all models on the held-out test set across all 651 parts,
`pred_lag_1` (next month ≈ last month's actual demand) achieved the lowest MAE of 36.50.
This becomes the recommended production model.

**2. XGBoost was saved as a real `.pkl` model.**

It is used as an explainable ML experiment with SHAP analysis.
It did not beat `lag_1` on the final all-parts test (MAE 53.72 vs 36.50),
but it provides value through interpretability:
SHAP reveals *which features drove each prediction*.

**3. This project does not claim ML always beats simple forecasting.**

Instead, it compares methods **objectively** on real held-out test data
and selects the method that performs best.
Honest evaluation is the core scientific contribution.

**4. SHAP makes the ML model explainable.**

Even though XGBoost was not selected as the production model,
SHAP allows business users to see why a particular forecast was made.
This is essential for building trust in data-driven decisions.
        """)

    with col2:
        st.subheader("System Deliverables")
        st.markdown("""
| Output | Description |
|--------|-------------|
| `06_…_report.xlsx` | Feature dataset with train/test splits |
| `08e_…_report.xlsx` | XGBoost predictions & per-part errors |
| `09_…_report.xlsx` | SHAP feature importance |
| `10b_…_report.xlsx` | Hybrid model test predictions |
| `11_future_forecasts.xlsx` | Production lag-1 forecasts (all 651 parts) |
| `12_xgboost_future_forecasts.xlsx` | XGBoost future forecasts |
| `12_xgboost_shap_input_sample.xlsx` | SHAP historical input sample |
| `models/*.pkl` | Saved XGBoost model for live SHAP |
| `outputs/charts/` | 47 saved PNG charts |
| `app.py` | This interactive dashboard |
        """)

        st.subheader("Forecast Horizon Guidance")
        st.markdown("""
| Horizon | Status | Reason |
|---------|--------|--------|
| Next month (t+1) | ✅ Safe | All lags use real historical data |
| 2–3 months (t+2, t+3) | ⚠️ Caution | Some lags use predicted values |
| 4+ months ahead | ❌ Risky | Most lags are predictions of predictions |

**Recommended production use**: Retrain/refresh monthly on latest actuals,
predict only one month ahead.
        """)

    st.divider()

    st.subheader("Final Metrics Summary")
    styled_final = FINAL_METRICS.style.apply(
        lambda row: ["background-color: #d4edda; font-weight: bold" if row["Note"] == "✅ Best MAE"
                     else "" for _ in row], axis=1,
    )
    st.dataframe(styled_final, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Technical Stack")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
| Component | Technology |
|-----------|------------|
| Data processing | `pandas`, `numpy` |
| Machine learning | `XGBoost` |
| Explainability | `SHAP` |
| Evaluation | `scikit-learn` |
        """)
    with c2:
        st.markdown("""
| Component | Technology |
|-----------|------------|
| Visualisation | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `Streamlit` |
| Source data | Excel Binary `.xlsb` via `pyxlsb` |
| Model storage | `joblib` `.pkl` |
        """)

    with st.expander("Technical note — evaluation split vs final model training"):
        st.markdown("""
**Why the XGBoost `.pkl` model uses more data than the evaluation split**

During evaluation (Step 10b), a strict train/validation/test split was used
to measure model accuracy on truly unseen data. This is the honest, correct way
to report model performance.

After evaluation, the XGBoost model saved in `models/` was **trained on all
available model-ready historical data** (not just the train split). This maximises
the information available for future forecasting.

This is standard practice in machine learning deployment:
evaluate on held-out data → report honest metrics → deploy a model trained on all data.
        """)

    st.divider()
    st.caption(
        "Diploma Project · Automotive Spare Parts Demand Forecasting · "
        "All forecasts loaded from saved Excel reports and `.pkl` model. "
        "No pipeline retraining runs inside this dashboard."
    )
