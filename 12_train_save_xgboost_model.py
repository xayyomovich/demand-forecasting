import json
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# ============================================================
# 12 TRAIN AND SAVE XGBOOST MODEL
# ============================================================
#
# Goal:
# Create a real saved ML model artifact for the Streamlit app.
#
# This script trains the XGBoost log-target model on the all-651-parts
# lag_12 feature-engineered dataset, then saves:
#
# 1. models/xgboost_log_lag12_all651.pkl
# 2. models/xgboost_log_lag12_all651_features.json
# 3. outputs/xlsx/12_xgboost_future_forecasts.xlsx
# 4. outputs/xlsx/12_xgboost_shap_input_sample.xlsx
#
# Important:
# This XGBoost model is NOT the best production model by MAE.
# Our tests showed lag_1 was stronger overall.
#
# We still save XGBoost because:
# - it is a real ML model artifact
# - it supports SHAP explainability
# - it is useful for the diploma ML/explainability section


# -----------------------------
# 1. SETTINGS
# -----------------------------

feature_file = "outputs/xlsx/06_feature_engineering_v3_lag12_all651_report.xlsx"

model_dir = "models"
output_dir = "outputs/xlsx"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

model_file = os.path.join(model_dir, "xgboost_log_lag12_all651.pkl")
feature_json_file = os.path.join(model_dir, "xgboost_log_lag12_all651_features.json")
metadata_json_file = os.path.join(model_dir, "xgboost_log_lag12_all651_metadata.json")

future_output_file = os.path.join(output_dir, "12_xgboost_future_forecasts.xlsx")
shap_sample_file = os.path.join(output_dir, "12_xgboost_shap_input_sample.xlsx")

forecast_horizon_months = 6


# -----------------------------
# 2. LOAD FEATURE-ENGINEERED DATA
# -----------------------------
#
# This file should come from:
#     06_feature_engineering_v3_lag12_all651.py
#
# Expected useful sheet name:
#     model_ready_data
#
# If your sheet name is different, the script prints available sheets.

excel_file = pd.ExcelFile(feature_file)

if "model_ready_data" not in excel_file.sheet_names:
    raise ValueError(
        "Sheet 'model_ready_data' was not found. "
        f"Available sheets: {excel_file.sheet_names}"
    )

model_data = pd.read_excel(feature_file, sheet_name="model_ready_data")

print("Loaded feature-engineered data.")
print("Rows:", len(model_data))
print("Columns:", len(model_data.columns))


# -----------------------------
# 3. DEFINE TARGET AND FEATURES
# -----------------------------
#
# Target:
#     net_qty = monthly demand after returns
#
# We train on log1p(net_qty):
#     log1p(x) = log(1 + x)
#
# Why?
# Demand is very uneven. Some parts sell thousands, many sell only a few.
# Log target reduces the dominance of extreme high-demand parts.

target_col = "net_qty"

identifier_cols = [
    "Part Number",
    "Month",
    "Month Date",
    "description",
    "fr",
]

not_features = [
    target_col,
    "sold_qty",
    "return_qty",
    "transaction_count",
    "sale_value",
    "profit",
] + identifier_cols

feature_cols = [
    col
    for col in model_data.columns
    if col not in not_features and pd.api.types.is_numeric_dtype(model_data[col])
]

if "lag_12" not in feature_cols:
    raise ValueError("lag_12 is missing from feature columns. Run the v3 lag12 feature script first.")

X = model_data[feature_cols].copy()
y = model_data[target_col].clip(lower=0)
y_log = np.log1p(y)


# -----------------------------
# 4. TRAIN FINAL XGBOOST MODEL
# -----------------------------
#
# For model artifact creation, we train on all available model-ready rows.
# This is normal for a final saved model after evaluation is complete.
#
# Evaluation was already done in previous scripts:
# - 08e_xgboost_log_target_lag12_all651.py
# - 10b_hybrid_validation_test_split.py

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

model.fit(X, y_log)


# -----------------------------
# 5. SAVE MODEL ARTIFACTS
# -----------------------------

joblib.dump(model, model_file)

with open(feature_json_file, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=4)

metadata = {
    "model_name": "xgboost_log_lag12_all651",
    "target": "log1p(net_qty)",
    "prediction_inverse": "expm1(prediction_log)",
    "training_rows": int(len(model_data)),
    "feature_count": int(len(feature_cols)),
    "latest_real_month": str(model_data["Month"].max()),
    "forecast_horizon_months": forecast_horizon_months,
    "important_note": (
        "This model is saved for ML forecasting and SHAP explainability. "
        "The lag_1 baseline had better final all-parts MAE."
    ),
}

with open(metadata_json_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)


# -----------------------------
# 6. CREATE FUTURE XGBOOST FORECASTS
# -----------------------------
#
# We recursively forecast future months.
#
# Example:
# - 2026-02 uses real lag values from historical data.
# - 2026-03 uses the 2026-02 XGBoost prediction as part of lag history.
#
# This is useful, but uncertainty increases with every future step.

latest_real_month = str(model_data["Month"].max())

future_months = pd.period_range(
    start=pd.Period(latest_real_month, freq="M") + 1,
    periods=forecast_horizon_months,
    freq="M",
).astype(str)

part_info_cols = ["Part Number", "description", "fr"]
part_info = model_data[part_info_cols].drop_duplicates("Part Number")

future_rows = []

for part_number in sorted(model_data["Part Number"].unique()):
    part_rows = model_data[model_data["Part Number"] == part_number].sort_values("Month").copy()

    if part_rows.empty:
        continue

    description = part_rows["description"].iloc[-1]
    fr = part_rows["fr"].iloc[-1]

    demand_history = part_rows["net_qty"].clip(lower=0).tolist()
    sold_history = part_rows["sold_qty"].clip(lower=0).tolist()
    return_history = part_rows["return_qty"].clip(lower=0).tolist()
    transaction_history = part_rows["transaction_count"].clip(lower=0).tolist()

    # Static part-level features already exist in the engineered dataset.
    # We reuse the latest row values for these features.
    latest_row = part_rows.iloc[-1].copy()

    for step_number, forecast_month in enumerate(future_months, start=1):
        feature_row = latest_row.copy()

        month_period = pd.Period(forecast_month, freq="M")
        month_number = month_period.month
        quarter = ((month_number - 1) // 3) + 1

        lag_1 = demand_history[-1] if len(demand_history) >= 1 else 0
        lag_2 = demand_history[-2] if len(demand_history) >= 2 else 0
        lag_3 = demand_history[-3] if len(demand_history) >= 3 else 0
        lag_12 = demand_history[-12] if len(demand_history) >= 12 else 0

        last_3 = demand_history[-3:]
        last_6 = demand_history[-6:]

        rolling_mean_3 = sum(last_3) / len(last_3) if last_3 else 0
        rolling_mean_6 = sum(last_6) / len(last_6) if last_6 else 0
        rolling_std_3 = float(np.std(last_3, ddof=0)) if len(last_3) >= 2 else 0

        sold_qty_lag_1 = sold_history[-1] if sold_history else 0
        return_qty_lag_1 = return_history[-1] if return_history else 0
        transaction_count_lag_1 = transaction_history[-1] if transaction_history else 0
        return_ratio_lag_1 = return_qty_lag_1 / sold_qty_lag_1 if sold_qty_lag_1 != 0 else 0

        # Calendar features.
        feature_row["year"] = month_period.year
        feature_row["month_number"] = month_number
        feature_row["quarter"] = quarter
        feature_row["month_sin"] = np.sin(2 * np.pi * month_number / 12)
        feature_row["month_cos"] = np.cos(2 * np.pi * month_number / 12)

        # Lag / rolling features.
        feature_row["lag_1"] = lag_1
        feature_row["lag_2"] = lag_2
        feature_row["lag_3"] = lag_3
        feature_row["lag_12"] = lag_12
        feature_row["rolling_mean_3"] = rolling_mean_3
        feature_row["rolling_mean_6"] = rolling_mean_6
        feature_row["rolling_std_3"] = rolling_std_3
        feature_row["sold_qty_lag_1"] = sold_qty_lag_1
        feature_row["return_qty_lag_1"] = return_qty_lag_1
        feature_row["transaction_count_lag_1"] = transaction_count_lag_1
        feature_row["return_ratio_lag_1"] = return_ratio_lag_1

        X_future = pd.DataFrame([feature_row[feature_cols]]).astype(float)

        pred_log = model.predict(X_future)[0]
        pred_qty = float(np.expm1(pred_log))
        pred_qty = max(pred_qty, 0)

        future_rows.append(
            {
                "Part Number": part_number,
                "description": description,
                "fr": fr,
                "forecast_month": forecast_month,
                "forecast_step": step_number,
                "xgboost_prediction": round(pred_qty, 2),
                "prediction_log": round(float(pred_log), 6),
                "latest_real_month": latest_real_month,
                "lag_1_used": round(float(lag_1), 2),
                "lag_2_used": round(float(lag_2), 2),
                "lag_3_used": round(float(lag_3), 2),
                "lag_12_used": round(float(lag_12), 2),
                "warning": (
                    "Recursive forecast: later months depend on previous predictions."
                    if step_number > 1
                    else "First-step forecast uses real historical lag values."
                ),
            }
        )

        # Add XGBoost prediction to history for recursive forecasting.
        demand_history.append(pred_qty)

        # For future months, we do not know sold/return/transaction details.
        # We use simple approximations so later lag fields can continue.
        sold_history.append(pred_qty)
        return_history.append(0)
        transaction_history.append(transaction_count_lag_1)

future_forecasts = pd.DataFrame(future_rows)

future_forecasts["search_text"] = (
    future_forecasts["Part Number"].astype(str)
    + " "
    + future_forecasts["description"].astype(str)
).str.lower()


# -----------------------------
# 7. SAVE SHAP INPUT SAMPLE
# -----------------------------
#
# Streamlit can use this sample to compute/show SHAP explanations.
# It contains real feature rows and identifiers.
#
# For dynamic SHAP in Streamlit:
# - load the .pkl model
# - load feature list JSON
# - select one row from this sample
# - run shap.TreeExplainer(model)

shap_sample = model_data[
    ["Part Number", "Month", "description", "fr", target_col] + feature_cols
].copy()

# Keep file size reasonable for Streamlit.
shap_sample = shap_sample.sort_values(target_col, ascending=False).head(2000)


# -----------------------------
# 8. SAVE EXCEL OUTPUTS
# -----------------------------

with pd.ExcelWriter(future_output_file, engine="openpyxl") as writer:
    future_forecasts.to_excel(writer, sheet_name="xgboost_future_forecasts", index=False)
    pd.DataFrame({"feature": feature_cols}).to_excel(writer, sheet_name="feature_columns", index=False)
    pd.DataFrame([metadata]).to_excel(writer, sheet_name="metadata", index=False)

with pd.ExcelWriter(shap_sample_file, engine="openpyxl") as writer:
    shap_sample.to_excel(writer, sheet_name="shap_input_sample", index=False)
    pd.DataFrame({"feature": feature_cols}).to_excel(writer, sheet_name="feature_columns", index=False)
    pd.DataFrame([metadata]).to_excel(writer, sheet_name="metadata", index=False)


# -----------------------------
# 9. TERMINAL SUMMARY
# -----------------------------

print(f"Saved model: {model_file}")
print(f"Saved feature list: {feature_json_file}")
print(f"Saved metadata: {metadata_json_file}")
print(f"Created: {future_output_file}")
print(f"Created: {shap_sample_file}")
print("Latest real month:", latest_real_month)
print("Forecast months:", ", ".join(future_months))
print("Future forecast rows:", len(future_forecasts))
print("Feature count:", len(feature_cols))
print("Reminder: lag_1 is still the stronger production model by final all-parts MAE.")
