import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

# ============================================================
# 08_xgboost_log_target_v2.py
#
# Purpose:
# Train XGBoost on the upgraded 2023-2026 dataset.
#
# Target:
# log1p(net_qty)
#
# Why log1p:
# Spare-parts demand has extreme values. log1p reduces the
# influence of very large demand rows.
#
# Prediction:
# expm1(prediction_log) converts predictions back to quantity.
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
input_file = "results/xlsx/06b_feature_engineered_v4_no_lag24.csv"
baseline_file = "results/xlsx/07_baseline_forecasting_v2_report.xlsx"

output_folder = "results/xlsx"
chart_folder = "results/charts/xgboost_v2"
model_folder = "models"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(chart_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

output_file = os.path.join(output_folder, "08_xgboost_log_target_v2_report.xlsx")

model_file = os.path.join(model_folder, "xgboost_log_target_v2.pkl")
feature_file = os.path.join(model_folder, "xgboost_log_target_v2_features.json")
metadata_file = os.path.join(model_folder, "xgboost_log_target_v2_metadata.json")


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(input_file, low_memory=False)

train_df = df[df["data_split"] == "train"].copy()
validation_df = df[df["data_split"] == "validation"].copy()
test_df = df[df["data_split"] == "test"].copy()

print("Loaded feature-engineered data.")
print("Train rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))


# ------------------------------------------------------------
# 3. FEATURE COLUMNS
# ------------------------------------------------------------
feature_columns = [
    "year",
    "month_number",
    "quarter",
    "month_sin",
    "month_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_13",

    "sold_qty_lag_1",
    "return_qty_lag_1",
    "transaction_count_lag_1",
    "return_ratio_lag_1",

    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_median_3",
    "rolling_median_6",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_min_6",
    "rolling_max_6",

    "diff_1",
    "diff_3",
    "pct_change_1",

    "sales_months_last_6",
    "sales_months_last_12",
    "months_since_last_sale",
    "zero_demand_streak",

    "part_train_avg_qty",
    "part_train_median_qty",
    "part_train_std_qty",
    "part_train_total_qty",
    "part_train_max_qty",
    "part_train_active_months",
    "part_train_avg_transactions"
]


# ------------------------------------------------------------
# 4. TARGET PREPARATION
# ------------------------------------------------------------
# XGBoost cannot train on negative log values from negative demand.
# Monthly net_qty can occasionally be negative because returns can exceed sales.
#
# For demand forecasting, we clip target at 0:
# negative net demand is treated as zero forecastable demand.
for split_df in [train_df, validation_df, test_df]:
    split_df["target_qty"] = split_df["net_qty"].clip(lower=0)
    split_df["target_log"] = np.log1p(split_df["target_qty"])


X_train = train_df[feature_columns]
y_train = train_df["target_log"]

X_val = validation_df[feature_columns]
y_val = validation_df["target_log"]

X_test = test_df[feature_columns]
y_test_qty = test_df["target_qty"]


# ------------------------------------------------------------
# 5. TRAIN XGBOOST MODEL
# ------------------------------------------------------------
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="rmse"
)

# Note:
# Some XGBoost versions support early_stopping_rounds in fit(),
# some support it in constructor, some older versions do not.
# To avoid version errors, we train fixed estimators here.
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# ------------------------------------------------------------
# 6. PREDICT ON TEST SET
# ------------------------------------------------------------
test_df["pred_xgboost_log_v2_log"] = model.predict(X_test)
test_df["pred_xgboost_log_v2"] = np.expm1(test_df["pred_xgboost_log_v2_log"])
test_df["pred_xgboost_log_v2"] = test_df["pred_xgboost_log_v2"].clip(lower=0)


# ------------------------------------------------------------
# 7. METRIC FUNCTIONS
# ------------------------------------------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    result = np.zeros_like(denominator, dtype=float)

    valid = denominator != 0
    result[valid] = np.abs(y_true[valid] - y_pred[valid]) / denominator[valid]

    return np.mean(result) * 100


def evaluate_prediction(data, prediction_column):
    y_true = data["target_qty"].values
    y_pred = data[prediction_column].values

    return {
        "model": prediction_column,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred)
    }


xgb_metrics = pd.DataFrame([
    evaluate_prediction(test_df, "pred_xgboost_log_v2")
])

print("\nXGBoost v2 metrics:")
print(xgb_metrics.to_string(index=False))


# ------------------------------------------------------------
# 8. LOAD BASELINE METRICS AND COMPARE
# ------------------------------------------------------------
baseline_metrics = pd.read_excel(baseline_file, sheet_name="baseline_metrics")

comparison_metrics = pd.concat(
    [
        baseline_metrics,
        xgb_metrics.rename(columns={"model": "model"})
    ],
    ignore_index=True
).sort_values("MAE")

best_baseline_mae = baseline_metrics["MAE"].min()
xgb_mae = xgb_metrics["MAE"].iloc[0]

improvement_vs_best_baseline = (
    (best_baseline_mae - xgb_mae) / best_baseline_mae
) * 100

print("\nComparison metrics:")
print(comparison_metrics.to_string(index=False))

print("\nImprovement vs best baseline MAE:")
print(f"{improvement_vs_best_baseline:.2f}%")


# ------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------------------------
feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(12, 8))
top_importance = feature_importance.head(20).sort_values("importance")
plt.barh(top_importance["feature"], top_importance["importance"])
plt.title("XGBoost V2 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

feature_chart_path = os.path.join(chart_folder, "xgboost_v2_feature_importance.png")
plt.savefig(feature_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 10. MODEL COMPARISON CHART
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(comparison_metrics["model"], comparison_metrics["MAE"])
plt.title("Model Comparison by MAE - XGBoost V2")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

comparison_chart_path = os.path.join(chart_folder, "xgboost_v2_model_comparison_mae.png")
plt.savefig(comparison_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 11. EXAMPLE PART CHART
# ------------------------------------------------------------
example_part = (
    test_df.groupby("Part Number")["target_qty"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

example_df = test_df[test_df["Part Number"] == example_part].sort_values("Month")

plt.figure(figsize=(12, 6))
plt.plot(example_df["Month"], example_df["target_qty"], marker="o", label="Actual")
plt.plot(example_df["Month"], example_df["pred_xgboost_log_v2"], marker="o", label="XGBoost V2")
plt.plot(example_df["Month"], example_df["lag_1"].clip(lower=0), marker="o", label="Lag 1 Baseline")

plt.title(f"Actual vs XGBoost V2 Prediction: {example_part}")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

example_chart_path = os.path.join(chart_folder, "xgboost_v2_example_part_prediction.png")
plt.savefig(example_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 12. SAVE MODEL ARTIFACTS
# ------------------------------------------------------------
joblib.dump(model, model_file)

with open(feature_file, "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, indent=4)

metadata = {
    "model_name": "xgboost_log_target_v2",
    "data_range": "2023-01 to 2026-01",
    "forecastable_parts": int(df["Part Number"].nunique()),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "test_rows": int(len(test_df)),
    "test_months": sorted(test_df["Month"].unique().tolist()),
    "target": "log1p(net_qty clipped at 0)",
    "feature_count": len(feature_columns),
    "best_baseline_mae": float(best_baseline_mae),
    "xgboost_mae": float(xgb_mae),
    "improvement_vs_best_baseline_percent": float(improvement_vs_best_baseline)
}

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)


# ------------------------------------------------------------
# 13. SAVE EXCEL REPORT
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    xgb_metrics.to_excel(writer, sheet_name="xgboost_metrics", index=False)
    comparison_metrics.to_excel(writer, sheet_name="comparison_metrics", index=False)
    feature_importance.to_excel(writer, sheet_name="feature_importance", index=False)
    test_df.to_excel(writer, sheet_name="test_predictions", index=False)
    example_df.to_excel(writer, sheet_name="example_part", index=False)
    pd.DataFrame([metadata]).to_excel(writer, sheet_name="metadata", index=False)


print("\nCreated:", output_file)
print("Charts saved in:", chart_folder)
print("Saved model:", model_file)
print("Saved feature list:", feature_file)
print("Saved metadata:", metadata_file)
print("Example part:", example_part)





# Loaded feature-engineered data.
# Train rows: 16016
# Validation rows: 3003
# Test rows: 5005

# XGBoost v2 metrics:
#               model       MAE       RMSE     sMAPE
# pred_xgboost_log_v2 25.524848 353.393542 94.054433

# Comparison metrics:
#                  model       MAE       RMSE     sMAPE
#             pred_lag_1 22.828172 222.045448 96.832225
#    pred_xgboost_log_v2 25.524848 353.393542 94.054433
#    pred_rolling_mean_3 27.056210 341.325523 90.602719
# pred_part_train_median 36.717782 487.274604 96.294472
#    pred_rolling_mean_6 36.992607 510.593792 90.576537
#    pred_part_train_avg 40.687995 541.603275 95.977249

# Improvement vs best baseline MAE:
# -11.81%

# Created: results/xlsx\08_xgboost_log_target_v2_report.xlsx
# Charts saved in: results/charts/xgboost_v2
# Saved model: models\xgboost_log_target_v2.pkl
# Saved feature list: models\xgboost_log_target_v2_features.json
# Saved metadata: models\xgboost_log_target_v2_metadata.json
# Example part: MBA000 989 33 09/11ABDW