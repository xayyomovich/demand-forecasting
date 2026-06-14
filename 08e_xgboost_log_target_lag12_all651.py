import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD LAG-12 ALL-PARTS DATA
# ============================================================

input_file = "outputs/xlsx/06_feature_engineering_v3_lag12_all651_report.xlsx"

train_data = pd.read_excel(input_file, sheet_name="train_data")
test_data = pd.read_excel(input_file, sheet_name="test_data")

os.makedirs("outputs/charts/xgboost_lag12_all651", exist_ok=True)
os.makedirs("outputs/xlsx", exist_ok=True)

print("Train rows:", len(train_data))
print("Test rows:", len(test_data))


# ============================================================
# 2. METRICS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    result = np.where(
        denominator == 0,
        0,
        np.abs(y_true - y_pred) / denominator
    )

    return np.mean(result) * 100


# ============================================================
# 3. SAFE PART-LEVEL FEATURES FROM TRAIN ONLY
# ============================================================

part_train_features = (
    train_data.groupby("Part Number", as_index=False)
    .agg(
        part_train_avg_qty=("net_qty", "mean"),
        part_train_median_qty=("net_qty", "median"),
        part_train_max_qty=("net_qty", "max"),
        part_train_std_qty=("net_qty", "std"),
        part_train_total_qty=("net_qty", "sum"),
        part_train_avg_transactions=("transaction_count", "mean")
    )
)

part_train_features["part_train_std_qty"] = part_train_features["part_train_std_qty"].fillna(0)

train_data = train_data.merge(part_train_features, on="Part Number", how="left")
test_data = test_data.merge(part_train_features, on="Part Number", how="left")


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

target = "net_qty"

feature_columns = [
    "year",
    "month_number",
    "quarter",
    "month_sin",
    "month_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",

    "sold_qty_lag_1",
    "return_qty_lag_1",
    "transaction_count_lag_1",

    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",

    "return_ratio_lag_1",

    "part_train_avg_qty",
    "part_train_median_qty",
    "part_train_max_qty",
    "part_train_std_qty",
    "part_train_total_qty",
    "part_train_avg_transactions"
]

X_train = train_data[feature_columns]
y_train = train_data[target]

X_test = test_data[feature_columns]
y_test = test_data[target]

y_train_log = np.log1p(y_train)


# ============================================================
# 5. TRAIN XGBOOST LOG MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    reg_alpha=0.2,
    random_state=42,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1
)

model.fit(X_train, y_train_log)


# ============================================================
# 6. PREDICT
# ============================================================

test_results = test_data.copy()

log_predictions = model.predict(X_test)

test_results["xgboost_lag12_prediction"] = np.expm1(log_predictions)
test_results["xgboost_lag12_prediction"] = test_results["xgboost_lag12_prediction"].clip(lower=0)


# ============================================================
# 7. EVALUATE
# ============================================================

xgb_mae = mean_absolute_error(y_test, test_results["xgboost_lag12_prediction"])
xgb_rmse = rmse(y_test, test_results["xgboost_lag12_prediction"])
xgb_smape = smape(y_test, test_results["xgboost_lag12_prediction"])

xgb_metrics = pd.DataFrame({
    "model": ["xgboost_log_lag12_all651"],
    "MAE": [xgb_mae],
    "RMSE": [xgb_rmse],
    "sMAPE": [xgb_smape]
})


# ============================================================
# 8. BASELINES ON SAME TEST SET
# ============================================================

# Important:
# Because this all-651 lag12 dataset has a different test set than Top 50,
# we must calculate baselines again on this same test_data.

baseline_results = test_data.copy()

baseline_results["pred_lag_1"] = baseline_results["lag_1"].clip(lower=0)
baseline_results["pred_rolling_mean_3"] = baseline_results["rolling_mean_3"].clip(lower=0)

# Safe part average baseline from train period only.
part_avg_baseline = (
    train_data.groupby("Part Number", as_index=False)
    .agg(pred_part_train_avg=("net_qty", "mean"))
)

baseline_results = baseline_results.merge(part_avg_baseline, on="Part Number", how="left")
baseline_results["pred_part_train_avg"] = baseline_results["pred_part_train_avg"].clip(lower=0)

baseline_prediction_columns = [
    "pred_lag_1",
    "pred_rolling_mean_3",
    "pred_part_train_avg"
]

baseline_metrics = []

for col in baseline_prediction_columns:
    baseline_metrics.append({
        "model": col,
        "MAE": mean_absolute_error(baseline_results["net_qty"], baseline_results[col]),
        "RMSE": rmse(baseline_results["net_qty"], baseline_results[col]),
        "sMAPE": smape(baseline_results["net_qty"], baseline_results[col])
    })

baseline_metrics = pd.DataFrame(baseline_metrics)


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

comparison_metrics = pd.concat(
    [baseline_metrics, xgb_metrics],
    ignore_index=True
).sort_values("MAE")

best_baseline_mae = baseline_metrics["MAE"].min()

improvement_vs_best_baseline = (
    (best_baseline_mae - xgb_mae) / best_baseline_mae * 100
)

print("\nXGBoost lag12 all651 metrics:")
print(xgb_metrics)

print("\nComparison metrics:")
print(comparison_metrics)

print("\nImprovement vs best baseline MAE:")
print(f"{improvement_vs_best_baseline:.2f}%")


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)


# ============================================================
# 11. PER-PART ERROR
# ============================================================

per_part_errors = []

for part_number, part_df in test_results.groupby("Part Number"):
    per_part_errors.append({
        "Part Number": part_number,
        "description": part_df["description"].iloc[0],
        "MAE": mean_absolute_error(part_df["net_qty"], part_df["xgboost_lag12_prediction"]),
        "RMSE": rmse(part_df["net_qty"], part_df["xgboost_lag12_prediction"]),
        "sMAPE": smape(part_df["net_qty"], part_df["xgboost_lag12_prediction"]),
        "actual_total": part_df["net_qty"].sum(),
        "predicted_total": part_df["xgboost_lag12_prediction"].sum()
    })

per_part_errors_df = pd.DataFrame(per_part_errors).sort_values("MAE", ascending=False)


# ============================================================
# 12. CHARTS
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_metrics, x="model", y="MAE")
plt.title("Model Comparison by MAE - Lag 12 All Forecastable Parts")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/xgboost_lag12_all651/model_comparison_mae.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x="importance", y="feature")
plt.title("XGBoost Lag12 All651 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("outputs/charts/xgboost_lag12_all651/feature_importance.png", dpi=300)
plt.close()

# Example part: highest actual total in test set.
example_part = (
    test_results.groupby("Part Number")["net_qty"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

example_df = test_results[
    test_results["Part Number"] == example_part
].sort_values("Month")

plt.figure(figsize=(12, 6))
plt.plot(example_df["Month"], example_df["net_qty"], marker="o", label="Actual")
plt.plot(example_df["Month"], example_df["xgboost_lag12_prediction"], marker="o", label="XGBoost Prediction")
plt.title(f"Actual vs XGBoost Lag12 Prediction: {example_part}")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/xgboost_lag12_all651/example_part_prediction.png", dpi=300)
plt.close()


# ============================================================
# 13. SAVE REPORT
# ============================================================

output_file = "outputs/xlsx/08e_xgboost_log_target_lag12_all651_report.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    comparison_metrics.to_excel(writer, sheet_name="model_comparison", index=False)
    xgb_metrics.to_excel(writer, sheet_name="xgboost_metrics", index=False)
    baseline_metrics.to_excel(writer, sheet_name="baseline_metrics_same_test", index=False)
    test_results.to_excel(writer, sheet_name="xgboost_predictions", index=False)
    feature_importance.to_excel(writer, sheet_name="feature_importance", index=False)
    per_part_errors_df.to_excel(writer, sheet_name="per_part_errors", index=False)
    part_train_features.to_excel(writer, sheet_name="safe_part_features", index=False)

print(f"\nCreated: {output_file}")
print("Charts saved in: outputs/charts/xgboost_lag12_all651/")
