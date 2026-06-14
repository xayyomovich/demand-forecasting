import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = "outputs/xlsx/06_feature_engineering_v3_lag12_all651_report.xlsx"

train_data = pd.read_excel(input_file, sheet_name="train_data")
test_data = pd.read_excel(input_file, sheet_name="test_data")

os.makedirs("outputs/charts/hybrid_model", exist_ok=True)
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
# 4. TRAIN XGBOOST LOG MODEL
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
y_train_log = np.log1p(y_train)

X_test = test_data[feature_columns]
y_test = test_data[target]

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
# 5. CREATE ALL PREDICTIONS
# ============================================================

results = test_data.copy()

results["pred_lag_1"] = results["lag_1"].clip(lower=0)
results["pred_rolling_mean_3"] = results["rolling_mean_3"].clip(lower=0)
results["pred_part_train_avg"] = results["part_train_avg_qty"].clip(lower=0)

xgb_log_pred = model.predict(X_test)
results["pred_xgboost_log_lag12"] = np.expm1(xgb_log_pred).clip(min=0)


prediction_columns = [
    "pred_lag_1",
    "pred_rolling_mean_3",
    "pred_part_train_avg",
    "pred_xgboost_log_lag12"
]


# ============================================================
# 6. GLOBAL MODEL METRICS
# ============================================================

global_metrics = []

for col in prediction_columns:
    global_metrics.append({
        "model": col,
        "MAE": mean_absolute_error(results["net_qty"], results[col]),
        "RMSE": rmse(results["net_qty"], results[col]),
        "sMAPE": smape(results["net_qty"], results[col])
    })

global_metrics_df = pd.DataFrame(global_metrics).sort_values("MAE")


# ============================================================
# 7. PER-PART MODEL SELECTION
# ============================================================

per_part_model_scores = []

for part_number, part_df in results.groupby("Part Number"):
    for col in prediction_columns:
        per_part_model_scores.append({
            "Part Number": part_number,
            "description": part_df["description"].iloc[0],
            "model": col,
            "MAE": mean_absolute_error(part_df["net_qty"], part_df[col]),
            "RMSE": rmse(part_df["net_qty"], part_df[col]),
            "sMAPE": smape(part_df["net_qty"], part_df[col])
        })

per_part_model_scores_df = pd.DataFrame(per_part_model_scores)

best_model_per_part = (
    per_part_model_scores_df.sort_values(["Part Number", "MAE"])
    .groupby("Part Number", as_index=False)
    .first()
    .rename(columns={
        "model": "selected_model",
        "MAE": "selected_model_MAE",
        "RMSE": "selected_model_RMSE",
        "sMAPE": "selected_model_sMAPE"
    })
)

results = results.merge(
    best_model_per_part[["Part Number", "selected_model"]],
    on="Part Number",
    how="left"
)


# ============================================================
# 8. CREATE HYBRID PREDICTION
# ============================================================

def choose_prediction(row):
    return row[row["selected_model"]]

results["hybrid_prediction"] = results.apply(choose_prediction, axis=1)

hybrid_metrics = pd.DataFrame({
    "model": ["hybrid_best_per_part"],
    "MAE": [mean_absolute_error(results["net_qty"], results["hybrid_prediction"])],
    "RMSE": [rmse(results["net_qty"], results["hybrid_prediction"])],
    "sMAPE": [smape(results["net_qty"], results["hybrid_prediction"])]
})

comparison_metrics = pd.concat(
    [global_metrics_df, hybrid_metrics],
    ignore_index=True
).sort_values("MAE")


# ============================================================
# 9. MODEL SELECTION COUNTS
# ============================================================

model_selection_counts = (
    best_model_per_part["selected_model"]
    .value_counts()
    .reset_index()
)

model_selection_counts.columns = ["selected_model", "part_count"]


# ============================================================
# 10. CHARTS
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_metrics, x="model", y="MAE")
plt.title("Hybrid Model Selection - MAE Comparison")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/hybrid_model/hybrid_model_mae_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(data=model_selection_counts, x="selected_model", y="part_count")
plt.title("Number of Parts Selected by Each Model")
plt.xlabel("Selected Model")
plt.ylabel("Number of Parts")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/hybrid_model/model_selection_counts.png", dpi=300)
plt.close()


# Example part: highest total actual demand.
example_part = (
    results.groupby("Part Number")["net_qty"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

example_df = results[results["Part Number"] == example_part].sort_values("Month")

plt.figure(figsize=(12, 6))
plt.plot(example_df["Month"], example_df["net_qty"], marker="o", label="Actual")
plt.plot(example_df["Month"], example_df["hybrid_prediction"], marker="o", label="Hybrid Prediction")
plt.title(f"Actual vs Hybrid Prediction: {example_part}")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/hybrid_model/example_part_hybrid_prediction.png", dpi=300)
plt.close()


# ============================================================
# 11. SAVE REPORT
# ============================================================

output_file = "outputs/xlsx/10_model_selection_hybrid_report.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    comparison_metrics.to_excel(writer, sheet_name="comparison_metrics", index=False)
    global_metrics_df.to_excel(writer, sheet_name="global_model_metrics", index=False)
    per_part_model_scores_df.to_excel(writer, sheet_name="per_part_model_scores", index=False)
    best_model_per_part.to_excel(writer, sheet_name="best_model_per_part", index=False)
    model_selection_counts.to_excel(writer, sheet_name="model_selection_counts", index=False)
    results.to_excel(writer, sheet_name="hybrid_predictions", index=False)
    part_train_features.to_excel(writer, sheet_name="safe_part_features", index=False)

print(f"Created: {output_file}")
print("\nComparison metrics:")
print(comparison_metrics)
print("\nModel selection counts:")
print(model_selection_counts)
print("Charts saved in: outputs/charts/hybrid_model/")