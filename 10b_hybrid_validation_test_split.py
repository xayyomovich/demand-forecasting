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

model_ready_data = pd.read_excel(input_file, sheet_name="model_ready_data")

os.makedirs("outputs/charts/hybrid_validation", exist_ok=True)
os.makedirs("outputs/xlsx", exist_ok=True)

print("Total rows:", len(model_ready_data))


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
# 3. TIME SPLIT: TRAIN / VALIDATION / TEST
# ============================================================

all_months = sorted(model_ready_data["Month"].unique())

test_months = all_months[-5:]
validation_months = all_months[-8:-5]
train_months = all_months[:-8]

train_data = model_ready_data[
    model_ready_data["Month"].isin(train_months)
].copy()

validation_data = model_ready_data[
    model_ready_data["Month"].isin(validation_months)
].copy()

test_data = model_ready_data[
    model_ready_data["Month"].isin(test_months)
].copy()

print("Train months:", train_months)
print("Validation months:", validation_months)
print("Test months:", test_months)

print("Train rows:", len(train_data))
print("Validation rows:", len(validation_data))
print("Test rows:", len(test_data))


# ============================================================
# 4. SAFE PART FEATURES FROM TRAIN ONLY
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
validation_data = validation_data.merge(part_train_features, on="Part Number", how="left")
test_data = test_data.merge(part_train_features, on="Part Number", how="left")


# ============================================================
# 5. TRAIN XGBOOST LOG MODEL
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
# 6. CREATE PREDICTIONS FUNCTION
# ============================================================

def add_predictions(data):
    data = data.copy()

    X = data[feature_columns]

    data["pred_lag_1"] = data["lag_1"].clip(lower=0)
    data["pred_rolling_mean_3"] = data["rolling_mean_3"].clip(lower=0)
    data["pred_part_train_avg"] = data["part_train_avg_qty"].clip(lower=0)

    xgb_log_pred = model.predict(X)
    data["pred_xgboost_log_lag12"] = np.expm1(xgb_log_pred).clip(min=0)

    return data


validation_results = add_predictions(validation_data)
test_results = add_predictions(test_data)

prediction_columns = [
    "pred_lag_1",
    "pred_rolling_mean_3",
    "pred_part_train_avg",
    "pred_xgboost_log_lag12"
]


# ============================================================
# 7. SELECT BEST MODEL PER PART USING VALIDATION ONLY
# ============================================================

validation_scores = []

for part_number, part_df in validation_results.groupby("Part Number"):
    for col in prediction_columns:
        validation_scores.append({
            "Part Number": part_number,
            "description": part_df["description"].iloc[0],
            "model": col,
            "validation_MAE": mean_absolute_error(part_df["net_qty"], part_df[col]),
            "validation_RMSE": rmse(part_df["net_qty"], part_df[col]),
            "validation_sMAPE": smape(part_df["net_qty"], part_df[col])
        })

validation_scores_df = pd.DataFrame(validation_scores)

best_model_per_part = (
    validation_scores_df.sort_values(["Part Number", "validation_MAE"])
    .groupby("Part Number", as_index=False)
    .first()
    .rename(columns={"model": "selected_model"})
)


# ============================================================
# 8. APPLY SELECTED MODEL TO TEST SET
# ============================================================

test_results = test_results.merge(
    best_model_per_part[["Part Number", "selected_model"]],
    on="Part Number",
    how="left"
)

# If a part somehow has no selected model, default to lag_1.
test_results["selected_model"] = test_results["selected_model"].fillna("pred_lag_1")

def choose_prediction(row):
    return row[row["selected_model"]]

test_results["hybrid_prediction"] = test_results.apply(choose_prediction, axis=1)


# ============================================================
# 9. TEST METRICS
# ============================================================

test_model_metrics = []

for col in prediction_columns:
    test_model_metrics.append({
        "model": col,
        "MAE": mean_absolute_error(test_results["net_qty"], test_results[col]),
        "RMSE": rmse(test_results["net_qty"], test_results[col]),
        "sMAPE": smape(test_results["net_qty"], test_results[col])
    })

test_model_metrics_df = pd.DataFrame(test_model_metrics)

hybrid_metrics = pd.DataFrame({
    "model": ["hybrid_validation_selected"],
    "MAE": [mean_absolute_error(test_results["net_qty"], test_results["hybrid_prediction"])],
    "RMSE": [rmse(test_results["net_qty"], test_results["hybrid_prediction"])],
    "sMAPE": [smape(test_results["net_qty"], test_results["hybrid_prediction"])]
})

comparison_metrics = pd.concat(
    [test_model_metrics_df, hybrid_metrics],
    ignore_index=True
).sort_values("MAE")


# ============================================================
# 10. MODEL SELECTION COUNTS
# ============================================================

model_selection_counts = (
    best_model_per_part["selected_model"]
    .value_counts()
    .reset_index()
)

model_selection_counts.columns = ["selected_model", "part_count"]


# ============================================================
# 11. CHARTS
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_metrics, x="model", y="MAE")
plt.title("Validation-Selected Hybrid Model - Test MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/hybrid_validation/test_mae_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(data=model_selection_counts, x="selected_model", y="part_count")
plt.title("Validation-Selected Best Model Counts")
plt.xlabel("Selected Model")
plt.ylabel("Number of Parts")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/hybrid_validation/model_selection_counts.png", dpi=300)
plt.close()


# Example part: highest actual total in test set.
example_part = (
    test_results.groupby("Part Number")["net_qty"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

example_df = test_results[test_results["Part Number"] == example_part].sort_values("Month")

plt.figure(figsize=(12, 6))
plt.plot(example_df["Month"], example_df["net_qty"], marker="o", label="Actual")
plt.plot(example_df["Month"], example_df["hybrid_prediction"], marker="o", label="Hybrid Prediction")
plt.title(f"Actual vs Validation-Selected Hybrid Prediction: {example_part}")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/hybrid_validation/example_part_prediction.png", dpi=300)
plt.close()


# ============================================================
# 12. SAVE REPORT
# ============================================================

output_file = "outputs/xlsx/10b_hybrid_validation_test_split_report.xlsx"

split_summary = pd.DataFrame({
    "metric": [
        "train_months",
        "validation_months",
        "test_months",
        "train_rows",
        "validation_rows",
        "test_rows"
    ],
    "value": [
        ", ".join(train_months),
        ", ".join(validation_months),
        ", ".join(test_months),
        len(train_data),
        len(validation_data),
        len(test_data)
    ]
})

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    split_summary.to_excel(writer, sheet_name="split_summary", index=False)
    comparison_metrics.to_excel(writer, sheet_name="test_comparison_metrics", index=False)
    validation_scores_df.to_excel(writer, sheet_name="validation_model_scores", index=False)
    best_model_per_part.to_excel(writer, sheet_name="best_model_per_part", index=False)
    model_selection_counts.to_excel(writer, sheet_name="model_selection_counts", index=False)
    test_results.to_excel(writer, sheet_name="test_predictions", index=False)
    part_train_features.to_excel(writer, sheet_name="safe_part_features", index=False)

print(f"Created: {output_file}")
print("\nTest comparison metrics:")
print(comparison_metrics)
print("\nModel selection counts:")
print(model_selection_counts)
print("Charts saved in: outputs/charts/hybrid_validation/")