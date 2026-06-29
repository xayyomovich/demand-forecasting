import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 07_baseline_forecasting_v2.py
#
# Purpose:
# Evaluate simple benchmark forecasting methods on the new
# 2023-2026 feature-engineered dataset.
#
# Why:
# XGBoost must be compared against simple baselines.
# If it cannot beat these, we should not call it better.
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
input_file = "results/xlsx/06b_feature_engineered_v4_no_lag24.csv"

output_folder = "results/xlsx"
chart_folder = "results/charts/baseline_v2"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(chart_folder, exist_ok=True)

output_file = os.path.join(output_folder, "07_baseline_forecasting_v2_report.xlsx")


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(input_file, low_memory=False)

print("Loaded feature-engineered data.")
print("Rows:", len(df))

test_df = df[df["data_split"] == "test"].copy()

print("Test rows:", len(test_df))
print("Test months:", sorted(test_df["Month"].unique()))


# ------------------------------------------------------------
# 3. METRIC FUNCTIONS
# ------------------------------------------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    result = np.where(
        denominator == 0,
        0,
        np.abs(y_true - y_pred) / denominator
    )
    return np.mean(result) * 100


def evaluate_model(data, prediction_column):
    y_true = data["net_qty"].values
    y_pred = data[prediction_column].values

    return {
        "model": prediction_column,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred)
    }


# ------------------------------------------------------------
# 4. BASELINE PREDICTIONS
# ------------------------------------------------------------
# Simple benchmark models:
#
# pred_lag_1:
#   next month = previous month
#
# pred_rolling_mean_3:
#   next month = average of last 3 months
#
# pred_rolling_mean_6:
#   next month = average of last 6 months
#
# pred_part_train_avg:
#   next month = historical train average of that part
#
# pred_part_train_median:
#   next month = historical train median of that part

test_df["pred_lag_1"] = test_df["lag_1"]
test_df["pred_rolling_mean_3"] = test_df["rolling_mean_3"]
test_df["pred_rolling_mean_6"] = test_df["rolling_mean_6"]
test_df["pred_part_train_avg"] = test_df["part_train_avg_qty"]
test_df["pred_part_train_median"] = test_df["part_train_median_qty"]

# Forecasts cannot be negative demand.
baseline_columns = [
    "pred_lag_1",
    "pred_rolling_mean_3",
    "pred_rolling_mean_6",
    "pred_part_train_avg",
    "pred_part_train_median"
]

for col in baseline_columns:
    test_df[col] = test_df[col].clip(lower=0)


# ------------------------------------------------------------
# 5. EVALUATE BASELINES
# ------------------------------------------------------------
metrics = pd.DataFrame([
    evaluate_model(test_df, col)
    for col in baseline_columns
]).sort_values("MAE")

print("\nBaseline metrics:")
print(metrics.to_string(index=False))


# ------------------------------------------------------------
# 6. PER-PART ERROR TABLE
# ------------------------------------------------------------
per_part_errors = []

for part_number, part_data in test_df.groupby("Part Number"):
    row = {
        "Part Number": part_number,
        "description": part_data["description"].iloc[0],
        "fr": part_data["fr"].iloc[0],
        "test_months": part_data["Month"].nunique(),
        "actual_total_net_qty": part_data["net_qty"].sum()
    }

    for col in baseline_columns:
        row[f"{col}_MAE"] = mae(part_data["net_qty"].values, part_data[col].values)

    per_part_errors.append(row)

per_part_errors = pd.DataFrame(per_part_errors)

# Best baseline per part
mae_cols = [f"{col}_MAE" for col in baseline_columns]

per_part_errors["best_baseline"] = (
    per_part_errors[mae_cols]
    .idxmin(axis=1)
    .str.replace("_MAE", "", regex=False)
)

per_part_errors["best_baseline_mae"] = per_part_errors[mae_cols].min(axis=1)

best_baseline_counts = (
    per_part_errors["best_baseline"]
    .value_counts()
    .reset_index()
)

best_baseline_counts.columns = ["baseline_model", "part_count"]


# ------------------------------------------------------------
# 7. EXAMPLE PART CHART
# ------------------------------------------------------------
# Choose the part with highest actual test demand.
example_part = (
    test_df.groupby("Part Number")["net_qty"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

example_df = test_df[test_df["Part Number"] == example_part].sort_values("Month")

plt.figure(figsize=(12, 6))
plt.plot(example_df["Month"], example_df["net_qty"], marker="o", label="Actual")
plt.plot(example_df["Month"], example_df["pred_lag_1"], marker="o", label="Lag 1")
plt.plot(example_df["Month"], example_df["pred_rolling_mean_3"], marker="o", label="Rolling Mean 3")
plt.plot(example_df["Month"], example_df["pred_rolling_mean_6"], marker="o", label="Rolling Mean 6")
plt.plot(example_df["Month"], example_df["pred_part_train_avg"], marker="o", label="Part Train Avg")

plt.title(f"Actual vs Baseline Forecasts: {example_part}")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

example_chart_path = os.path.join(chart_folder, "example_part_baseline_v2.png")
plt.savefig(example_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 8. METRIC BAR CHART
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(metrics["model"], metrics["MAE"])
plt.title("Baseline Model Comparison by MAE - V2")
plt.xlabel("Baseline Model")
plt.ylabel("MAE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

mae_chart_path = os.path.join(chart_folder, "baseline_mae_comparison_v2.png")
plt.savefig(mae_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 9. BEST BASELINE COUNT CHART
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(best_baseline_counts["baseline_model"], best_baseline_counts["part_count"])
plt.title("Best Baseline Model Count by Part - V2")
plt.xlabel("Baseline Model")
plt.ylabel("Number of Parts")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

count_chart_path = os.path.join(chart_folder, "best_baseline_counts_v2.png")
plt.savefig(count_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 10. SAVE REPORT
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    metrics.to_excel(writer, sheet_name="baseline_metrics", index=False)
    test_df.to_excel(writer, sheet_name="test_predictions", index=False)
    per_part_errors.to_excel(writer, sheet_name="per_part_errors", index=False)
    best_baseline_counts.to_excel(writer, sheet_name="best_baseline_counts", index=False)
    example_df.to_excel(writer, sheet_name="example_part", index=False)


print("\nCreated:", output_file)
print("Charts saved in:", chart_folder)
print("Example part:", example_part)



# Loaded feature-engineered data.
# Rows: 24024
# Test rows: 5005
# Test months: ['2025-09', '2025-10', '2025-11', '2025-12', '2026-01']
# D:\Diploma_work\diploma\diploma_v2\07_baseline_forecasting_v2.py:63: RuntimeWarning: invalid value encountered in divide
#   np.abs(y_true - y_pred) / denominator

# Baseline metrics:
#                  model       MAE       RMSE     sMAPE
#             pred_lag_1 22.828172 222.045448 96.832225
#    pred_rolling_mean_3 27.056210 341.325523 90.602719
# pred_part_train_median 36.717782 487.274604 96.294472
#    pred_rolling_mean_6 36.992607 510.593792 90.576537
#    pred_part_train_avg 40.687995 541.603275 95.977249

# Created: results/xlsx\07_baseline_forecasting_v2_report.xlsx
# Charts saved in: results/charts/baseline_v2
# Example part: MBA000 989 33 09/11ABDW