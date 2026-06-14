import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 1. LOAD FEATURE ENGINEERING OUTPUT -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

input_file = "06_feature_engineering_report.xlsx"

model_ready_data = pd.read_excel(input_file, sheet_name="model_ready_data")
train_data = pd.read_excel(input_file, sheet_name="train_data")
test_data = pd.read_excel(input_file, sheet_name="test_data")

os.makedirs("outputs/charts/baseline", exist_ok=True)

print("Loaded feature engineered data.")
print("Train rows:", len(train_data))
print("Test rows:", len(test_data))



# 2. DEFINE ERROR METRICS -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def mae(y_true, y_pred):
    """
    MAE = Mean Absolute Error.
    It shows the average absolute mistake in units.
    Example: MAE = 20 means model is wrong by about 20 units on average.
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    RMSE = Root Mean Squared Error.
    It penalizes large errors more strongly than MAE.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    """
    sMAPE = Symmetric Mean Absolute Percentage Error.
    It is safer than MAPE when values are small or zero.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero.
    result = np.where(
        denominator == 0,
        0,
        np.abs(y_true - y_pred) / denominator
    )

    return np.mean(result) * 100




# 3. CREATE BASELINE PREDICTIONS -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

baseline_results = test_data.copy()

# Baseline 1:
# Predict this month's demand using previous month's demand.
baseline_results["pred_lag_1"] = baseline_results["lag_1"]

# Baseline 2:
# Predict using the average of the previous 3 months.
baseline_results["pred_rolling_mean_3"] = baseline_results["rolling_mean_3"]

# Baseline 3:
# Predict using each part's historical average demand.
baseline_results["pred_part_avg"] = baseline_results["part_avg_monthly_qty"]

# Optional safety:
# Demand cannot be negative, so baseline predictions below 0 are clipped to 0.
prediction_columns = [
    "pred_lag_1",
    "pred_rolling_mean_3",
    "pred_part_avg"
]

for col in prediction_columns:
    baseline_results[col] = baseline_results[col].clip(lower=0)




# 4. EVALUATE BASELINES -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

y_true = baseline_results["net_qty"]

metrics = []

for col in prediction_columns:
    y_pred = baseline_results[col]

    metrics.append({
        "model": col,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred)
    })

metrics_df = pd.DataFrame(metrics).sort_values("MAE")



# 5. PER-PART BASELINE ERROR -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

per_part_errors = []

for part_number, part_df in baseline_results.groupby("Part Number"):
    for col in prediction_columns:
        per_part_errors.append({
            "Part Number": part_number,
            "description": part_df["description"].iloc[0],
            "baseline_model": col,
            "MAE": mae(part_df["net_qty"], part_df[col]),
            "RMSE": rmse(part_df["net_qty"], part_df[col]),
            "sMAPE": smape(part_df["net_qty"], part_df[col])
        })

per_part_errors_df = pd.DataFrame(per_part_errors)



# 6. CHART: BASELINE METRICS COMPARISON -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df, x="model", y="MAE")
plt.title("Baseline Models Comparison by MAE")
plt.xlabel("Baseline Model")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/baseline/baseline_mae_comparison.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df, x="model", y="RMSE")
plt.title("Baseline Models Comparison by RMSE")
plt.xlabel("Baseline Model")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/charts/baseline/baseline_rmse_comparison.png", dpi=300)
plt.close()



# 7. CHART: ACTUAL VS BASELINE FOR ONE EXAMPLE PART -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Choose the first part in test data as an example.
example_part = baseline_results["Part Number"].iloc[0]

example_df = baseline_results[
    baseline_results["Part Number"] == example_part
].sort_values("Month")

plt.figure(figsize=(12, 6))
plt.plot(example_df["Month"], example_df["net_qty"], marker="o", label="Actual")
plt.plot(example_df["Month"], example_df["pred_lag_1"], marker="o", label="Lag 1 Baseline")
plt.plot(example_df["Month"], example_df["pred_rolling_mean_3"], marker="o", label="Rolling Mean 3 Baseline")
plt.plot(example_df["Month"], example_df["pred_part_avg"], marker="o", label="Part Average Baseline")
plt.title(f"Actual vs Baseline Forecasts: {example_part}")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/baseline/example_part_baseline_forecast.png", dpi=300)
plt.close()



# 8. SAVE REPORT -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

output_file = "07_baseline_forecasting_report.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="baseline_metrics", index=False)
    baseline_results.to_excel(writer, sheet_name="baseline_predictions", index=False)
    per_part_errors_df.to_excel(writer, sheet_name="per_part_errors", index=False)

print(f"Created: {output_file}")
print("Charts saved in: outputs/charts/baseline/")
print("\nBaseline metrics:")
print(metrics_df)