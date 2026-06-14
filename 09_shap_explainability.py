import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from xgboost import XGBRegressor


# 1. LOAD DATA
# ============================================================

input_file = "outputs/xlsx/06_feature_engineering_v2_report.xlsx"

train_data = pd.read_excel(input_file, sheet_name="train_data")
test_data = pd.read_excel(input_file, sheet_name="test_data")

os.makedirs("outputs/charts/shap", exist_ok=True)
os.makedirs("outputs/xlsx", exist_ok=True)

print("Train rows:", len(train_data))
print("Test rows:", len(test_data))


# ============================================================
# 2. CREATE SAFE PART-LEVEL FEATURES FROM TRAIN ONLY

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
# 3. FEATURES AND TARGET

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
# 4. TRAIN FINAL XGBOOST LOG MODEL

model = XGBRegressor(
    n_estimators=300,
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

test_results = test_data.copy()

log_predictions = model.predict(X_test)
test_results["prediction_log"] = log_predictions
test_results["prediction_qty"] = np.expm1(log_predictions).clip(min=0)


# ============================================================
# 5. CREATE SHAP EXPLAINER

# SHAP explains the model output.
# Because this model was trained on log target,
# SHAP values explain changes in log-demand, not direct unit demand.
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("SHAP values created.")


# ============================================================
# 6. GLOBAL SHAP SUMMARY PLOT

plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_columns,
    show=False
)
plt.tight_layout()
plt.savefig("outputs/charts/shap/shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 7. GLOBAL SHAP BAR PLOT

plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_columns,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("outputs/charts/shap/shap_global_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 8. SHAP IMPORTANCE TABLE

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)


# ============================================================
# 9. LOCAL EXPLANATION FOR ONE FORECAST

# Choose the row with the largest actual demand in test set.
example_index = test_results["net_qty"].idxmax()
example_position = test_results.index.get_loc(example_index)

example_row = test_results.iloc[example_position]
example_features = X_test.iloc[example_position]
example_shap_values = shap_values[example_position]

local_explanation = pd.DataFrame({
    "feature": feature_columns,
    "feature_value": example_features.values,
    "shap_value_log_scale": example_shap_values
}).sort_values("shap_value_log_scale", key=abs, ascending=False)

# Save waterfall plot for this one prediction.
shap_explanation = shap.Explanation(
    values=example_shap_values,
    base_values=explainer.expected_value,
    data=example_features.values,
    feature_names=feature_columns
)

plt.figure()
shap.plots.waterfall(shap_explanation, show=False, max_display=12)
plt.tight_layout()
plt.savefig("outputs/charts/shap/shap_waterfall_example.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 10. SAVE REPORT


output_file = "outputs/xlsx/09_shap_explainability_report.xlsx"

example_summary = pd.DataFrame({
    "field": [
        "Part Number",
        "description",
        "Month",
        "actual_net_qty",
        "predicted_qty",
        "prediction_log",
        "note"
    ],
    "value": [
        example_row["Part Number"],
        example_row["description"],
        example_row["Month"],
        example_row["net_qty"],
        example_row["prediction_qty"],
        example_row["prediction_log"],
        "SHAP values explain the log-transformed prediction because the model was trained on log1p(net_qty)."
    ]
})

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    test_results.to_excel(writer, sheet_name="predictions", index=False)
    shap_importance.to_excel(writer, sheet_name="global_shap_importance", index=False)
    local_explanation.to_excel(writer, sheet_name="local_example_explanation", index=False)
    example_summary.to_excel(writer, sheet_name="example_summary", index=False)

print(f"Created: {output_file}")
print("Charts saved in: outputs/charts/shap/")
print("Example explained:")
print(example_summary)