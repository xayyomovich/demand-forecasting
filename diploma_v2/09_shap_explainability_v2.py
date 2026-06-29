import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ============================================================
# 09_shap_explainability_v2.py
#
# Purpose:
# Explain the saved XGBoost v2 model using SHAP.
#
# Important:
# SHAP explains the XGBoost model, not the lag-1 production model.
#
# The XGBoost model predicts log1p(net_qty), so SHAP values are
# on the log-transformed scale.
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
data_file = "results/xlsx/06b_feature_engineered_v4_no_lag24.csv"
model_file = "models/xgboost_log_target_v2.pkl"
feature_file = "models/xgboost_log_target_v2_features.json"
xgboost_report_file = "results/xlsx/08_xgboost_log_target_v2_report.xlsx"

output_folder = "results/xlsx"
chart_folder = "results/charts/shap_v2"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(chart_folder, exist_ok=True)

output_file = os.path.join(output_folder, "09_shap_explainability_v2_report.xlsx")


# ------------------------------------------------------------
# 2. LOAD MODEL, FEATURES, DATA
# ------------------------------------------------------------
model = joblib.load(model_file)

with open(feature_file, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

df = pd.read_csv(data_file, low_memory=False)

test_df = df[df["data_split"] == "test"].copy()

print("Loaded model and test data.")
print("Test rows:", len(test_df))
print("Feature count:", len(feature_columns))


# ------------------------------------------------------------
# 3. PREPARE SHAP SAMPLE
# ------------------------------------------------------------
# SHAP can be slow on all 5005 rows, so we use a sample.
# This is enough for global explanation.
shap_sample_size = min(1000, len(test_df))

shap_sample = test_df.sample(
    n=shap_sample_size,
    random_state=42
).copy()

X_shap = shap_sample[feature_columns]


# ------------------------------------------------------------
# 4. CREATE SHAP VALUES
# ------------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_shap)

print("SHAP values created.")


# ------------------------------------------------------------
# 5. GLOBAL SHAP IMPORTANCE TABLE
# ------------------------------------------------------------
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)


# ------------------------------------------------------------
# 6. SHAP SUMMARY PLOT
# ------------------------------------------------------------
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_shap,
    show=False,
    max_display=20
)
summary_plot_path = os.path.join(chart_folder, "shap_v2_summary_plot.png")
plt.tight_layout()
plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 7. GLOBAL SHAP BAR PLOT
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))
top_shap = shap_importance.head(20).sort_values("mean_abs_shap")

plt.barh(top_shap["feature"], top_shap["mean_abs_shap"])
plt.title("SHAP V2 Global Feature Importance")
plt.xlabel("mean(|SHAP value|) on log prediction scale")
plt.ylabel("Feature")
plt.tight_layout()

global_bar_path = os.path.join(chart_folder, "shap_v2_global_feature_importance.png")
plt.savefig(global_bar_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 8. LOCAL WATERFALL EXPLANATION
# ------------------------------------------------------------
# Choose one high-demand test row to explain.
# This makes the waterfall easier to understand.
example_index = (
    test_df["net_qty"]
    .clip(lower=0)
    .sort_values(ascending=False)
    .index[0]
)

example_row = test_df.loc[[example_index]].copy()
X_example = example_row[feature_columns]

example_shap_values = explainer(X_example)

plt.figure()
shap.plots.waterfall(
    example_shap_values[0],
    max_display=15,
    show=False
)

waterfall_path = os.path.join(chart_folder, "shap_v2_waterfall_example.png")
plt.tight_layout()
plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# 9. EXAMPLE EXPLANATION TABLE
# ------------------------------------------------------------
example_prediction_log = model.predict(X_example)[0]
example_prediction_qty = np.expm1(example_prediction_log)
example_prediction_qty = max(0, example_prediction_qty)

example_info = pd.DataFrame({
    "field": [
        "Part Number",
        "description",
        "Month",
        "actual_net_qty",
        "actual_target_qty_clipped",
        "predicted_qty",
        "prediction_log",
        "note"
    ],
    "value": [
        example_row["Part Number"].iloc[0],
        example_row["description"].iloc[0],
        example_row["Month"].iloc[0],
        example_row["net_qty"].iloc[0],
        max(0, example_row["net_qty"].iloc[0]),
        example_prediction_qty,
        example_prediction_log,
        "SHAP values explain the XGBoost log1p(net_qty) prediction, not the lag-1 production forecast."
    ]
})

example_feature_effects = pd.DataFrame({
    "feature": feature_columns,
    "feature_value": X_example.iloc[0].values,
    "shap_value_log_scale": example_shap_values.values[0]
}).sort_values("shap_value_log_scale", ascending=False)


# ------------------------------------------------------------
# 10. SAVE SHAP SAMPLE WITH PREDICTIONS
# ------------------------------------------------------------
shap_sample_predictions_log = model.predict(X_shap)
shap_sample_predictions_qty = np.expm1(shap_sample_predictions_log)
shap_sample_predictions_qty = np.clip(shap_sample_predictions_qty, 0, None)

shap_sample_output = shap_sample[
    [
        "Part Number",
        "Month",
        "description",
        "fr",
        "net_qty",
        "data_split"
    ]
].copy()

shap_sample_output["predicted_qty"] = shap_sample_predictions_qty
shap_sample_output["prediction_log"] = shap_sample_predictions_log


# ------------------------------------------------------------
# 11. SAVE EXCEL REPORT
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    shap_importance.to_excel(writer, sheet_name="shap_global_importance", index=False)
    example_info.to_excel(writer, sheet_name="example_info", index=False)
    example_feature_effects.to_excel(writer, sheet_name="example_feature_effects", index=False)
    shap_sample_output.to_excel(writer, sheet_name="shap_sample_predictions", index=False)
    pd.DataFrame({"feature": feature_columns}).to_excel(writer, sheet_name="feature_columns", index=False)


# ------------------------------------------------------------
# 12. PRINT RESULTS
# ------------------------------------------------------------
print("\nCreated:", output_file)
print("Charts saved in:", chart_folder)

print("\nTop 15 SHAP features:")
print(shap_importance.head(15).to_string(index=False))

print("\nExample explained:")
print(example_info.to_string(index=False))



# Loaded model and test data.
# Test rows: 5005
# Feature count: 38
# SHAP values created.

# Created: results/xlsx\09_shap_explainability_v2_report.xlsx
# Charts saved in: results/charts/shap_v2

# Top 15 SHAP features:
#                     feature  mean_abs_shap
#       part_train_median_qty       0.350651
#          part_train_avg_qty       0.279986
#     transaction_count_lag_1       0.098211
#             rolling_mean_12       0.080289
#              rolling_mean_3       0.080270
#    part_train_active_months       0.072783
#        part_train_total_qty       0.062087
#                   month_sin       0.059250
#            rolling_median_6       0.057232
#        sales_months_last_12       0.049934
#                   month_cos       0.043212
#                      lag_13       0.039858
# part_train_avg_transactions       0.036044
#              rolling_mean_6       0.032476
#                month_number       0.031943

# Example explained:
#                     field                                                                                         value
#               Part Number                                                                       MBA000 989 33 09/11ABDW
#               description                                                                                    ENGINE OIL
#                     Month                                                                                       2025-09
#            actual_net_qty                                                                                       14396.0
# actual_target_qty_clipped                                                                                       14396.0
#             predicted_qty                                                                                   1481.062378
#            prediction_log                                                                                       7.30119
#                      note SHAP values explain the XGBoost log1p(net_qty) prediction, not the lag-1 production forecast.