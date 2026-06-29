import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 10_final_model_selection_v2.py
#
# Purpose:
# Create final model selection report for the upgraded project.
#
# Final decision:
# Select the model with lowest MAE on final test months.
#
# Current evidence:
# pred_lag_1 has the best MAE, so it is the production method.
# XGBoost remains as explainable ML model with SHAP.
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
baseline_report = "results/xlsx/07_baseline_forecasting_v2_report.xlsx"
xgboost_report = "results/xlsx/08_xgboost_log_target_v2_report.xlsx"

output_folder = "results/xlsx"
chart_folder = "results/charts/final_model_selection_v2"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(chart_folder, exist_ok=True)

output_file = os.path.join(output_folder, "10_final_model_selection_v2_report.xlsx")


# ------------------------------------------------------------
# 2. LOAD METRICS
# ------------------------------------------------------------
baseline_metrics = pd.read_excel(baseline_report, sheet_name="baseline_metrics")
xgboost_metrics = pd.read_excel(xgboost_report, sheet_name="xgboost_metrics")

all_metrics = pd.concat(
    [baseline_metrics, xgboost_metrics],
    ignore_index=True
).sort_values("MAE")

best_model = all_metrics.iloc[0]["model"]
best_mae = all_metrics.iloc[0]["MAE"]


# ------------------------------------------------------------
# 3. FINAL DECISION TABLE
# ------------------------------------------------------------
if best_model == "pred_lag_1":
    production_decision = (
        "pred_lag_1 is selected as the final production forecasting method "
        "because it achieved the lowest MAE on the final time-based test set."
    )
else:
    production_decision = (
        f"{best_model} is selected as the final production forecasting method "
        "because it achieved the lowest MAE on the final time-based test set."
    )

final_decision = pd.DataFrame({
    "field": [
        "final_production_model",
        "final_production_model_MAE",
        "selection_metric",
        "test_months",
        "production_decision",
        "explainable_ml_model",
        "xgboost_role",
        "important_warning"
    ],
    "value": [
        best_model,
        best_mae,
        "MAE",
        "2025-09, 2025-10, 2025-11, 2025-12, 2026-01",
        production_decision,
        "pred_xgboost_log_v2",
        "XGBoost is retained as an explainable machine learning model using SHAP, but it is not the best production model by MAE.",
        "Forecasting performance should be monitored when new months are added. The selected model is based on current historical evidence, not assumed to be optimal forever."
    ]
})


# ------------------------------------------------------------
# 4. MODEL ROLE TABLE
# ------------------------------------------------------------
model_roles = pd.DataFrame({
    "model": [
        "pred_lag_1",
        "pred_rolling_mean_3",
        "pred_rolling_mean_6",
        "pred_part_train_avg",
        "pred_part_train_median",
        "pred_xgboost_log_v2"
    ],
    "role": [
        "Final production forecasting method",
        "Simple benchmark / smoothing method",
        "Simple benchmark / longer smoothing method",
        "Historical-average benchmark",
        "Historical-median benchmark",
        "Explainable ML model with SHAP"
    ],
    "interpretation": [
        "Predicts next month using previous month demand.",
        "Predicts next month using average demand from previous 3 months.",
        "Predicts next month using average demand from previous 6 months.",
        "Predicts using the part's average demand from training data.",
        "Predicts using the part's median demand from training data.",
        "Learns nonlinear patterns from lag, rolling, seasonal, intermittent, and part-level features."
    ]
})


# ------------------------------------------------------------
# 5. SUMMARY FOR REPORT
# ------------------------------------------------------------
report_summary = pd.DataFrame({
    "statement": [
        "The upgraded dataset contains 37 months of historical data from January 2023 to January 2026.",
        "The number of forecastable parts increased to 1,001 using the rule active_months >= 24, transaction_count >= 50, and total_sold_qty > 0.",
        "The final model-ready dataset contains 24,024 rows after feature engineering.",
        "The final test period covers five unseen future months from September 2025 to January 2026.",
        "The lag-1 baseline achieved the best MAE and is selected as the production forecasting method.",
        "XGBoost v2 did not beat lag-1 by MAE, but SHAP explanations showed that it used meaningful signals such as historical part demand, rolling means, transaction activity, and seasonality.",
        "The final system should present lag-1 as the production forecast and XGBoost + SHAP as the explainable ML analysis component."
    ]
})


# ------------------------------------------------------------
# 6. CHART: MAE COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(all_metrics["model"], all_metrics["MAE"])
plt.title("Final Model Selection V2 - MAE Comparison")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

mae_chart_path = os.path.join(chart_folder, "final_model_selection_v2_mae_comparison.png")
plt.savefig(mae_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 7. CHART: RMSE COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(all_metrics["model"], all_metrics["RMSE"])
plt.title("Final Model Selection V2 - RMSE Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

rmse_chart_path = os.path.join(chart_folder, "final_model_selection_v2_rmse_comparison.png")
plt.savefig(rmse_chart_path, dpi=200)
plt.close()


# ------------------------------------------------------------
# 8. SAVE REPORT
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    all_metrics.to_excel(writer, sheet_name="all_model_metrics", index=False)
    final_decision.to_excel(writer, sheet_name="final_decision", index=False)
    model_roles.to_excel(writer, sheet_name="model_roles", index=False)
    report_summary.to_excel(writer, sheet_name="report_summary", index=False)


# ------------------------------------------------------------
# 9. PRINT RESULTS
# ------------------------------------------------------------
print("\nCreated:", output_file)
print("Charts saved in:", chart_folder)

print("\nAll model metrics:")
print(all_metrics.to_string(index=False))

print("\nFinal decision:")
print(final_decision.to_string(index=False))



# Created: results/xlsx\10_final_model_selection_v2_report.xlsx
# Charts saved in: results/charts/final_model_selection_v2

# All model metrics:
#                  model       MAE       RMSE     sMAPE
#             pred_lag_1 22.828172 222.045448 96.832225
#    pred_xgboost_log_v2 25.524848 353.393542 94.054433
#    pred_rolling_mean_3 27.056210 341.325523 90.602719
# pred_part_train_median 36.717782 487.274604 96.294472
#    pred_rolling_mean_6 36.992607 510.593792 90.576537
#    pred_part_train_avg 40.687995 541.603275 95.977249

# Final decision:
#                      field                                                                                                                                       
#                           value
#     final_production_model                                                                                                                                       
#                      pred_lag_1
# final_production_model_MAE                                                                                                                                       
#                       22.828172
#           selection_metric                                                                                                                                       
#                             MAE
#                test_months                                                                                                                           2025-09, 2025-10, 2025-11, 2025-12, 2026-01
#        production_decision                                pred_lag_1 is selected as the final production forecasting method because it achieved the lowest MAE on the final time-based test set.
#       explainable_ml_model                                                                                                                                       
#             pred_xgboost_log_v2
#               xgboost_role                                              XGBoost is retained as an explainable machine learning model using SHAP, but it is not the best production model by MAE.
#          important_warning Forecasting performance should be monitored when new months are added. The selected model is based on current historical evidence, not assumed to be optimal forever.