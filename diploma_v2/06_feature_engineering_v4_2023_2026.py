import os
import numpy as np
import pandas as pd

# ============================================================
# 06_feature_engineering_v4_2023_2026.py
#
# Purpose:
# Create modelling dataset from the new 2023-2026 monthly demand.
#
# Key idea:
# We only use information that would be available before the
# forecast month. This avoids target leakage.
# ============================================================

input_monthly = "results/xlsx/04_monthly_forecastable_demand_v2.csv"
input_parts = "results/xlsx/04_forecastable_parts_v2.csv"

output_folder = "results/xlsx"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "06_feature_engineering_v4_2023_2026_report.xlsx")
output_csv = os.path.join(output_folder, "06_feature_engineered_v4_2023_2026.csv")


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
monthly = pd.read_csv(input_monthly, low_memory=False)
parts = pd.read_csv(input_parts, low_memory=False)

print("Loaded monthly forecastable demand.")
print("Monthly rows:", len(monthly))
print("Forecastable parts:", parts["Part Number"].nunique())


# ------------------------------------------------------------
# 2. PREPARE MONTH DATE
# ------------------------------------------------------------
monthly["Month Date"] = pd.to_datetime(monthly["Month"] + "-01")
monthly = monthly.sort_values(["Part Number", "Month Date"])


# ------------------------------------------------------------
# 3. CREATE COMPLETE MONTH GRID
# ------------------------------------------------------------
# Some parts may not appear in some months.
# Missing month means demand was 0, so we create full grid:
# every forecastable part x every month.
all_parts = parts["Part Number"].unique()
all_months = pd.date_range(
    monthly["Month Date"].min(),
    monthly["Month Date"].max(),
    freq="MS"
)

grid = pd.MultiIndex.from_product(
    [all_parts, all_months],
    names=["Part Number", "Month Date"]
).to_frame(index=False)

model_df = grid.merge(
    monthly,
    on=["Part Number", "Month Date"],
    how="left"
)

model_df["Month"] = model_df["Month Date"].dt.to_period("M").astype(str)

# Fill missing demand values with 0.
demand_cols = [
    "net_qty",
    "sold_qty",
    "return_qty",
    "transaction_count",
    "sale_value",
    "retail_value",
    "cost_value",
    "profit"
]

for col in demand_cols:
    model_df[col] = model_df[col].fillna(0)

# Fill descriptive columns from part summary.
part_info = parts[["Part Number", "description", "fr"]].drop_duplicates("Part Number")
model_df = model_df.drop(columns=["description", "fr"], errors="ignore")
model_df = model_df.merge(part_info, on="Part Number", how="left")


# ------------------------------------------------------------
# 4. CALENDAR FEATURES
# ------------------------------------------------------------
model_df["year"] = model_df["Month Date"].dt.year
model_df["month_number"] = model_df["Month Date"].dt.month
model_df["quarter"] = model_df["Month Date"].dt.quarter

# Cyclical encoding:
# Month 12 and month 1 are close in real life, but normal numbers
# make them look far apart. sin/cos fixes that.
model_df["month_sin"] = np.sin(2 * np.pi * model_df["month_number"] / 12)
model_df["month_cos"] = np.cos(2 * np.pi * model_df["month_number"] / 12)


# ------------------------------------------------------------
# 5. LAG FEATURES
# ------------------------------------------------------------
# lag_1 = previous month demand
# lag_12 = same month last year
# lag_24 = same month two years ago
group = model_df.groupby("Part Number")

for lag in [1, 2, 3, 6, 12, 13, 24]:
    model_df[f"lag_{lag}"] = group["net_qty"].shift(lag)

model_df["sold_qty_lag_1"] = group["sold_qty"].shift(1)
model_df["return_qty_lag_1"] = group["return_qty"].shift(1)
model_df["transaction_count_lag_1"] = group["transaction_count"].shift(1)


# ------------------------------------------------------------
# 6. ROLLING FEATURES
# ------------------------------------------------------------
# shift(1) is important:
# it prevents current month demand from leaking into current month features.
shifted_net_qty = group["net_qty"].shift(1)

model_df["rolling_mean_3"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

model_df["rolling_mean_6"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(6)
    .mean()
    .reset_index(level=0, drop=True)
)

model_df["rolling_mean_12"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(12)
    .mean()
    .reset_index(level=0, drop=True)
)

model_df["rolling_median_3"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(3)
    .median()
    .reset_index(level=0, drop=True)
)

model_df["rolling_median_6"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(6)
    .median()
    .reset_index(level=0, drop=True)
)

model_df["rolling_std_3"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(3)
    .std()
    .reset_index(level=0, drop=True)
)

model_df["rolling_std_6"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(6)
    .std()
    .reset_index(level=0, drop=True)
)

model_df["rolling_min_6"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(6)
    .min()
    .reset_index(level=0, drop=True)
)

model_df["rolling_max_6"] = (
    shifted_net_qty.groupby(model_df["Part Number"])
    .rolling(6)
    .max()
    .reset_index(level=0, drop=True)
)


# ------------------------------------------------------------
# 7. CHANGE / MOMENTUM FEATURES
# ------------------------------------------------------------
model_df["diff_1"] = model_df["lag_1"] - model_df["lag_2"]
model_df["diff_3"] = model_df["lag_1"] - model_df["lag_3"]

# +1 avoids division by zero.
model_df["pct_change_1"] = (model_df["lag_1"] - model_df["lag_2"]) / (model_df["lag_2"].abs() + 1)


# ------------------------------------------------------------
# 8. RETURN FEATURES
# ------------------------------------------------------------
model_df["return_ratio_lag_1"] = model_df["return_qty_lag_1"] / (model_df["sold_qty_lag_1"] + 1)


# ------------------------------------------------------------
# 9. INTERMITTENT DEMAND FEATURES
# ------------------------------------------------------------
# These are useful for spare parts because many parts do not sell every month.
#
# We avoid groupby().apply() here because it can remove "Part Number"
# from normal columns depending on pandas version.

model_df = model_df.sort_values(["Part Number", "Month Date"]).reset_index(drop=True)

# Previous month sale flag:
# 1 = previous month had positive demand
# 0 = previous month had zero/negative demand
model_df["previous_sale_flag"] = (
    model_df.groupby("Part Number")["net_qty"]
    .shift(1)
    .gt(0)
    .astype(float)
)

# How many of the previous 6 months had sales?
model_df["sales_months_last_6"] = (
    model_df.groupby("Part Number")["previous_sale_flag"]
    .transform(lambda x: x.rolling(6).sum())
)

# How many of the previous 12 months had sales?
model_df["sales_months_last_12"] = (
    model_df.groupby("Part Number")["previous_sale_flag"]
    .transform(lambda x: x.rolling(12).sum())
)


def months_since_last_sale(previous_qty_series):
    result = []
    last_sale_position = None

    for position, qty in enumerate(previous_qty_series):
        if pd.notna(qty) and qty > 0:
            last_sale_position = position

        if last_sale_position is None:
            result.append(np.nan)
        else:
            result.append(position - last_sale_position)

    return pd.Series(result, index=previous_qty_series.index)


def zero_demand_streak(previous_qty_series):
    result = []
    streak = 0

    for qty in previous_qty_series:
        if pd.isna(qty):
            result.append(np.nan)
        elif qty <= 0:
            streak += 1
            result.append(streak)
        else:
            streak = 0
            result.append(streak)

    return pd.Series(result, index=previous_qty_series.index)


# Previous demand, used to calculate sale gaps safely.
model_df["previous_net_qty"] = (
    model_df.groupby("Part Number")["net_qty"]
    .shift(1)
)

model_df["months_since_last_sale"] = (
    model_df.groupby("Part Number")["previous_net_qty"]
    .transform(months_since_last_sale)
)

model_df["zero_demand_streak"] = (
    model_df.groupby("Part Number")["previous_net_qty"]
    .transform(zero_demand_streak)
)

# We do not need helper columns in the final model.
model_df = model_df.drop(columns=["previous_sale_flag", "previous_net_qty"])

print("Columns after intermittent features:")
print(model_df.columns.tolist())

if "Part Number" not in model_df.columns:
    raise ValueError("Part Number column disappeared after intermittent feature creation.")


# ------------------------------------------------------------
# 10. TIME SPLIT
# ------------------------------------------------------------
# We keep the same final test months for fair comparison.
test_months = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01"]
validation_months = ["2025-06", "2025-07", "2025-08"]

model_df["data_split"] = "train"
model_df.loc[model_df["Month"].isin(validation_months), "data_split"] = "validation"
model_df.loc[model_df["Month"].isin(test_months), "data_split"] = "test"


# ------------------------------------------------------------
# 11. SAFE PART-LEVEL FEATURES
# ------------------------------------------------------------
# These features must be calculated from TRAIN MONTHS ONLY.
# If we use all months, we leak future information into the model.

train_for_part_features = model_df[model_df["data_split"] == "train"].copy()

safe_part_features = (
    train_for_part_features.groupby("Part Number", as_index=False)
    .agg(
        part_train_avg_qty=("net_qty", "mean"),
        part_train_median_qty=("net_qty", "median"),
        part_train_std_qty=("net_qty", "std"),
        part_train_total_qty=("net_qty", "sum"),
        part_train_max_qty=("net_qty", "max"),
        part_train_active_months=("net_qty", lambda x: (x > 0).sum()),
        part_train_avg_transactions=("transaction_count", "mean")
    )
)

model_df = model_df.merge(safe_part_features, on="Part Number", how="left")


# ------------------------------------------------------------
# 12. FINAL CLEANING
# ------------------------------------------------------------
# Rows with lag_24 missing cannot be used by models requiring lag_24.
# Since we now have 37 months, lag_24 is possible but removes early rows.
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
    "lag_24",

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

model_rows_before_dropna = len(model_df)

model_ready = model_df.dropna(subset=feature_columns + ["net_qty"]).copy()

model_rows_after_dropna = len(model_ready)


# ------------------------------------------------------------
# 13. SUMMARY
# ------------------------------------------------------------
summary = pd.DataFrame({
    "metric": [
        "forecastable_parts",
        "all_months",
        "model_rows_before_dropna",
        "model_rows_after_dropna",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_months_min",
        "train_months_max",
        "validation_months",
        "test_months",
        "feature_count",
        "note"
    ],
    "value": [
        model_df["Part Number"].nunique(),
        model_df["Month"].nunique(),
        model_rows_before_dropna,
        model_rows_after_dropna,
        (model_ready["data_split"] == "train").sum(),
        (model_ready["data_split"] == "validation").sum(),
        (model_ready["data_split"] == "test").sum(),
        model_ready.loc[model_ready["data_split"] == "train", "Month"].min(),
        model_ready.loc[model_ready["data_split"] == "train", "Month"].max(),
        ", ".join(validation_months),
        ", ".join(test_months),
        len(feature_columns),
        "Uses 2023-2026 data, 1001 parts, lag_24, seasonal, rolling, intermittent, and safe train-only part features."
    ]
})


# ------------------------------------------------------------
# 14. SAVE OUTPUTS
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    model_ready.head(5000).to_excel(writer, sheet_name="model_ready_sample", index=False)
    safe_part_features.to_excel(writer, sheet_name="safe_part_features", index=False)
    pd.DataFrame({"feature": feature_columns}).to_excel(writer, sheet_name="feature_columns", index=False)

model_ready.to_csv(output_csv, index=False)


# ------------------------------------------------------------
# 15. PRINT RESULTS
# ------------------------------------------------------------
print("\nCreated:", output_file)
print("Created:", output_csv)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nFeature columns:")
for feature in feature_columns:
    print("-", feature)








# Loaded monthly forecastable demand.
# Monthly rows: 31759
# Forecastable parts: 1001
# Columns after intermittent features:
# ['Part Number', 'Month Date', 'Month', 'net_qty', 'sold_qty', 'return_qty', 'transaction_count', 'sale_value', 'retail_value', 'cost_value', 'profit', 'branch', 'description', 'fr', 'year', 'month_number', 'quarter', 'month_sin', 'month_cos', 'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_13', 'lag_24', 'sold_qty_lag_1', 'return_qty_lag_1', 'transaction_count_lag_1', 'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12', 'rolling_median_3', 'rolling_median_6', 'rolling_std_3', 'rolling_std_6', 'rolling_min_6', 'rolling_max_6', 'diff_1', 'diff_3', 'pct_change_1', 'return_ratio_lag_1', 'sales_months_last_6', 'sales_months_last_12', 'months_since_last_sale', 'zero_demand_streak']

# Created: results/xlsx\06_feature_engineering_v4_2023_2026_report.xlsx
# Created: results/xlsx\06_feature_engineered_v4_2023_2026.csv

# Summary:
#                   metric                                                                                                        value
#       forecastable_parts                                                                                                         1001
#               all_months                                                                                                           37
# model_rows_before_dropna                                                                                                        37037
#  model_rows_after_dropna                                                                                                        13013
#               train_rows                                                                                                         5005
#          validation_rows                                                                                                         3003
#                test_rows                                                                                                         5005
#         train_months_min                                                                                                      2025-01
#         train_months_max                                                                                                      2025-05
#        validation_months                                                                                    2025-06, 2025-07, 2025-08
#              test_months                                                                  2025-09, 2025-10, 2025-11, 2025-12, 2026-01
#            feature_count                                                                                                           39
#                     note Uses 2023-2026 data, 1001 parts, lag_24, seasonal, rolling, intermittent, and safe train-only part features.

# Feature columns:
# - year
# - month_number
# - quarter
# - month_sin
# - month_cos
# - lag_1
# - lag_2
# - lag_3
# - lag_6
# - lag_12
# - lag_13
# - lag_24
# - sold_qty_lag_1
# - return_qty_lag_1
# - transaction_count_lag_1
# - return_ratio_lag_1
# - rolling_mean_3
# - rolling_mean_6
# - rolling_mean_12
# - rolling_median_3
# - rolling_median_6
# - rolling_std_3
# - rolling_std_6
# - rolling_min_6
# - rolling_max_6
# - diff_1
# - diff_3
# - pct_change_1
# - sales_months_last_6
# - sales_months_last_12
# - months_since_last_sale
# - zero_demand_streak
# - part_train_avg_qty
# - part_train_median_qty
# - part_train_std_qty
# - part_train_total_qty
# - part_train_max_qty
# - part_train_active_months
# - part_train_avg_transactions