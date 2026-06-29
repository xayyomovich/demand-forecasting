import os
import pandas as pd
import numpy as np

# Purpose:
# Explore the combined 2023-2026 dataset before modelling.
#
# This script checks:
# - business summary
# - date summary
# - quantity summary
# - decimal Qty rows
# - top parts by quantity
# - top parts by frequency
# - monthly summary
# - forecastable parts using updated 37-month dataset
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
input_file = "results/xlsx/02_combined_2023_2024_2025_raw.csv"

output_folder = "results/xlsx"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "03_combined_data_discovery_report.xlsx")


# ------------------------------------------------------------
# 2. LOAD COMBINED DATA
# ------------------------------------------------------------
df = pd.read_csv(input_file)

print("Loaded combined data.")
print("Rows and columns:", df.shape)


# ------------------------------------------------------------
# 3. BASIC DATE PREPARATION
# ------------------------------------------------------------
# Date Converted was saved to CSV as text, so we convert it back
# into pandas datetime type.
df["Date Converted"] = pd.to_datetime(df["Date Converted"], errors="coerce")

# Month is text like 2023-01. If it is missing for any reason,
# recreate it from Date Converted.
if "Month" not in df.columns:
    df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)


# ------------------------------------------------------------
# 4. NUMERIC PREPARATION
# ------------------------------------------------------------
numeric_columns = [
    "Retail Val",
    "Sale val",
    "Disc val",
    "Disc %",
    "Cost val",
    "Profit",
    "PC",
    "Qty",
    "BR"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------------------
# 5. CREATE SOLD / RETURN QUANTITY COLUMNS
# ------------------------------------------------------------
# Positive Qty = sold quantity
# Negative Qty = returned quantity / credit note
# net_qty = sum of Qty
df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()


# ------------------------------------------------------------
# 6. BUSINESS SUMMARY
# ------------------------------------------------------------
business_summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "total_columns",
        "unique_parts",
        "unique_descriptions",
        "unique_franchises",
        "unique_branches",
        "unique_accounts",
        "total_net_qty",
        "total_sold_qty",
        "total_return_qty",
        "total_sale_value",
        "total_retail_value",
        "total_cost_value",
        "total_profit"
    ],
    "value": [
        len(df),
        len(df.columns),
        df["Part Number"].nunique(),
        df["Description"].nunique(),
        df["FR"].nunique(),
        df["BR"].nunique(),
        df["Account"].nunique(),
        df["Qty"].sum(),
        df["sold_qty"].sum(),
        df["return_qty"].sum(),
        df["Sale val"].sum(),
        df["Retail Val"].sum(),
        df["Cost val"].sum(),
        df["Profit"].sum()
    ]
})


# ------------------------------------------------------------
# 7. DATE SUMMARY
# ------------------------------------------------------------
date_summary = pd.DataFrame({
    "metric": [
        "minimum_date",
        "maximum_date",
        "total_days_range",
        "unique_months",
        "invalid_dates"
    ],
    "value": [
        df["Date Converted"].min(),
        df["Date Converted"].max(),
        (df["Date Converted"].max() - df["Date Converted"].min()).days,
        df["Month"].nunique(),
        df["Date Converted"].isna().sum()
    ]
})


# ------------------------------------------------------------
# 8. QUANTITY SUMMARY
# ------------------------------------------------------------
# Decimal Qty matters because 2023 contains values like 0.5.
# That can be valid for oils/fluids, but we need to document it.
decimal_qty_rows = df[
    df["Qty"].notna() & (df["Qty"] % 1 != 0)
].copy()

qty_summary = pd.DataFrame({
    "metric": [
        "total_net_qty",
        "total_sold_qty",
        "total_return_qty",
        "positive_qty_rows",
        "negative_qty_rows",
        "zero_qty_rows",
        "decimal_qty_rows",
        "minimum_qty",
        "maximum_qty",
        "average_qty",
        "median_qty"
    ],
    "value": [
        df["Qty"].sum(),
        df["sold_qty"].sum(),
        df["return_qty"].sum(),
        (df["Qty"] > 0).sum(),
        (df["Qty"] < 0).sum(),
        (df["Qty"] == 0).sum(),
        len(decimal_qty_rows),
        df["Qty"].min(),
        df["Qty"].max(),
        df["Qty"].mean(),
        df["Qty"].median()
    ]
})


# ------------------------------------------------------------
# 9. MONTHLY SUMMARY
# ------------------------------------------------------------
monthly_summary = (
    df.groupby("Month", as_index=False)
    .agg(
        total_net_qty=("Qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        total_sale_value=("Sale val", "sum"),
        total_profit=("Profit", "sum"),
        transaction_count=("Qty", "count"),
        unique_parts=("Part Number", "nunique")
    )
    .sort_values("Month")
)


# ------------------------------------------------------------
# 10. TOP PARTS BY NET QUANTITY
# ------------------------------------------------------------
top_parts_by_qty = (
    df.groupby("Part Number", as_index=False)
    .agg(
        total_net_qty=("Qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        transaction_count=("Qty", "count"),
        active_months=("Month", "nunique"),
        total_sale_value=("Sale val", "sum"),
        total_profit=("Profit", "sum"),
        description=("Description", "first"),
        fr=("FR", "first")
    )
    .sort_values("total_net_qty", ascending=False)
    .head(50)
)


# ------------------------------------------------------------
# 11. TOP PARTS BY TRANSACTION FREQUENCY
# ------------------------------------------------------------
top_parts_by_frequency = (
    df.groupby("Part Number", as_index=False)
    .agg(
        transaction_count=("Qty", "count"),
        total_net_qty=("Qty", "sum"),
        active_months=("Month", "nunique"),
        description=("Description", "first"),
        fr=("FR", "first")
    )
    .sort_values("transaction_count", ascending=False)
    .head(50)
)


# ------------------------------------------------------------
# 12. PART SUMMARY FOR FORECASTABLE PART SELECTION
# ------------------------------------------------------------
part_summary = (
    df.groupby("Part Number", as_index=False)
    .agg(
        active_months=("Month", "nunique"),
        total_net_qty=("Qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        transaction_count=("Qty", "count"),
        total_sale_value=("Sale val", "sum"),
        total_profit=("Profit", "sum"),
        description=("Description", "first"),
        fr=("FR", "first")
    )
)

# Old rule used 18 months because old data had only 25 months.
# With 37 months of data, we use 24 active months for stronger stability.
forecastable_parts_24_months = part_summary[
    (part_summary["active_months"] >= 24) &
    (part_summary["transaction_count"] >= 50) &
    (part_summary["total_sold_qty"] > 0)
].sort_values(
    ["active_months", "transaction_count", "total_sold_qty"],
    ascending=False
)

# We also calculate the old 18-month rule for comparison.
forecastable_parts_18_months = part_summary[
    (part_summary["active_months"] >= 18) &
    (part_summary["transaction_count"] >= 50) &
    (part_summary["total_sold_qty"] > 0)
].sort_values(
    ["active_months", "transaction_count", "total_sold_qty"],
    ascending=False
)

forecastable_summary = pd.DataFrame({
    "metric": [
        "total_unique_parts",
        "forecastable_parts_old_rule_18_months",
        "forecastable_parts_new_rule_24_months",
        "recommended_rule"
    ],
    "value": [
        part_summary["Part Number"].nunique(),
        len(forecastable_parts_18_months),
        len(forecastable_parts_24_months),
        "active_months >= 24, transaction_count >= 50, total_sold_qty > 0"
    ]
})


# ------------------------------------------------------------
# 13. DECIMAL QTY INSPECTION
# ------------------------------------------------------------
decimal_qty_sample = decimal_qty_rows[
    [
        "source_file",
        "Month",
        "Date Converted",
        "FR",
        "Part Number",
        "Description",
        "Invoice",
        "Qty",
        "Sale val",
        "Cost val",
        "Profit",
        "BR"
    ]
].head(100)


# ------------------------------------------------------------
# 14. SAVE EXCEL REPORT
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    business_summary.to_excel(writer, sheet_name="business_summary", index=False)
    date_summary.to_excel(writer, sheet_name="date_summary", index=False)
    qty_summary.to_excel(writer, sheet_name="qty_summary", index=False)
    monthly_summary.to_excel(writer, sheet_name="monthly_summary", index=False)
    top_parts_by_qty.to_excel(writer, sheet_name="top_parts_by_qty", index=False)
    top_parts_by_frequency.to_excel(writer, sheet_name="top_parts_by_frequency", index=False)
    part_summary.to_excel(writer, sheet_name="part_summary", index=False)
    forecastable_summary.to_excel(writer, sheet_name="forecastable_summary", index=False)
    forecastable_parts_24_months.to_excel(writer, sheet_name="forecastable_24_months", index=False)
    forecastable_parts_18_months.to_excel(writer, sheet_name="forecastable_18_months", index=False)
    decimal_qty_sample.to_excel(writer, sheet_name="decimal_qty_sample", index=False)


# ------------------------------------------------------------
# 15. PRINT IMPORTANT RESULTS
# ------------------------------------------------------------
print("\nCreated:", output_file)

print("\nBusiness summary:")
print(business_summary.to_string(index=False))

print("\nDate summary:")
print(date_summary.to_string(index=False))

print("\nQuantity summary:")
print(qty_summary.to_string(index=False))

print("\nForecastable summary:")
print(forecastable_summary.to_string(index=False))

print("\nRecommended next step:")
print("Open the Excel report and check forecastable_summary + decimal_qty_sample.")







# D:\Diploma_work\diploma\diploma_v2\03_combined_data_discovery.py:34: DtypeWarning: Columns (0: MLI, 1: DC, 2: source_file) have mixed types. Specify dtype option on import or set low_memory=False.
#   df = pd.read_csv(input_file)
# Loaded combined data.
# Rows and columns: (376633, 34)

# Created: results/xlsx\03_combined_data_discovery_report.xlsx

# Business summary:
#              metric        value
#          total_rows 3.766330e+05
#       total_columns 3.600000e+01
#        unique_parts 3.516100e+04
# unique_descriptions 6.684500e+04
#   unique_franchises 1.200000e+01
#     unique_branches 5.000000e+00
#     unique_accounts 3.000000e+02
#       total_net_qty 2.245330e+06
#      total_sold_qty 2.284218e+06
#    total_return_qty 3.888800e+04
#    total_sale_value 2.157012e+08
#  total_retail_value 4.677568e+08
#    total_cost_value 1.678341e+08
#        total_profit 4.786847e+07

# Date summary:
#           metric               value
#     minimum_date 2023-01-02 00:00:00
#     maximum_date 2026-01-31 00:00:00
# total_days_range                1125
#    unique_months                  37
#    invalid_dates                   0

# Quantity summary:
#            metric         value
#     total_net_qty  2.245330e+06
#    total_sold_qty  2.284218e+06
#  total_return_qty  3.888800e+04
# positive_qty_rows  3.666320e+05
# negative_qty_rows  9.168000e+03
#     zero_qty_rows  8.330000e+02
#  decimal_qty_rows  1.631000e+03
#       minimum_qty -1.440000e+03
#       maximum_qty  5.760000e+03
#       average_qty  5.961586e+00
#        median_qty  1.000000e+00

# Forecastable summary:
#                                metric                                                            value
#                    total_unique_parts                                                            35161
# forecastable_parts_old_rule_18_months                                                             1087
# forecastable_parts_new_rule_24_months                                                             1001
#                      recommended_rule active_months >= 24, transaction_count >= 50, total_sold_qty > 0

# Recommended next step:
# Open the Excel report and check forecastable_summary + decimal_qty_sample.