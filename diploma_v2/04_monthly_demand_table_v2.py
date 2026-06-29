import os
import pandas as pd

# ============================================================
# 04_monthly_demand_table_v2.py
#
# Purpose:
# Build the monthly demand table from the combined 2023-2026
# dataset.
#
# This is the modelling foundation:
# raw transactions -> monthly demand per part
#
# Forecastable rule:
# active_months >= 24
# transaction_count >= 50
# total_sold_qty > 0
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
input_file = "results/xlsx/02_combined_2023_2024_2025_raw.csv"

output_folder = "results/xlsx"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "04_monthly_demand_table_v2_report.xlsx")


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(input_file, low_memory=False)

print("Loaded combined raw data.")
print("Rows and columns:", df.shape)


# ------------------------------------------------------------
# 3. PREPARE DATE AND NUMERIC COLUMNS
# ------------------------------------------------------------
df["Date Converted"] = pd.to_datetime(df["Date Converted"], errors="coerce")

if "Month" not in df.columns:
    df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)

numeric_columns = [
    "Qty",
    "Sale val",
    "Retail Val",
    "Cost val",
    "Profit"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------------------
# 4. CREATE SOLD / RETURN QUANTITY
# ------------------------------------------------------------
# Qty can be positive, negative, zero, or decimal.
#
# Positive Qty = sold demand
# Negative Qty = return / credit note
# Decimal Qty = allowed because fluids/oils may be sold partially
df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()


# ------------------------------------------------------------
# 5. CREATE MONTHLY PART DEMAND TABLE
# ------------------------------------------------------------
monthly_part_demand = (
    df.groupby(["Month", "Part Number"], as_index=False)
    .agg(
        net_qty=("Qty", "sum"),
        sold_qty=("sold_qty", "sum"),
        return_qty=("return_qty", "sum"),
        transaction_count=("Qty", "count"),
        sale_value=("Sale val", "sum"),
        retail_value=("Retail Val", "sum"),
        cost_value=("Cost val", "sum"),
        profit=("Profit", "sum"),
        description=("Description", "first"),
        fr=("FR", "first"),
        branch=("BR", "first")
    )
    .sort_values(["Part Number", "Month"])
)


# ------------------------------------------------------------
# 6. CREATE PART SUMMARY
# ------------------------------------------------------------
part_summary = (
    monthly_part_demand.groupby("Part Number", as_index=False)
    .agg(
        active_months=("Month", "nunique"),
        total_net_qty=("net_qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        total_transactions=("transaction_count", "sum"),
        avg_monthly_net_qty=("net_qty", "mean"),
        median_monthly_net_qty=("net_qty", "median"),
        max_monthly_net_qty=("net_qty", "max"),
        total_sale_value=("sale_value", "sum"),
        total_profit=("profit", "sum"),
        description=("description", "first"),
        fr=("fr", "first")
    )
)


# ------------------------------------------------------------
# 7. SELECT FORECASTABLE PARTS
# ------------------------------------------------------------
forecastable_parts = part_summary[
    (part_summary["active_months"] >= 24) &
    (part_summary["total_transactions"] >= 50) &
    (part_summary["total_sold_qty"] > 0)
].sort_values(
    ["active_months", "total_transactions", "total_sold_qty"],
    ascending=False
)

top_50_candidates = forecastable_parts.head(50)


# ------------------------------------------------------------
# 8. FILTER MONTHLY TABLE TO FORECASTABLE PARTS
# ------------------------------------------------------------
forecastable_part_numbers = forecastable_parts["Part Number"].unique()

monthly_forecastable_demand = monthly_part_demand[
    monthly_part_demand["Part Number"].isin(forecastable_part_numbers)
].copy()


# ------------------------------------------------------------
# 9. OUTLIER INSPECTION TABLES
# ------------------------------------------------------------
# These are not removed automatically. We save them for review.
transaction_outliers = df[
    (df["Qty"] <= -100) | (df["Qty"] >= 100)
].sort_values("Qty", ascending=False)

monthly_demand_99 = monthly_part_demand["net_qty"].quantile(0.99)
monthly_demand_999 = monthly_part_demand["net_qty"].quantile(0.999)

monthly_demand_outliers = monthly_part_demand[
    monthly_part_demand["net_qty"] >= monthly_demand_99
].sort_values("net_qty", ascending=False)


# ------------------------------------------------------------
# 10. SUMMARY
# ------------------------------------------------------------
summary = pd.DataFrame({
    "metric": [
        "raw_transaction_rows",
        "raw_unique_parts",
        "monthly_part_demand_rows",
        "forecastable_parts",
        "monthly_forecastable_demand_rows",
        "minimum_month",
        "maximum_month",
        "unique_months",
        "forecastable_rule",
        "monthly_net_qty_99th_percentile",
        "monthly_net_qty_99_9th_percentile",
        "transaction_outlier_rows_abs_qty_100"
    ],
    "value": [
        len(df),
        df["Part Number"].nunique(),
        len(monthly_part_demand),
        len(forecastable_parts),
        len(monthly_forecastable_demand),
        monthly_part_demand["Month"].min(),
        monthly_part_demand["Month"].max(),
        monthly_part_demand["Month"].nunique(),
        "active_months >= 24, total_transactions >= 50, total_sold_qty > 0",
        monthly_demand_99,
        monthly_demand_999,
        len(transaction_outliers)
    ]
})


# ------------------------------------------------------------
# 11. SAVE REPORT
# ------------------------------------------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    monthly_part_demand.to_excel(writer, sheet_name="monthly_part_demand", index=False)
    part_summary.to_excel(writer, sheet_name="part_summary", index=False)
    forecastable_parts.to_excel(writer, sheet_name="forecastable_parts", index=False)
    top_50_candidates.to_excel(writer, sheet_name="top_50_candidates", index=False)
    monthly_forecastable_demand.to_excel(writer, sheet_name="monthly_forecastable", index=False)
    transaction_outliers.head(200).to_excel(writer, sheet_name="transaction_outliers", index=False)
    monthly_demand_outliers.head(200).to_excel(writer, sheet_name="monthly_demand_outliers", index=False)


# Save the most important sheets as CSV too, because future scripts can load CSV faster.
monthly_part_demand.to_csv(
    os.path.join(output_folder, "04_monthly_part_demand_v2.csv"),
    index=False
)

forecastable_parts.to_csv(
    os.path.join(output_folder, "04_forecastable_parts_v2.csv"),
    index=False
)

monthly_forecastable_demand.to_csv(
    os.path.join(output_folder, "04_monthly_forecastable_demand_v2.csv"),
    index=False
)


# ------------------------------------------------------------
# 12. PRINT RESULTS
# ------------------------------------------------------------
print("\nCreated:", output_file)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nTop 10 forecastable parts:")
print(
    forecastable_parts[
        [
            "Part Number",
            "active_months",
            "total_net_qty",
            "total_sold_qty",
            "total_return_qty",
            "total_transactions",
            "description",
            "fr"
        ]
    ].head(10).to_string(index=False)
)

print("\nCSV files created:")
print("results/xlsx/04_monthly_part_demand_v2.csv")
print("results/xlsx/04_forecastable_parts_v2.csv")
print("results/xlsx/04_monthly_forecastable_demand_v2.csv")





# (venv) PS D:\Diploma_work\diploma\diploma_v2> python 04_monthly_demand_table_v2.py
# Loaded combined raw data.
# Rows and columns: (376633, 34)

# Created: results/xlsx\04_monthly_demand_table_v2_report.xlsx

# Summary:
#                               metric                                                             value
#                 raw_transaction_rows                                                            376633
#                     raw_unique_parts                                                             35161
#             monthly_part_demand_rows                                                            148202
#                   forecastable_parts                                                              1001
#     monthly_forecastable_demand_rows                                                             31759
#                        minimum_month                                                           2023-01
#                        maximum_month                                                           2026-01
#                        unique_months                                                                37
#                    forecastable_rule active_months >= 24, total_transactions >= 50, total_sold_qty > 0
#      monthly_net_qty_99th_percentile                                                             150.0
#    monthly_net_qty_99_9th_percentile                                                          1411.578
# transaction_outlier_rows_abs_qty_100                                                              2619

# Top 10 forecastable parts:
#        Part Number  active_months  total_net_qty  total_sold_qty  total_return_qty  total_transactions                 description fr
#   MBN007603 014106             37         4233.0          4355.0             122.0                4441             SEAL RING,VLRUB MB
#        LU481371111             37        43261.0         44058.0             797.0                3027 MOBIL SUPER 3000XE 5W30 0.5 LU
# MBN000000 00200864             37         3177.0          3229.0              52.0                2984                     BATTERY MB
#   MBA270 180 01 09             37        29359.0         29429.0              70.0                2802      #TS OIL FILTER ELEMENT MB
#   MBA002 990 20 17             37         3059.0          3110.0              51.0                2771                  SCREW PLUG MB
#   MBA166 830 02 18             37         4804.0          4854.0              50.0                2613                #DUST FILTER MB
#            ZZSUND2             37         2513.0          2554.0              41.0                2595        Sundries/Consumables ZZ
#   MBA205 835 01 47             37         3254.0          3293.0              39.0                2224        FINE PARTICLE FILTER MB
#       CJ68191349AC             37         3035.0          3060.0              25.0                1778                  OIL FILTER CJ
#   MBA264 094 01 00             37         2210.0          2236.0              26.0                1583           AIR FILTER INSERT MB

# CSV files created:
# results/xlsx/04_monthly_part_demand_v2.csv
# results/xlsx/04_forecastable_parts_v2.csv
# results/xlsx/04_monthly_forecastable_demand_v2.csv