import os 
import pandas as pd
import matplotlib.pyplot as plt     # the foundational Python plotting library. It controls the figure (canvas), axes, labels, titles, saving, and closing.
import seaborn as sns               #  built on top of matplotlib. It provides higher-level chart types with better default styling. Here it draws the actual bar chart in one line.


file_path = 'D:/Diploma_work/diploma/DB_31_24&25.xlsb'

df = pd.read_excel(file_path, engine="pyxlsb")

df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin = '1899-12-30',                # Excel's epoch starts at day 1 = January 1, 1900, but Microsoft intentionally introduced an off-by-two bug (falsely treating 1900 as a leap year). The correct offset to reproduce Excel's behavior is 1899-12-30 as day 0.
    unit = "D",                           # The raw values are floating-point day counts (e.g., 45000.0), so pandas adds that many days to the origin.
    errors = "coerce"                     # Any unparseable value (empty cell, text, NaT-inducing garbage) becomes NaT instead of raising an exception.
)

df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)

df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()

os.makedirs("outputs/charts/outliers", exist_ok=True)



#  MONTHLY PART DEMAND TABLE

monthly_part_demand = (
    df.groupby(["Month", "Part Number"], as_index=False)
    .agg(
        net_qty=("Qty", "sum"),
        sold_qty=("sold_qty", "sum"),
        return_qty=("return_qty", "sum"),
        transaction_count=("Qty", "count"),
        sale_value=("Sale val", "sum"),
        profit=("Profit", "sum"),
        description=("Description", "first"),
        fr=("FR","first")
    )
    .sort_values(["Part Number", "Month"])
)



#  PART-LEVEL FORECASTING CANDIDATE SUMMARY

part_summary = (
    monthly_part_demand.groupby("Part Number", as_index=False)
    .agg(
        active_months=("Month", "nunique"),
        total_net_qty=("net_qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        total_transactions=("transaction_count", "sum"),
        avg_monthly_net_qty=("net_qty", "mean"),
        max_monthly_net_qty=("net_qty", "max"),
        total_sale_value=("sale_value", "sum"),
        total_profit=("profit", "sum"),
        description=("description", "first"),
        fr=("fr", "first")
    )
)



# Forecastable parts rule:
# - appeared in at least 18 months
# - had at least 50 transactions
# - sold more than 0 total units
forecastable_parts = part_summary[
    (part_summary["active_months"] >= 18) &
    (part_summary["total_transactions"] >= 50) &
    (part_summary["total_sold_qty"] > 0)
].sort_values(
    ["active_months", "total_transactions", "total_sold_qty"], ascending=False
)

top_50_candidates = forecastable_parts.head(50)





###########################
#  OUTLIER DETECTION RULES
###########################

# Rule A: very large transaction-level quantities
transaction_outliers = df[
    (df["Qty"] >= 100) | (df["Qty"] <= -100)
].sort_values("Qty", ascending=False)


# Rule B: very large monthly part demand values
q99 = monthly_part_demand["net_qty"].quantile(0.99)       # A quantile divides your data into ranked portions. The quantile(0.99) call answers the question: "What value do 99% of all rows fall below?"  So its top 1%
q999 = monthly_part_demand["net_qty"].quantile(0.999)     # Top 0.1 %

monthly_part_demand["is_outlier_99"] = monthly_part_demand["net_qty"] >= q99       #  Adds a boolean column (True/False) to every row: True if that row's net_qty is in the top 1% of all monthly demand values, False otherwise.
monthly_part_demand["is_outlier_999"] = monthly_part_demand["net_qty"] >= q999

monthly_demand_outliers = monthly_part_demand[
    monthly_part_demand["is_outlier_99"]
    ].sort_values("net_qty", ascending=False)



#  CHART: DISTRIBUTION OF TRANSACTION QTY

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.histplot(df["Qty"], bins=100)
plt.title("Distribution of Transaction Quantity")
plt.xlabel("Transaction Quantity")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig("outputs/charts/outliers/transaction_qty_distribution.png", dpi=300)
plt.close()



#  CHART: ZOOMED TRANSACTION QTY DISTRIBUTION

zoom_qty = df[(df["Qty"] >= -50) & (df["Qty"] <= 50)]

plt.figure(figsize=(12, 6))
sns.histplot(zoom_qty["Qty"], bins=100)
plt.title("Distribution of Transaction Quantity (-50 to 50 Zoom)")
plt.xlabel("Transaction Quantity")
plt.ylabel("Row Count")
plt.tight_layout()
plt.savefig("outputs/charts/outliers/transaction_qty_distribution_zoomed.png", dpi=300)
plt.close()



#  CHART: TOP 20 LARGEST TRANSACTION QUANTITIES

top_transaction_outliers = transaction_outliers.head(20).copy()
top_transaction_outliers["label"] = (
    top_transaction_outliers["Part Number"].astype(str)
    + " | "
    + top_transaction_outliers["Description"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_transaction_outliers, x="Qty", y="label")
plt.title("Top 20 Largest Transaction Quantities")
plt.xlabel("Quantity")
plt.ylabel("Part")
plt.tight_layout()
plt.savefig("outputs/charts/outliers/top_20_transaction_qty_outliers.png", dpi=300)
plt.close()



#  CHART: MONTHLY NET QTY DISTRIBUTION

plt.figure(figsize=(12, 6))
sns.histplot(monthly_part_demand["net_qty"], bins=100)
plt.axvline(q99, color="red", linestyle="--", label=f"99th percentile = {q99:.2f}")
plt.axvline(q999, color="orange", linestyle="--", label=f"99.9th percentile = {q999:.2f}")
plt.title("Distribution of Monthly Part Net Quantity")
plt.xlabel("Monthly Net Quantity")
plt.ylabel("Part-Month Count")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/outliers/monthly_net_qty_distribution.png", dpi=300)
plt.close()



#  CHART: TOP 20 MONTHLY DEMAND OUTLIERS

top_monthly_outliers = monthly_demand_outliers.head(20).copy()
top_monthly_outliers["label"] = (
    top_monthly_outliers["Month"].astype(str)
    + " | "
    + top_monthly_outliers["Part Number"].astype(str)
    + " | "
    + top_monthly_outliers["description"].astype(str).str.slice(0, 30)
)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_monthly_outliers, x="net_qty", y="label")
plt.title("Top 20 Monthly Part Demand Outliers")
plt.xlabel("Monthly Net Quantity")
plt.ylabel("Month | Part")
plt.tight_layout()
plt.savefig("outputs/charts/outliers/top_20_monthly_demand_outliers.png", dpi=300)
plt.close()


outlier_summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "transaction_outlier_rows_abs_qty_100",
        "monthly_part_demand_rows",
        "monthly_net_qty_99th_percentile",
        "monthly_net_qty_99_9th_percentile",
        "monthly_outlier_rows_99th_percentile",
        "forecastable_parts",
        "top_50_candidates"
    ],
    "value": [
        len(df),
        len(transaction_outliers),
        len(monthly_part_demand),
        q99,
        q999,
        len(monthly_demand_outliers),
        len(forecastable_parts),
        len(top_50_candidates)
    ]
})



output_file = "04_outlier_and_candidate_check.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    outlier_summary.to_excel(writer, sheet_name="outlier_summary", index=False)
    transaction_outliers.head(1000).to_excel(writer, sheet_name="transaction_outliers", index=False)
    monthly_demand_outliers.head(1000).to_excel(writer, sheet_name="monthly_demand_outliers", index=False)
    forecastable_parts.to_excel(writer, sheet_name="forecastable_parts", index=False)
    top_50_candidates.to_excel(writer, sheet_name="top_50_candidates", index=False)

print(f"Created: {output_file}")
print("Charts saved in: outputs/charts/outliers/")
print("Transaction outlier rows:", len(transaction_outliers))
print("Monthly demand 99th percentile:", q99)
print("Monthly demand 99.9th percentile:", q999)
print("Monthly demand outlier rows:", len(monthly_demand_outliers))
print("Forecastable parts:", len(forecastable_parts))






