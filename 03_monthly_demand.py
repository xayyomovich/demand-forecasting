import pandas as pd

file_path = 'D:/Diploma_work/diploma/DB_31_24&25.xlsb'

df = pd.read_excel(file_path, engine="pyxlsb")

df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin = '1899-12-30',                # Excel's epoch starts at day 1 = January 1, 1900, but Microsoft intentionally introduced an off-by-two bug (falsely treating 1900 as a leap year). The correct offset to reproduce Excel's behavior is 1899-12-30 as day 0.
    unit = "D",                           # The raw values are floating-point day counts (e.g., 45000.0), so pandas adds that many days to the origin.
    errors = "coerce"                     # Any unparseable value (empty cell, text, NaT-inducing garbage) becomes NaT instead of raising an exception.
)

df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)




# Qty can be positive or negative.
# Positive Qty = sold quantity.
# Negative Qty = return / credit note.
#
# net_qty = sold quantity minus returns
# sold_qty = only positive sales
# return_qty = absolute value of negative quantities

df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()




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




# Our discovery showed minimum Qty = -1440 and maximum Qty = 5760.
# These are huge values, so we save them for inspection.
outlier_qty_rows = df[
    (df["Qty"] <= -100) | (df["Qty"] >= 100)
].sort_values("Qty", ascending=False)



output_file = "03_monthly_demand_report.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    monthly_part_demand.to_excel(writer, sheet_name="monthly_part_demand", index=False)
    part_summary.to_excel(writer, sheet_name="part_summary", index=False)
    forecastable_parts.to_excel(writer, sheet_name="forecastable_parts", index=False)
    top_50_candidates.to_excel(writer, sheet_name="top_50_candidates", index=False)
    outlier_qty_rows.head(50).to_excel(writer, sheet_name="outlier_qty_rows", index=False)


print(f"Created: {output_file}")
print("Monthly part demand rows:", len(monthly_part_demand))
print("Total unique parts:", part_summary["Part Number"].nunique())
print("Forecastable parts:", len(forecastable_parts))
print("Top 50 candidates saved.")




























