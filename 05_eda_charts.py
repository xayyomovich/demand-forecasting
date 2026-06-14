import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "D:/Diploma_work/diploma/DB_31_24&25.xlsb"

df = pd.read_excel(file_path, engine="pyxlsb")

df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)

df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()

os.makedirs("outputs/charts/eda", exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. MONTHLY SUMMARY
# ============================================================

monthly_summary = (
    df.groupby("Month", as_index=False)
    .agg(
        net_qty=("Qty", "sum"),
        sold_qty=("sold_qty", "sum"),
        return_qty=("return_qty", "sum"),
        sale_value=("Sale val", "sum"),
        profit=("Profit", "sum"),
        transaction_count=("Qty", "count"),
        unique_parts=("Part Number", "nunique")
    )
    .sort_values("Month")
)


# ============================================================
# 3. MONTHLY NET QUANTITY TREND
# ============================================================

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_summary, x="Month", y="net_qty", marker="o")
plt.title("Monthly Net Quantity Trend")
plt.xlabel("Month")
plt.ylabel("Net Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/charts/eda/monthly_net_quantity_trend.png", dpi=300)
plt.close()


# ============================================================
# 4. MONTHLY SALES VALUE TREND
# ============================================================

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_summary, x="Month", y="sale_value", marker="o")
plt.title("Monthly Sales Value Trend")
plt.xlabel("Month")
plt.ylabel("Sales Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/charts/eda/monthly_sales_value_trend.png", dpi=300)
plt.close()


# ============================================================
# 5. MONTHLY SOLD VS RETURNED QUANTITY
# ============================================================

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_summary, x="Month", y="sold_qty", marker="o", label="Sold Quantity")
sns.lineplot(data=monthly_summary, x="Month", y="return_qty", marker="o", label="Returned Quantity")
plt.title("Monthly Sold Quantity vs Returned Quantity")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/eda/monthly_sold_vs_returned_quantity.png", dpi=300)
plt.close()


# ============================================================
# 6. UNIQUE PARTS BY MONTH
# ============================================================

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_summary, x="Month", y="unique_parts", marker="o")
plt.title("Unique Parts Sold by Month")
plt.xlabel("Month")
plt.ylabel("Unique Parts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/charts/eda/unique_parts_by_month.png", dpi=300)
plt.close()


# ============================================================
# 7. TOP 20 PARTS BY NET QUANTITY
# ============================================================

top_parts_qty = (
    df.groupby(["Part Number", "Description"], as_index=False)
    .agg(net_qty=("Qty", "sum"))
    .sort_values("net_qty", ascending=False)
    .head(20)
)

top_parts_qty["label"] = (
    top_parts_qty["Part Number"].astype(str)
    + " | "
    + top_parts_qty["Description"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_parts_qty, x="net_qty", y="label")
plt.title("Top 20 Parts by Net Quantity")
plt.xlabel("Net Quantity")
plt.ylabel("Part")
plt.tight_layout()
plt.savefig("outputs/charts/eda/top_20_parts_by_net_quantity.png", dpi=300)
plt.close()


# ============================================================
# 8. TOP 20 PARTS BY TRANSACTION COUNT
# ============================================================

top_parts_frequency = (
    df.groupby(["Part Number", "Description"], as_index=False)
    .agg(transaction_count=("Qty", "count"))
    .sort_values("transaction_count", ascending=False)
    .head(20)
)

top_parts_frequency["label"] = (
    top_parts_frequency["Part Number"].astype(str)
    + " | "
    + top_parts_frequency["Description"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_parts_frequency, x="transaction_count", y="label")
plt.title("Top 20 Parts by Transaction Count")
plt.xlabel("Transaction Count")
plt.ylabel("Part")
plt.tight_layout()
plt.savefig("outputs/charts/eda/top_20_parts_by_transaction_count.png", dpi=300)
plt.close()


# ============================================================
# 9. QUANTITY BY FRANCHISE
# ============================================================

fr_summary = (
    df.groupby("FR", as_index=False)
    .agg(net_qty=("Qty", "sum"))
    .sort_values("net_qty", ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(data=fr_summary, x="FR", y="net_qty")
plt.title("Net Quantity by Franchise")
plt.xlabel("Franchise")
plt.ylabel("Net Quantity")
plt.tight_layout()
plt.savefig("outputs/charts/eda/net_quantity_by_franchise.png", dpi=300)
plt.close()


# ============================================================
# 10. QUANTITY BY BRANCH
# ============================================================

branch_summary = (
    df.groupby("BR", as_index=False)
    .agg(net_qty=("Qty", "sum"))
    .sort_values("net_qty", ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(data=branch_summary, x="BR", y="net_qty")
plt.title("Net Quantity by Branch")
plt.xlabel("Branch")
plt.ylabel("Net Quantity")
plt.tight_layout()
plt.savefig("outputs/charts/eda/net_quantity_by_branch.png", dpi=300)
plt.close()


# ============================================================
# 11. SAVE TABLES
# ============================================================

output_file = "05_eda_charts_summary.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    monthly_summary.to_excel(writer, sheet_name="monthly_summary", index=False)
    top_parts_qty.to_excel(writer, sheet_name="top_parts_qty", index=False)
    top_parts_frequency.to_excel(writer, sheet_name="top_parts_frequency", index=False)
    fr_summary.to_excel(writer, sheet_name="fr_summary", index=False)
    branch_summary.to_excel(writer, sheet_name="branch_summary", index=False)

print(f"Created: {output_file}")
print("Charts saved in: outputs/charts/eda/")






"""EDA showed that demand is highly concentrated in the MB franchise and branch 31. 
Therefore, the first forecasting version uses aggregated demand across branches rather 
than separate branch-level models. Branch and franchise variables are retained for 
descriptive analysis and possible future model features."""