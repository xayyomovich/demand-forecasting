import pandas as pd

file_path = 'D:/Diploma_work/diploma/DB_31_24&25.xlsb'

df = pd.read_excel(file_path, engine='pyxlsb')


# Your Date column is stored as Excel serial numbers.
# Example: 45350 means 2024-02-28.
df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin = '1899-12-30',
    unit = 'D',
    errors = 'coerce'
)

# Create Month column for future demand forecasting.
df['Month'] = df['Date Converted'].dt.to_period('M').astype(str)


# ============================================================
# 3. COLUMN SUMMARY
# ============================================================

# This table explains each column technically:
# dtype = data type
# missing_count = how many empty values
# missing_percent = percent of missing values
# unique_values = how many different values exist
column_summary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_count": df.isnull().sum().values,
    "missing_percent": (df.isnull().sum().values / len(df) * 100).round(2),
    "unique_values": df.nunique(dropna=True).values
})


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
        df["Date Converted"].isnull().sum()
    ]
})


qty_summary = pd.DataFrame({
    "metric": [
        "total_qty",
        "positive_qty_rows",
        "negative_qty_rows",
        "zero_qty_rows",
        "minimum_qty",
        "maximum_qty",
        "average_qty"
    ],
    "value": [
        df['Qty'].sum(),
        (df['Qty'] > 0).sum(),
        (df['Qty'] < 0).sum(),
        (df['Qty'] == 0).sum(),
        df["Qty"].min(),
        df["Qty"].max(),
        df["Qty"].mean()
    ]
})

business_summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "total_columns",
        "unique_parts",
        "unique_descriptions",
        "unique_fr",
        "unique_branches",
        "unique_accounts",
        "total_sale_value",
        "total_retail_value",
        "total_cost_value",
        "total_profit"
    ],

    "value": [
        len(df),
        len(df.columns),
        df['Part Number'].nunique(),
        df['Description'].nunique(),
        df['FR'].nunique(),
        df['BR'].nunique(),
        df['Account'].nunique(),
        df['Sale val'].sum(),
        df["Retail Val"].sum(),
        df["Cost val"].sum(),
        df["Profit"].sum()
    ]
})


# ============================================================
# 7. TOP PARTS ANALYSIS
# ============================================================

# Top parts by total quantity.
top_parts_by_qty = (
    df.groupby('Part Number', as_index=False)
        .agg(
            total_qty=("Qty", "sum"),
            transaction_count=("Qty", "count"),
            active_months=("Month", "nunique"),
            total_sale_value=("Sale val", "sum"),
            total_profit=("Profit", "sum"),
            description=("Description", "first"),
            fr=("FR", "first")
        )
        .sort_values('total_qty', ascending=False)
)


# Top parts by transaction frequency.
# This is important because frequent parts are usually easier to forecast.
top_parts_by_frequency = (
    df.groupby("Part Number", as_index=False)
    .agg(
        transaction_count=("Qty", "count"),
        total_qty=("Qty", "sum"),
        active_months=("Month", "nunique"),
        description=("Description", "first"),
        fr=("FR", "first")
    )
    .sort_values("transaction_count", ascending=False)
)


# ============================================================
# 8. FRANCHISE AND BRANCH SUMMARY
# ============================================================


fr_summary = (
    df.groupby("FR", as_index=False)
    .agg(
        rows=("FR", "count"),
        total_qty=("Qty", "sum"),
        unique_parts=("Part Number", "nunique"),
        total_sale_value=("Sale val", "sum"),
        total_profit=("Profit", "sum")
    )
    .sort_values("rows", ascending=False)
)

 
branch_summary = (
    df.groupby("BR", as_index=False)
    .agg(
        rows=("BR", "count"),
        total_qty=("Qty", "sum"),
        unique_parts=("Part Number", "nunique"),
        total_sale_value=("Sale val", "sum"),
        total_profit=("Profit", "sum")
    )
    .sort_values("rows", ascending=False)
)


# ============================================================
# 9. MONTHLY SUMMARY
# ============================================================

monthly_summary=(
    df.groupby("Month", as_index=False)
    .agg(
        total_qty=("Qty", "sum"),
        total_sale_value=("Sale val", "sum"),
        total_profit=("Profit", "sum"),
        transaction_count=("Qty", "count"),
        unique_parts=("Part Number", "nunique")
    )
    .sort_values("Month", ascending=False)
)


# ============================================================
# 10. SAVE EVERYTHING TO EXCEL REPORT
# ============================================================

output_file = "02_data_discovery_report.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.head(1000).to_excel(writer, sheet_name="sample_rows", index=False)     #index=False tells pandas don't write the row numbers (0, 1, 2, 3...) into the Excel file. Without it your Excel would have an extra ugly number column on the left.
    column_summary.to_excel(writer, sheet_name="column_summary", index=False)
    business_summary.to_excel(writer, sheet_name="business_summary", index=False)
    date_summary.to_excel(writer, sheet_name="date_summary", index=False)
    qty_summary.to_excel(writer, sheet_name="qty_summary", index=False)
    top_parts_by_qty.head(100).to_excel(writer, sheet_name="top_parts_by_qty", index=False)
    top_parts_by_frequency.head(100).to_excel(writer, sheet_name="top_parts_by_frequency", index=False)
    fr_summary.to_excel(writer, sheet_name="fr_summary", index=False)
    branch_summary.to_excel(writer, sheet_name="branch_summary", index=False)
    monthly_summary.to_excel(writer, sheet_name="monthly_summary", index=False)


print(f"Created: {output_file}")
print("Open this Excel file and inspect each sheet.")



