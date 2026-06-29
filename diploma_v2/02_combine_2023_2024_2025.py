import os
import pandas as pd

# ============================================================
# 02_combine_2023_2024_2025.py
#
# Purpose:
# Combine the new 2023 database with the old 2024-2025/2026
# database into one clean raw dataset.
#
# This does NOT build the forecasting model yet.
# It only prepares one combined source file for the next steps.
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------
# 2023 database
file_2023 = "D:/Diploma_work/diploma/diploma_v2/DB_31_2023.xlsb"

# Old database with 2024-2025 and January 2026 data.
# Change this if your old file has a different location/name.
file_2024_2025 = "D:/Diploma_work/diploma/DB_31_24&25.xlsb"

# Output folder
output_folder = "results/xlsx"
os.makedirs(output_folder, exist_ok=True)


# ------------------------------------------------------------
# 2. HELPER FUNCTION: LOAD AND STANDARDISE ONE FILE
# ------------------------------------------------------------
def load_and_standardise(file_path, source_label):
    print(f"\nLoading: {file_path}")

    # Read Excel Binary Workbook
    df = pd.read_excel(file_path, engine="pyxlsb")

    print(f"{source_label} original shape:", df.shape)

    # Standardise part number column name.
    # 2023 uses PartNo, old file uses Part Number.
    df = df.rename(columns={
        "PartNo": "Part Number",
        "PartNo.": "Part Number",
        "Part No": "Part Number"
    })

    # Add source label so we can trace where each row came from.
    df["source_file"] = source_label

    return df


# ------------------------------------------------------------
# 3. LOAD BOTH DATASETS
# ------------------------------------------------------------
df_2023 = load_and_standardise(file_2023, "2023")
df_old = load_and_standardise(file_2024_2025, "2024_2025_2026")


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------
# These columns are needed for our demand forecasting pipeline.
required_columns = [
    "FR",
    "Part Number",
    "Description",
    "WIPNo",
    "Invoice",
    "Date",
    "Account",
    "Retail Val",
    "Sale val",
    "Disc val",
    "Disc %",
    "Cost val",
    "Profit",
    "PC",
    "Qty",
    "DIV",
    "BR",
    "D",
    "AT",
    "AC",
    "A",
    "ST",
    "MLI",
    "DC",
    "LC",
    "SC",
    "ACLS",
    "DVC",
    "VLP",
    "VNP",
    "L",
    "source_file"
]

missing_2023 = [col for col in required_columns if col not in df_2023.columns]
missing_old = [col for col in required_columns if col not in df_old.columns]

print("\n================ REQUIRED COLUMN CHECK ================")
print("Missing in 2023:", missing_2023)
print("Missing in old file:", missing_old)

if missing_2023 or missing_old:
    raise ValueError(
        "Some required columns are missing. Fix column names before combining."
    )


# ------------------------------------------------------------
# 5. KEEP SAME COLUMNS IN SAME ORDER
# ------------------------------------------------------------
# This prevents column mismatch problems.
df_2023_clean = df_2023[required_columns].copy()
df_old_clean = df_old[required_columns].copy()


# ------------------------------------------------------------
# 6. COMBINE ROWS
# ------------------------------------------------------------
combined_df = pd.concat(
    [df_2023_clean, df_old_clean],
    ignore_index=True
)

print("\n================ COMBINED SHAPE CHECK ================")
print("2023 rows:", len(df_2023_clean))
print("Old rows:", len(df_old_clean))
print("Expected combined rows:", len(df_2023_clean) + len(df_old_clean))
print("Actual combined rows:", len(combined_df))


# ------------------------------------------------------------
# 7. CONVERT DATE
# ------------------------------------------------------------
# Date is Excel serial date.
combined_df["Date Converted"] = pd.to_datetime(
    combined_df["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

combined_df["Month"] = combined_df["Date Converted"].dt.to_period("M").astype(str)


# ------------------------------------------------------------
# 8. NUMERIC CLEANING
# ------------------------------------------------------------
# Convert important numeric columns to numeric type.
numeric_columns = [
    "Retail Val",
    "Sale val",
    "Disc val",
    "Disc %",
    "Cost val",
    "Profit",
    "PC",
    "Qty",
    "DIV",
    "BR",
    "D",
    "L"
]

for col in numeric_columns:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")


# ------------------------------------------------------------
# 9. VALIDATION SUMMARY
# ------------------------------------------------------------
summary = pd.DataFrame({
    "metric": [
        "rows_2023",
        "rows_old_2024_2025_2026",
        "rows_combined",
        "columns_combined",
        "minimum_date",
        "maximum_date",
        "unique_months",
        "invalid_dates",
        "unique_parts",
        "total_qty",
        "negative_qty_rows",
        "zero_qty_rows",
        "positive_qty_rows"
    ],
    "value": [
        len(df_2023_clean),
        len(df_old_clean),
        len(combined_df),
        len(combined_df.columns),
        combined_df["Date Converted"].min(),
        combined_df["Date Converted"].max(),
        combined_df["Month"].nunique(),
        combined_df["Date Converted"].isna().sum(),
        combined_df["Part Number"].nunique(),
        combined_df["Qty"].sum(),
        (combined_df["Qty"] < 0).sum(),
        (combined_df["Qty"] == 0).sum(),
        (combined_df["Qty"] > 0).sum()
    ]
})

monthly_row_count = (
    combined_df.groupby(["source_file", "Month"], as_index=False)
    .agg(row_count=("Qty", "count"), total_qty=("Qty", "sum"))
    .sort_values(["source_file", "Month"])
)

important_sample = combined_df[
    [
        "source_file",
        "FR",
        "Part Number",
        "Description",
        "Invoice",
        "Date",
        "Date Converted",
        "Month",
        "Account",
        "Retail Val",
        "Sale val",
        "Cost val",
        "Profit",
        "Qty",
        "BR"
    ]
].head(50)


print("\n================ VALIDATION SUMMARY ================")
print(summary.to_string(index=False))

print("\n================ MONTHLY ROW COUNT SAMPLE ================")
print(monthly_row_count.head(20).to_string(index=False))


# ------------------------------------------------------------
# 10. SAVE OUTPUTS
# ------------------------------------------------------------
output_file = os.path.join(output_folder, "02_combined_2023_2024_2025_raw.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    monthly_row_count.to_excel(writer, sheet_name="monthly_row_count", index=False)
    important_sample.to_excel(writer, sheet_name="important_sample", index=False)

    # Full combined data may be large but should still fit Excel:
    # 127,496 + 249,137 = about 376,633 rows.
    combined_df.to_excel(writer, sheet_name="combined_raw_data", index=False)


# Also save CSV because CSV is faster/easier for future scripts.
output_csv = os.path.join(output_folder, "02_combined_2023_2024_2025_raw.csv")
combined_df.to_csv(output_csv, index=False)


print("\n================ DONE ================")
print("Created Excel:", output_file)
print("Created CSV:", output_csv)
print("Next: open the Excel summary and confirm date range/monthly row counts.")