import os
import pandas as pd

# ============================================================
# 01_check_2023_file.py
#
# Purpose:
# This script checks the new 2023 spare-parts database before
# we combine it with the 2024-2025 database.
#
# We are checking:
# 1. Can Python read the file?
# 2. What columns does it have?
# 3. Are important columns present?
# 4. Does the Excel date convert correctly?
# 5. Are rows/columns shifted or messy?
# ============================================================


# ------------------------------------------------------------
# 1. FILE PATH
# ------------------------------------------------------------
# Put your real 2023 file path here.
# Example:
# file_path = "D:/Diploma_work/diploma_v2/DB_2023.xlsb"

file_path = "D:/Diploma_work/diploma/diploma_v2/DB_31_2023.xlsb"


# ------------------------------------------------------------
# 2. READ FILE
# ------------------------------------------------------------
# .xlsb means Excel Binary Workbook.
# pandas can read it using engine="pyxlsb".
#
# If your 2023 file is .xlsx, use:
# df = pd.read_excel(file_path, engine="openpyxl")

df = pd.read_excel(file_path, engine="pyxlsb")


# ------------------------------------------------------------
# 3. BASIC FILE INFORMATION
# ------------------------------------------------------------
print("\n================ BASIC FILE INFO ================")
print("File path:", file_path)
print("Rows and columns:", df.shape)

print("\n================ COLUMN NAMES ================")
for col in df.columns:
    print(col)


# ------------------------------------------------------------
# 4. STANDARDISE COLUMN NAMES
# ------------------------------------------------------------
# In the old database, the part column was called:
# "Part Number"
#
# In the 2023 screenshot, it looks like:
# "PartNo"
#
# To make the new file compatible with our old scripts,
# we rename PartNo to Part Number if needed.

df = df.rename(columns={
    "PartNo": "Part Number",
    "PartNo.": "Part Number",
    "Part No": "Part Number"
})


# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------
# These are the important columns for our forecasting project.

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
    "Cost val",
    "Profit",
    "Qty",
    "BR"
]

missing_columns = [col for col in required_columns if col not in df.columns]

print("\n================ REQUIRED COLUMN CHECK ================")

if len(missing_columns) == 0:
    print("All required columns are present.")
else:
    print("Missing columns:")
    for col in missing_columns:
        print("-", col)


# ------------------------------------------------------------
# 6. CONVERT DATE
# ------------------------------------------------------------
# The Date column in this database is stored as Excel serial date.
#
# Example:
# 44928 means 2023-01-02.
#
# origin="1899-12-30" is used because this matches Excel's
# date numbering system.

if "Date" in df.columns:
    df["Date Converted"] = pd.to_datetime(
        df["Date"],
        origin="1899-12-30",
        unit="D",
        errors="coerce"
    )

    print("\n================ DATE CHECK ================")
    print("Minimum raw Date:", df["Date"].min())
    print("Maximum raw Date:", df["Date"].max())
    print("Minimum converted Date:", df["Date Converted"].min())
    print("Maximum converted Date:", df["Date Converted"].max())
    print("Invalid converted dates:", df["Date Converted"].isna().sum())
else:
    print("\nDate column is missing, so date conversion was skipped.")


# ------------------------------------------------------------
# 7. NUMERIC COLUMN CHECK
# ------------------------------------------------------------
# These columns should behave like numbers.
# If many values become NaN after conversion, something is wrong.

numeric_columns = [
    "Retail Val",
    "Sale val",
    "Cost val",
    "Profit",
    "Qty"
]

print("\n================ NUMERIC COLUMN CHECK ================")

for col in numeric_columns:
    if col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid_count = converted.isna().sum()

        print(f"{col}:")
        print("  Invalid numeric values:", invalid_count)
        print("  Min:", converted.min())
        print("  Max:", converted.max())
    else:
        print(f"{col}: column missing")


# ------------------------------------------------------------
# 8. IMPORTANT COLUMNS SAMPLE
# ------------------------------------------------------------
# This helps us visually confirm that columns are not shifted.
#
# For example:
# - Date should look like 2023 dates after conversion.
# - Qty should contain quantities.
# - Part Number should contain part codes.
# - Description should contain part names.

check_columns = [
    "FR",
    "Part Number",
    "Description",
    "WIPNo",
    "Invoice",
    "Date",
    "Date Converted",
    "Account",
    "Retail Val",
    "Sale val",
    "Cost val",
    "Profit",
    "Qty",
    "BR"
]

existing_check_columns = [col for col in check_columns if col in df.columns]

print("\n================ FIRST 10 IMPORTANT ROWS ================")
print(df[existing_check_columns].head(10).to_string(index=False))


# ------------------------------------------------------------
# 9. SAVE CHECK OUTPUT
# ------------------------------------------------------------
# Saving a small checked sample helps us inspect it in Excel.

output_folder = "results/xlsx"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "01_check_2023_file_output.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.head(100).to_excel(writer, sheet_name="first_100_rows", index=False)

    pd.DataFrame({
        "column_name": df.columns
    }).to_excel(writer, sheet_name="columns", index=False)

    pd.DataFrame({
        "missing_required_columns": missing_columns
    }).to_excel(writer, sheet_name="missing_columns", index=False)

print("\n================ DONE ================")
print("Created:", output_file)
print("Now open this Excel output and check first_100_rows manually.")