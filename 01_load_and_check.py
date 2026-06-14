import pandas as pd

file_path = 'D:/Diploma_work/diploma/DB_31_24&25.xlsb'
df = pd.read_excel(file_path, engine='pyxlsb')

print("File loaded successfully.")
print("Rows and columns:", df.shape)

print("\nColumn list:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

# Expected number of columns from your file is 31.
expected_columns = 31

if df.shape[1] == expected_columns:
    print("\nColumn check: OK - 31 columns loaded.")
else:
    print("\nColumn check: WARNING - expected 31 columns, but got", df.shape[1])


# ============================================================
# 3. CONVERT EXCEL SERIAL DATE TO REAL DATE
# ============================================================

df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin='1899-12-30',
    unit='D',
    errors='coerce'
    )

print(df[["Date", "Date Converted"]].head(10))


# ============================================================
# 4. CONFIRM ROWS/COLUMNS ARE NOT SHIFTED
# ============================================================

# We check important columns together.
# If these values make business sense, columns are probably not shifted.
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
    "Qty",
    "BR"
]

print("\nFirst 10 rows of important columns:")
print(df[check_columns].head(10).to_string(index=False))



# ============================================================
# 5. SAVE THE CHECK RESULT TO EXCEL
# ============================================================

# Save a readable sample file so you can compare it with the original workbook.
df[check_columns].head(100).to_excel(
    "01_loaded_check_sample.xlsx",
    index=False
)

print("\nCreated file: 01_loaded_check_sample.xlsx")
