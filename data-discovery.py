import pandas as pd

file_path = 'D:/Diploma_work/diploma/DB_31_24&25.xlsb'
df = pd.read_excel(file_path, engine='pyxlsb')



# 1. Save first rows as CSV
df.head(50).to_excel("sample_50_rows.xlsx", index=False)

print("Done.")
print("Created files:")
print("- sample_50_rows.xlsx")

