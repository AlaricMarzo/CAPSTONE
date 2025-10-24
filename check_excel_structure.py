import pandas as pd
import os

# Load the Excel file
excel_path = 'Product List as of 10232024.xlsx'
df = pd.read_excel(excel_path)

print("Excel Columns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nSample data:")
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].head(3).tolist())
