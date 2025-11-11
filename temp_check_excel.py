import pandas as pd
import sys
sys.path.insert(0, 'backend/scripts')
from load_to_database import normalize_text

# Check the Excel file
df = pd.read_excel('temp_excel.xlsx')
print('Sample product names from Excel:')
for i, row in df.head(10).iterrows():
    product = str(row.get('Product Name', '')).strip()
    normalized = normalize_text(product)
    print(f'{i+1}. "{product}" -> "{normalized}"')

print()
print('Check for Gingerbon products:')
gingerbon_products = df[df['Product Name'].str.contains('gingerbon', case=False, na=False)]
for i, row in gingerbon_products.iterrows():
    product = str(row.get('Product Name', '')).strip()
    normalized = normalize_text(product)
    category = row.get('Category', '')
    tab = row.get('Tab', '')
    print(f'Gingerbon: "{product}" -> "{normalized}" | Cat: {category} | Tab: {tab}')
