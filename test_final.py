import pandas as pd
from backend.scripts.load_to_database import get_category_for_product, get_tab_for_product
import backend.scripts.load_to_database as ltdb
from backend.scripts.load_to_database import normalize_text

# Load lookups
ltdb.CATEGORY_LOOKUP.clear()
ltdb.TAB_LOOKUP.clear()
ltdb.PRODUCT_NAME_LOOKUP.clear()
df = pd.read_excel('temp_excel.xlsx')
for _, row in df.iterrows():
    if pd.notna(row.get('Barcode')):
        barcode = str(row['Barcode']).strip().upper()
        if barcode and pd.notna(row.get('Category')):
            ltdb.CATEGORY_LOOKUP[barcode] = row['Category'].strip()
        if barcode and pd.notna(row.get('Tab')):
            ltdb.TAB_LOOKUP[barcode] = row['Tab'].strip()
    if pd.notna(row.get('Product Name')):
        product_name = str(row['Product Name']).strip()
        normalized_name = normalize_text(product_name)
        if normalized_name and pd.notna(row.get('Category')) and pd.notna(row.get('Tab')):
            ltdb.PRODUCT_NAME_LOOKUP[normalized_name] = (
                row['Category'].strip(),
                row['Tab'].strip()
            )

print('Testing get_category_for_product with description:')
desc = 'Gingerbon regular 125g'
cat = get_category_for_product(None, desc)
tab = get_tab_for_product(None, desc)
print(f'Description "{desc}": Category={cat}, Tab={tab}')
