import pandas as pd
from backend.scripts.load_to_database import get_category_for_product, get_tab_for_product
import backend.scripts.load_to_database as ltdb

def _load_from_temp():
    ltdb.CATEGORY_LOOKUP.clear()
    ltdb.TAB_LOOKUP.clear()
    df = pd.read_excel('temp_excel.xlsx')
    for _, row in df.iterrows():
        if pd.notna(row.get('Barcode')):
            barcode = str(row['Barcode']).strip().upper()
            if barcode and pd.notna(row.get('Category')):
                ltdb.CATEGORY_LOOKUP[barcode] = row['Category'].strip()
            if barcode and pd.notna(row.get('Tab')):
                ltdb.TAB_LOOKUP[barcode] = row['Tab'].strip()

_load_from_temp()
print('Testing fuzzy matching:')
desc = 'Gingerbon regular 125g'
cat_desc = get_category_for_product(None, desc)
tab_desc = get_tab_for_product(None, desc)
print(f'Description "{desc}": Category={cat_desc}, Tab={tab_desc}')
