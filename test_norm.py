import pandas as pd
from backend.scripts.load_to_database import normalize_text
import backend.scripts.load_to_database as ltdb

print('Testing normalize_text function:')
test_strings = ['Gingerbon regular 125g', 'GINGERBON REGULAR 125G', 'gingerbon regular 125g']
for s in test_strings:
    norm = normalize_text(s)
    print(f'Original: "{s}" -> Normalized: "{norm}"')

print()
print('Checking if normalized version exists in PRODUCT_NAME_LOOKUP:')
ltdb.PRODUCT_NAME_LOOKUP.clear()
df = pd.read_excel('temp_excel.xlsx')
for _, row in df.iterrows():
    if pd.notna(row.get('Product Name')):
        product_name = str(row['Product Name']).strip()
        normalized_name = normalize_text(product_name)
        if normalized_name and pd.notna(row.get('Category')) and pd.notna(row.get('Tab')):
            ltdb.PRODUCT_NAME_LOOKUP[normalized_name] = (
                row['Category'].strip(),
                row['Tab'].strip()
            )

print('PRODUCT_NAME_LOOKUP size:', len(ltdb.PRODUCT_NAME_LOOKUP))
target_norm = normalize_text('Gingerbon regular 125g')
print(f'Target normalized: "{target_norm}"')
print(f'Exists in lookup: {target_norm in ltdb.PRODUCT_NAME_LOOKUP}')
if target_norm in ltdb.PRODUCT_NAME_LOOKUP:
    print(f'Value: {ltdb.PRODUCT_NAME_LOOKUP[target_norm]}')
else:
    print('First 10 keys in lookup:')
    for k in list(ltdb.PRODUCT_NAME_LOOKUP.keys())[:10]:
        print(f'  "{k}"')
