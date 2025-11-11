import sys
sys.path.append('backend/scripts')
from load_to_database import run_full_load
import pandas as pd

df = pd.DataFrame({
    'Date': ['2023-01-01'],
    'Receipt': ['R001'],
    'SO': [None],
    'Item Code': ['123'],
    'Description': ['Coca-Cola Original Taste 195ml'],
    'Qty': [1],
    'Unit': ['pcs'],
    'Sales': [10],
    'Cost': [8],
    'Profit': [2],
    'Payment': ['Cash'],
    'Cashier ID': ['C1']
})

try:
    run_id = run_full_load('test_file.csv', df)
    print(f'ETL run completed with run_id: {run_id}')
except Exception as e:
    print(f'Error: {e}')
