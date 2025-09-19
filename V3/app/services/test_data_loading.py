# test_data_loading.py
from app.services.data_service import DataService

print("Testing ETRM data loading...")
df = DataService.load_etrm_data()
if df is not None:
    print("Success! Data loaded with shape:", df.shape)
    print("First 5 records:")
    print(df.head())
else:
    print("Failed to load data")