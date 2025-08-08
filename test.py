import joblib

sectors = ["IT", "business", "semiconductor", "automobile", "Telecom"]

for sector in sectors:
    path = f"models/{sector}/stock_data.pkl"
    data = joblib.load(path)

    print(f"\n{sector.capitalize()} Companies:")

    if isinstance(data, list):
        print(data)

    elif isinstance(data, dict):
        print(list(data.keys()))

    elif hasattr(data, 'columns'):  
        if 'Company' in data.columns:
            print(data['Company'].unique().tolist())
        else:
            print("Company column not found in DataFrame.")
    else:
        print("Unknown data structure. Cannot extract company list.")
