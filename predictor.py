import json
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf
from keras.models import load_model
import joblib
import datetime
from tabulate import tabulate

# Load mapping between tickers and model folders (e.g., "AAPL": "IT")
with open("model_mapping.json") as f:
    MODEL_MAP = json.load(f)

def predict_stock_price(company_ticker):
    if company_ticker not in MODEL_MAP:
        return {"error": f"Company '{company_ticker}' not found in model mapping."}

    model_key = MODEL_MAP[company_ticker]
    model_folder = os.path.join("models", model_key)

    try:
        model = load_model(os.path.join(model_folder, "stock_model.keras"))
        scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
        data = joblib.load(os.path.join(model_folder, "stock_data.pkl"))
    except Exception as e:
        return {"error": f"Error loading model or files: {str(e)}"}

    try:
        last_sequence = data[company_ticker][-60:]
    except KeyError:
        return {"error": f"Data for '{company_ticker}' not found in dataset."}

    if len(last_sequence) < 60:
        return {"error": "Insufficient data for prediction (need at least 60 time steps)."}

    future_predictions = []
    current_input = last_sequence.copy().reshape(60, 1)

    for _ in range(30):
        pred = model.predict(current_input.reshape(1, 60, 1), verbose=0)[0][0]
        future_predictions.append(pred)
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred

    try:
        predicted = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        historical = scaler.inverse_transform(data[company_ticker][-120:].reshape(-1, 1))
    except Exception as e:
        return {"error": f"Scaler inverse transform failed: {str(e)}"}

    try:
        plt.figure(figsize=(10, 4))
        plt.plot(historical, label='Historical')
        plt.plot(np.arange(len(historical), len(historical) + len(predicted)), predicted, label='Predicted')
        plt.title(f"📈 Prediction for {company_ticker}")
        plt.xlabel("Days")
        plt.ylabel("Stock Price")
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except Exception as e:
        return {"error": f"Plotting error: {str(e)}"}

        # 🧾 Format predicted values with future dates
    start_date = datetime.date.today() + datetime.timedelta(days=1)
    prediction_dates = [start_date + datetime.timedelta(days=i) for i in range(30)]
    prediction_table = list(zip(
        [date.strftime('%Y-%m-%d') for date in prediction_dates],
        predicted.flatten().tolist()
    ))

    # 🖨️ Print to console
    print("\n📊 Predicted Stock Prices for", company_ticker)
    print(tabulate(prediction_table, headers=["Date", "Predicted Price"], tablefmt="pretty"))

    return {
        "company": company_ticker,
        "predictions": predicted.flatten().tolist(),
        "plot": img_base64,
        "table": [
            {"date": date.strftime('%Y-%m-%d'), "price": float(price)}
            for date, price in zip(prediction_dates, predicted.flatten())
    ]
}
