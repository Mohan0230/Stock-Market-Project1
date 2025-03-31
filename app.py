from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Model & Scaler
loaded_model = joblib.load("linear_regression_stock_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive data from user input
    df = pd.DataFrame([data])  # Convert to DataFrame

    # Scale the input data
    scaled_features = scaler.transform(df)

    # Predict closing price
    prediction = loaded_model.predict(scaled_features)

    return jsonify({'Predicted Closing Price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
