# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({
        'prediction': int(prediction),
        'message': 'Fraud' if prediction == 1 else 'Legitimate'
    })

if __name__ == '__main__':
    app.run(debug=True)