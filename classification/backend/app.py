from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    FEATURES = ['n','p','k','temperature','humidity','ph','rainfall']
    features = np.array([[data[f] for f in FEATURES]])

    prediction = model.predict(features)

    return jsonify({"label": prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)
