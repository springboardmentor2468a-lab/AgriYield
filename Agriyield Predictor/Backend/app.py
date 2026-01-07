from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ================================
# LOAD MODELS
# ================================
try:
    # Classification
    clf_model = joblib.load("crop_classification_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Regression
    reg_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")

except Exception as e:
    print("❌ Error loading model files:", e)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ================================
# HEALTH CHECK
# ================================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "Backend running",
        "models": {
            "classification": "crop_classification_model.pkl + label_encoder.pkl",
            "regression": "random_forest_model.pkl + scaler.pkl"
        }
    })


# ================================
# CLASSIFICATION API
# ================================
@app.route("/predict-classification", methods=["POST"])
def predict_classification():
    try:
        data = request.json

        # Validate input
        for f in FEATURES:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        # Create DataFrame (NO SCALER here)
        X = pd.DataFrame([data], columns=FEATURES)

        # Predict class index
        pred_idx = clf_model.predict(X)[0]

        # Convert index → crop name
        crop = label_encoder.inverse_transform([pred_idx])[0]

        # OPTIONAL: fake top-5 probabilities for chart
        crops = label_encoder.classes_
        probs = np.zeros(len(crops))
        probs[pred_idx] = 1.0

        top5 = dict(zip(crops[:5], probs[:5]))

        return jsonify({
            "prediction": crop,
            "top5": top5
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# REGRESSION API
# ================================
@app.route("/predict-regression", methods=["POST"])
def predict_regression():
    try:
        data = request.json

        # Validate input
        for f in FEATURES:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        # Create DataFrame
        X = pd.DataFrame([data], columns=FEATURES)

        # SCALE input (ONLY HERE)
        X_scaled = scaler.transform(X)

        # Predict yield
        predicted_yield = float(reg_model.predict(X_scaled)[0])

        # Dummy comparison data for visualization
        crops = ["rice", "maize", "cotton", "banana", "wheat"]
        top5 = {
            crop: round(predicted_yield - i * 0.3, 2)
            for i, crop in enumerate(crops)
        }

        return jsonify({
            "predicted_yield": round(predicted_yield, 2),
            "top5": top5
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

