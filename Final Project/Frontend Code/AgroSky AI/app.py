from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# App Init
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Models
# -----------------------------
try:
    # Crop recommendation models
    crop_classifier = joblib.load("crop_classifier.pkl")
    classifier_scaler = joblib.load("classifier_scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Yield prediction models (USER BASED)
    yield_model = joblib.load("yield_model_user_based.pkl")
    yield_scaler = joblib.load("yield_scaler_user_based.pkl")

    print("✅ All models loaded")
    print("Yield model features:", yield_model.n_features_in_)

except Exception as e:
    print("❌ Error loading models:", e)
    raise

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "classifier_features": classifier_scaler.n_features_in_,
        "yield_model_features": yield_model.n_features_in_,
        "message": "AgroSky AI Backend is running"
    })

# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # -----------------------------
        # Validate Input
        # -----------------------------
        required_fields = [
            "nitrogen",
            "phosphorus",
            "potassium",
            "temperature",
            "humidity",
            "rainfall",
            "ph"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # -----------------------------
        # Build Feature Array (USER INPUT)
        # Order MUST match training
        # -----------------------------
        features = np.array([[
            float(data["nitrogen"]),
            float(data["phosphorus"]),
            float(data["potassium"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["rainfall"]),
            float(data["ph"])
        ]])

        # -----------------------------
        # Crop Recommendation
        # -----------------------------
        features_cls_scaled = classifier_scaler.transform(features)
        crop_encoded = crop_classifier.predict(features_cls_scaled)
        crop_name = label_encoder.inverse_transform(crop_encoded)[0]

        # -----------------------------
        # Yield Prediction
        # -----------------------------
        features_yield_scaled = yield_scaler.transform(features)
        predicted_yield = yield_model.predict(features_yield_scaled)[0]

        # -----------------------------
        # Response
        # -----------------------------
        return jsonify({
            "recommended_crop": crop_name,
            "predicted_yield": round(float(predicted_yield), 2),
            "unit": "kg/ha",
            "model": "Random Forest",
            "platform": "AgroSky AI"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
