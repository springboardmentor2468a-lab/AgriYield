from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# ================= LOAD MODELS =================
crop_model = joblib.load("crop_recommendation_model.pkl")
crop_columns = joblib.load("crop_label_encoder_columns.pkl")

# Extract crop names
AVAILABLE_CROPS = sorted(
    col.replace("Crop_", "").capitalize()
    for col in crop_columns if col.startswith("Crop_")
)

# ================= HELPER: YIELD ESTIMATION =================
def estimate_yield_kg(N, rainfall, temperature, humidity, crop):
    """
    Rule-based agronomic yield estimation
    Output: kg per hectare
    """

    # Base yield ranges (kg/ha) — conservative & realistic
    BASE_YIELD = {
        "Rice": 3500,
        "Maize": 3000,
        "Jute": 2500,
        "Coffee": 900,
        "Banana": 20000,
        "Mango": 8000,
        "Papaya": 12000,
        "Cotton": 1800,
        "Pigeonpeas": 1200,
        "Chickpea": 1000,
        "Lentil": 900,
        "Default": 1500
    }

    base = BASE_YIELD.get(crop, BASE_YIELD["Default"])

    # Climate penalties
    if temperature < 10 or temperature > 45:
        temp_factor = 0.3
    elif 20 <= temperature <= 35:
        temp_factor = 1.0
    else:
        temp_factor = 0.7

    # Rainfall factor
    rain_factor = min(rainfall / 200, 1.2)

    # Nitrogen factor
    nutrient_factor = min(N / 90, 1.1)

    # Humidity factor
    humidity_factor = min(humidity / 80, 1.05)

    yield_kg = base * temp_factor * rain_factor * nutrient_factor * humidity_factor
    return round(yield_kg, 2)

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get-crops")
def get_crops():
    return jsonify(AVAILABLE_CROPS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # -------- INPUTS --------
    N = float(data["N"])
    P = float(data["P"])
    K = float(data["K"])
    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    ph = float(data["ph"])
    rainfall = float(data["rainfall"])
    selected_crop = data["selected_crop"].capitalize()

    # -------- CROP RECOMMENDATION (ML) --------
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    probabilities = crop_model.predict_proba(features)[0]
    crop_names = crop_model.classes_

    top_indices = np.argsort(probabilities)[::-1][:5]

    top_crops = [
        {
            "crop": crop_names[i],
            "score": round(probabilities[i] * 100, 2)
        }
        for i in top_indices
    ]

    # -------- YIELD ESTIMATION (LOGIC) --------
    predicted_yield_kg = estimate_yield_kg(
        N, rainfall, temperature, humidity, selected_crop
    )

    return jsonify({
        "selected_crop": selected_crop,
        "predicted_yield_kg": predicted_yield_kg,
        "top_crops": top_crops
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
