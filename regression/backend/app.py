from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL & FEATURES ----------------
model = joblib.load("crop_weather_model.pkl")
FEATURES = joblib.load("feature_list.pkl")

# ---------------- CROP REFERENCE INDEX ----------------
# (used only for relative scaling)
CROP_REFERENCE_INDEX = {
    "apple": 1500,
    "banana": 1700,
    "chickpea": 1700,
    "coconut": 1200,
    "coffee": 1700,
    "cotton": 1900,
    "grapes": 2000,
    "jute": 1800,
    "lentil": 1600,
    "maize": 1900,
    "mango": 2400,
    "mothbeans": 2000,
    "muskmelon": 1600,
    "orange": 1300,
    "papaya": 2300,
    "pigeonpeas": 1900,
    "rice": 1700,
    "watermelon": 1600
}

# ---------------- AVERAGE YIELD (FAO / GOVT ESTIMATES) ----------------
# Metric tons per hectare (approximate)
CROP_AVG_YIELD_TONS = {
    "apple": 15.0,
    "banana": 20.0,
    "chickpea": 1.2,
    "coconut": 6.0,
    "coffee": 1.8,
    "cotton": 2.2,
    "grapes": 10.0,
    "jute": 2.5,
    "lentil": 1.3,
    "maize": 5.5,
    "mango": 9.0,
    "mothbeans": 0.8,
    "muskmelon": 3.5,
    "orange": 12.0,
    "papaya": 35.0,
    "pigeonpeas": 1.4,
    "rice": 4.0,
    "watermelon": 4.5
}

@app.route("/")
def home():
    return "AgriYield Predictor API (Estimated Metric Tons) is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    target_crop = data.pop("targetcrop")

    # -------- ML PRODUCTIVITY INDEX --------
    input_df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    productivity_index = float(model.predict(input_df)[0])

    # -------- ESTIMATED METRIC TONS (TARGET CROP) --------
    base_index = CROP_REFERENCE_INDEX[target_crop]
    avg_tons = CROP_AVG_YIELD_TONS[target_crop]

    estimated_tons = round(
        (productivity_index / base_index) * avg_tons,
        2
    )

    # -------- GRAPH DATA (ESTIMATED METRIC TONS) --------
    graph_data = {}

    for crop, ref_index in CROP_REFERENCE_INDEX.items():
        graph_data[crop] = round(
            (ref_index / base_index) * estimated_tons,
            2
        )

    # Sort graph data (descending)
    graph_data = dict(
        sorted(graph_data.items(), key=lambda x: x[1], reverse=True)
    )

    # -------- AI CROP RECOMMENDATION (TOP 5) --------
    sorted_crops = list(graph_data.items())
    max_yield = sorted_crops[0][1]

    recommendations = []
    for rank, (crop, value) in enumerate(sorted_crops[:5], start=1):
        ratio = value / max_yield

        if ratio >= 0.8:
            trend = "High"
        elif ratio >= 0.5:
            trend = "Medium"
        else:
            trend = "Low"

        recommendations.append({
            "rank": rank,
            "crop": crop.capitalize(),
            "yield": value,
            "trend": trend
        })

    # -------- AI TEXT --------
    if estimated_tons >= avg_tons:
        ai_text = (
            f"Estimated yield is good for {target_crop.capitalize()}. "
            f"Current conditions are favorable 🌱"
        )
    elif estimated_tons >= 0.8 * avg_tons:
        ai_text = (
            f"Moderate estimated yield for {target_crop.capitalize()}. "
            f"Soil or irrigation improvement recommended ⚠"
        )
    else:
        ai_text = (
            f"Low estimated yield for {target_crop.capitalize()}. "
            f"Consider alternative crops or soil improvement 🚨"
        )

    return jsonify({
        "estimated_tons": estimated_tons,
        "graph_data": graph_data,
        "recommendations": recommendations,
        "ai_text": ai_text
    })

if __name__ == "__main__":
    app.run(debug=True)
