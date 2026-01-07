from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------------------------------------------
# Load model & feature columns
# ---------------------------------------------------
model = joblib.load("models/model_final_1.joblib")
feature_columns = joblib.load("models/columns_final_1.pkl")

# Extract crop names from one-hot columns
crops = sorted([c.replace("Crop_", "") for c in feature_columns if c.startswith("Crop_")])

# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html", crops=crops)

# ---------------------------------------------------
# Prediction API (KG OUTPUT)
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Base numerical features (same order as training)
    base_features = {
        "N": float(data["N"]),
        "P": float(data["P"]),
        "K": float(data["K"]),
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "ph": float(data["ph"]),
        "rainfall": float(data["rainfall"]),
    }

    selected_crop = data["crop"]

    # Build dataframe exactly like training data
    def build_input_df(crop):
        row = base_features.copy()

        # One-hot encode crops
        for col in feature_columns:
            if col.startswith("Crop_"):
                row[col] = 1 if col == f"Crop_{crop}" else 0

        return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)

    # ---------------------------------------------------
    # Prediction for selected crop (KG)
    # ---------------------------------------------------
    predicted_kg = round(float(model.predict(build_input_df(selected_crop))[0]), 2)

    # ---------------------------------------------------
    # Top 5 crops (KG)
    # ---------------------------------------------------
    top5_results = []
    for crop in crops:
        pred = float(model.predict(build_input_df(crop))[0])
        top5_results.append((crop, round(pred, 2)))

    top5 = sorted(top5_results, key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        "predicted_yield_kg": predicted_kg,
        "unit": "kg",
        "top5_kg": top5
    })

# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)