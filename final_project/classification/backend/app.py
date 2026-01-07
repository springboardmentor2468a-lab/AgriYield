from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load pre-trained models
rf_model = joblib.load("models/crop_final_1.pkl")
scaler = joblib.load("models/scaler_final_1.pkl")
label_encoder = joblib.load("models/encoder_final_1.pkl")

# Flask app
app = Flask(__name__)

# ---- ROUTES ----

# Home page
@app.route("/")
def home():
    return render_template("home.html")  # Fixed: use home.html instead of base.html

# About page
@app.route("/about")
def about():
    return render_template("about2.html")  # or "about.html" if that's the file you have

# Predict/classify page
@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

# API for classification
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    
    # Predict crop
    pred_encoded = rf_model.predict(df_scaled)[0]
    pred_crop = label_encoder.inverse_transform([pred_encoded])[0]

    # Top 3 crops
    probs = rf_model.predict_proba(df_scaled)[0]
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3 = [[label_encoder.inverse_transform([i])[0], round(probs[i]*100,2)] for i in top3_indices]

    return jsonify({"predicted_crop": pred_crop, "top3": top3})

if __name__ == "__main__":
    app.run(debug=True)
