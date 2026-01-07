from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

from flask import Flask, request, jsonify, render_template
import os

app = Flask(
    __name__,
    template_folder="../Frontend/templates",
    static_folder="../Frontend/static"
)


# Load saved production model (NOT retraining)
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/predictor", methods=["GET"])
def predictor_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    user_input = {
        "N": float(data["N"]),
        "P": float(data["P"]),
        "K": float(data["K"]),
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "ph": float(data["ph"]),
        "rainfall": float(data["rainfall"]),
    }

    requested_crop = data.get("requested_crop", "").lower().strip()

    df_input = pd.DataFrame([user_input])
    scaled = scaler.transform(df_input)

    probs = model.predict_proba(scaled)[0]

    # Create crop→probability list
    all_results = [
        (label_encoder.classes_[i], round(probs[i] * 100, 2))
        for i in range(len(probs))
    ]

    # Sort desc (top first)
    all_results.sort(key=lambda x: x[1], reverse=True)

    # Response format (your JS expects)
    response = {
        "requested_crop": requested_crop,
        "results": [{"crop": c, "probability": p} for c, p in all_results],
        "top_crop": all_results[0][0],
        "top_prob": all_results[0][1],
    }

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
