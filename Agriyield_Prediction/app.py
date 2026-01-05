from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import io, base64
from matplotlib.figure import Figure
import os

app = Flask(__name__)

# ================= SAFE MODEL LOADING =================
try:
    model = joblib.load("models/random_forest_crop_yield_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    trained_columns = joblib.load("models/rf_trained_columns.pkl")

    crop_model = joblib.load("models/crop_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    print("✅ All models loaded successfully")

except Exception as e:
    print("❌ Model loading error:", e)
    model = scaler = trained_columns = crop_model = label_encoder = None

# ================= CONSTANTS =================
NUM_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

CROP_LIST = []
if trained_columns is not None:
    CROP_LIST = [
        c.replace("Crop_", "").lower()
        for c in trained_columns
        if c.startswith("Crop_")
    ]

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

# =====================================================
# ================= YIELD PREDICTION ==================
# =====================================================
@app.route("/yield", methods=["GET", "POST"])
def yield_prediction():

    prediction = None
    top_5_crops = []
    plot_url = None
    error = None

    if request.method == "POST":

        try:
            # ---------- INPUT ----------
            n = float(request.form["N"])
            p = float(request.form["P"])
            k = float(request.form["K"])
            temp = float(request.form["temperature"])
            hum = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rain = float(request.form["rainfall"])
            selected_crop = request.form["crop"].lower()

            crop_predictions = {}

            for crop in CROP_LIST:

                # EMPTY INPUT ROW
                row = pd.DataFrame(0, index=[0], columns=trained_columns)

                # NUMERIC INPUT
                row.loc[0, NUM_COLS] = [n, p, k, temp, hum, ph, rain]

                # SCALE NUMERIC VALUES
                row[NUM_COLS] = scaler.transform(row[NUM_COLS])

                # ONE-HOT ENCODE CROP
                row.loc[0, f"Crop_{crop}"] = 1

                # LOG PREDICTION
                log_pred = model.predict(row)[0]

                # INVERSE LOG
                production = np.expm1(log_pred)

                # OPTIONAL UNIT CONVERSION (remove if not required)
                production = max(production, 0)

                crop_predictions[crop] = round(production, 2)

            # ---------- TOP 5 CROPS ----------
            top_5_crops = sorted(
                crop_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            prediction = crop_predictions.get(selected_crop)

            # ---------- GRAPH ----------
            if top_5_crops:
                names = [c.capitalize() for c, _ in top_5_crops]
                values = [v for _, v in top_5_crops]

                fig = Figure(figsize=(8, 5))
                ax = fig.subplots()
                bars = ax.bar(names, values)
                ax.set_title("Top 5 Crop Yield Prediction")
                ax.set_ylabel("Estimated Yield")
                ax.bar_label(bars)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                plot_url = "data:image/png;base64," + base64.b64encode(
                    buf.getvalue()
                ).decode()

        except Exception as e:
            print("❌ Yield Prediction Error:", e)
            error = "Invalid input or prediction error"

    return render_template(
        "yield.html",
        prediction=prediction,
        top_5_crops=top_5_crops,
        plot_url=plot_url,
        error=error
    )

# =====================================================
# ================= CROP CLASSIFICATION ================
# =====================================================
@app.route("/classify", methods=["GET", "POST"])
def classify_crop():

    prediction = None
    top_3 = []
    confidence = None
    error = None

    if request.method == "POST":
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # 🔹 If you used scaler during training
            if scaler is not None:
                input_data = scaler.transform(input_data)

            # 🔹 Probabilities
            probs = crop_model.predict_proba(input_data)[0]

            # 🔹 Top 3 indices
            top_indices = np.argsort(probs)[::-1][:3]

            # 🔹 Decode crop names
            crops = label_encoder.inverse_transform(top_indices)

            top_3 = [
                (crops[i], round(probs[top_indices[i]] * 100, 2))
                for i in range(3)
            ]

            # 🔹 Best crop
            prediction = top_3[0][0]
            confidence = top_3[0][1]

        except Exception as e:
            print("Classification Error:", e)
            error = "Invalid input or prediction error"

    return render_template(
        "classify.html",
        prediction=prediction,
        confidence=confidence,
        top_3=top_3,
        error=error
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run()
