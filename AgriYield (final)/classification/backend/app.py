from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and encoder
model = joblib.load("crop_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = []

    if request.method == "POST":
        try:
            # Collect inputs from form
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Prepare input for model
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Get probabilities (required for top-3)
            probabilities = model.predict_proba(input_data)[0]

            # Get indices of top 3 crops
            top3_indices = np.argsort(probabilities)[-3:][::-1]

            # Prepare result list
            for idx in top3_indices:
                crop_name = label_encoder.inverse_transform([idx])[0].capitalize()
                confidence = round(probabilities[idx] * 100, 2)

                predictions.append({
                    "crop": crop_name,
                    "confidence": confidence
                })

        except Exception as e:
            print("Prediction error:", e)

    return render_template("index.html", predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
