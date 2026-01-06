from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("yield_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("crop_ohe.pkl")

crops_list = ohe.get_feature_names_out(["crop"])
crops_list = [c.replace("crop_", "") for c in crops_list]

numeric_features = ['N','P','K','temperature','humidity','ph','rainfall']

@app.route('/')
def home():
    return render_template('welcome.html', crops=crops_list)

@app.route('/predict', methods=['GET','POST'])
def predict():

    if request.method == 'GET':
        return render_template('input_form.html', crops=crops_list)

    data = request.form.to_dict()

    for f in numeric_features:
        data[f] = float(data[f])

    selected_crop = data["Crop"]

    base = {
        "N": data["N"],
        "P": data["P"],
        "K": data["K"],
        "temperature": data["temperature"],
        "humidity": data["humidity"],
        "ph": data["ph"],
        "rainfall": data["rainfall"],
        "NPK_sum": data["N"] + data["P"] + data["K"],
        "NP_ratio": data["N"] / (data["P"] + 1),
        "NK_ratio": data["N"] / (data["K"] + 1),
        "PK_ratio": data["P"] / (data["K"] + 1),
        "temp_humidity": data["temperature"] * data["humidity"],
        "temp_ph": data["temperature"] * data["ph"],
        "rainfall_ph": data["rainfall"] * data["ph"]
    }

    all_preds = {}

    for crop in crops_list:
        df = pd.DataFrame([base])
        crop_ohe = ohe.transform([[crop]])
        crop_df = pd.DataFrame(
            crop_ohe,
            columns=ohe.get_feature_names_out(["crop"])
        )

        final_df = pd.concat([df, crop_df], axis=1)
        scaled = scaler.transform(final_df)
        pred = model.predict(scaled)[0] / 1_000_000

        all_preds[crop] = round(float(pred), 3)

    prediction = all_preds[selected_crop]
    best_crop = max(all_preds, key=all_preds.get)
    selected_is_best = (best_crop == selected_crop)

    return render_template(
        "result.html",
        prediction=prediction,
        all_preds=all_preds,
        best_crop=best_crop,
        selected_crop=selected_crop,
        selected_is_best=selected_is_best
    )

if __name__ == "__main__":
    app.run(debug=True)
