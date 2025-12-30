from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/xgb_model_final.joblib")
feature_columns = joblib.load("models/feature_columns_final.pkl")

crops = sorted([c.replace("Crop_", "") for c in feature_columns if c.startswith("Crop_")])

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html", crops=crops)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    base = {
        "N": float(data["N"]),
        "P": float(data["P"]),
        "K": float(data["K"]),
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
        "rainfall": float(data["rainfall"]),
        "ph": float(data["ph"]),
    }

    crop = data["crop"]

    def make_df(c):
        r = base.copy()
        for col in feature_columns:
            if col.startswith("Crop_"):
                r[col] = 1 if col == f"Crop_{c}" else 0
        return pd.DataFrame([r]).reindex(columns=feature_columns, fill_value=0)

    # Single prediction (tons → kilo-tonnes)
    pred_tons = float(model.predict(make_df(crop))[0])
    pred_kt = round(pred_tons / 1000, 2)

    # Top 5 crops
    results = []
    for c in crops:
        y = float(model.predict(make_df(c))[0])
        results.append((c, round(y / 1000, 2)))

    top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        "predicted_yield": pred_kt,
        "top5": top5
    })

if __name__ == "__main__":
    app.run(debug=True)
