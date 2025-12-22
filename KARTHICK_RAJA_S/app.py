from flask import Flask, jsonify, request
import pickle
import os
import numpy as np

BASE_DIR = os.getcwd()

model_path = os.path.join(BASE_DIR, "model.pkl")
encoder_path = os.path.join(BASE_DIR, "encoder.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

with open(model_path, "rb") as m:
    model = pickle.load(m)
with open(encoder_path, "rb") as e:
    encoder = pickle.load(e)
with open(scaler_path, "rb") as s:
    scaler = pickle.load(s)
print("Base directory:", BASE_DIR)

app = Flask(__name__)

crops=['banana', 'chickpea', 'coconut', 'coffee', 'cotton', 'jute',
       'lentil', 'maize', 'mango', 'mothbeans', 'muskmelon', 'orange',
       'papaya', 'pigeonpeas', 'watermelon']

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        N = data.get("N")
        P = data.get("P")
        K = data.get("K")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        ph = data.get("ph")
        rainfall = data.get("rainfall")
        year = data.get("Year")
        input_crop = data.get("Crops")
        required_fields = [N, P, K, temperature, humidity, ph, rainfall, year, input_crop]
        if any(v is None for v in required_fields):
            return jsonify({"error": "All fields are required"}), 400
        other_results={}
        for crop in crops:
            encoded_crop = encoder.transform([crop])[0]
            features_all = np.array([
                N, P, K, temperature, humidity, ph, rainfall, year, encoded_crop
            ]).reshape(1, -1)

            scaled_all = scaler.transform(features_all)
            yield_value = model.predict(scaled_all)[0]

            other_results[crop] = float(yield_value)

        return jsonify({
            "predicted_yield": float(other_results.get(input_crop)),
            "yield_for_all_crops": other_results
        }),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
