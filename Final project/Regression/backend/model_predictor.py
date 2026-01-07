import joblib
import numpy as np

class CropYieldPredictor:
    def __init__(self):
        print("🔄 Loading AgriSense AI model...")

        self.model = joblib.load("models/crop_yield_tuned_model.pkl")
        self.le = joblib.load("models/crop_yield_label_encoder.pkl")
        self.feature_names = joblib.load("models/yield_feature_names.pkl")

        self.crops = self.le.classes_.tolist()

        # Crop multipliers
        self.crop_yield_multiplier = {
            "apple": 0.95, "banana": 1.10, "chickpea": 0.55,
            "coffee": 0.60, "grapes": 0.90, "jute": 0.50,
            "lentil": 0.52, "maize": 0.85, "mango": 1.00,
            "muskmelon": 0.88, "orange": 0.93, "papaya": 1.05,
            "pigeonpeas": 0.50, "rice": 0.45, "watermelon": 0.48
        }

    def predict_yield(self, crop, N, P, K, temperature, humidity, ph, rainfall):
        crop_encoded = int(self.le.transform([crop])[0])

        X = np.array([[N, P, K, temperature, humidity, ph, rainfall, crop_encoded]])

        base_yield = float(self.model.predict(X)[0])
        multiplier = self.crop_yield_multiplier.get(crop, 1.0)

        return round(base_yield * multiplier, 2)

    def predict_all_crops(self, N, P, K, temperature, humidity, ph, rainfall):
        predictions = []

        for crop in self.crops:
            crop_encoded = int(self.le.transform([crop])[0])
            X = np.array([[N, P, K, temperature, humidity, ph, rainfall, crop_encoded]])

            base_yield = float(self.model.predict(X)[0])
            mult = self.crop_yield_multiplier.get(crop, 1.0)
            final = round(base_yield * mult, 2)

            predictions.append({"crop": crop, "yield": final})

        predictions.sort(key=lambda x: x["yield"], reverse=True)
        return predictions[:5]
