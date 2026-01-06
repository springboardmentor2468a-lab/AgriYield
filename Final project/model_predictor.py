import joblib
import numpy as np


class YieldPredictor:
    def __init__(self):
        self.model = joblib.load("models/yield_model.pkl")
        self.crop_encoder = joblib.load("models/yield_crop_encoder.pkl")

    def build_features(self, N, P, K, temp, humidity, ph, rainfall, crop):
        crop_encoded = int(self.crop_encoder.transform([crop])[0])
        return np.array([[N, P, K, temp, humidity, ph, rainfall, crop_encoded]])

    def predict(self, N, P, K, temp, humidity, ph, rainfall, crop):
        X = self.build_features(N, P, K, temp, humidity, ph, rainfall, crop)
        return round(float(self.model.predict(X)[0]), 2)
