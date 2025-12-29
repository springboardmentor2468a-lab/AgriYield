import pandas as pd
import joblib
import numpy as np
from typing import List, Dict

# Filenames exactly as in your folder (see screenshot)
MODELPATH = "crop_production_model_with_crop.pkl"
DATAPATH = "finalagridatasetclean.csv"
LE_PATH = "crop_label_encoder.pkl"
FEATURE_PATH = "feature_names_with_crop.pkl"


class CropYieldPredictor:
    def __init__(
        self,
        modelpath: str = MODELPATH,
        datapath: str = DATAPATH,
        le_path: str = LE_PATH,
        feature_path: str = FEATURE_PATH,
    ):
        print("Loading model...")
        self.model = joblib.load(modelpath)

        print("Loading dataset...")
        self.data = pd.read_csv(datapath)

        print("Loading label encoder...")
        self.le = joblib.load(le_path)

        print("Loading feature names...")
        self.feature_names = joblib.load(feature_path)

        self.crops = sorted(self.data["crop"].unique())
        print(f"Loaded {len(self.crops)} crops: {self.crops}")

    def predict_single(
        self,
        N: float,
        P: float,
        K: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        crop: str,
    ) -> Dict:
        """Predict yield for one given crop."""
        if crop not in self.crops:
            return {"error": f"Crop {crop} not in available crops."}

        crop_encoded = self.le.transform([crop])[0]

        # 2D array with shape (1, 8) – what RandomForestRegressor expects
        X = np.array([[N, P, K, temperature, humidity, ph, rainfall, crop_encoded]])

        ypred = float(self.model.predict(X)[0])
        return {"crop": crop, "yield": round(ypred, 2)}

    def predict_all_crops(
        self,
        N: float,
        P: float,
        K: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
    ) -> List[Dict]:
        """Predict yield for all crops and return top 5."""
        preds: List[Dict] = []

        for c in self.crops:
            crop_encoded = self.le.transform([c])[0]
            X = np.array([[N, P, K, temperature, humidity, ph, rainfall, crop_encoded]])
            ypred = float(self.model.predict(X)[0])
            preds.append({"crop": c, "yield": round(ypred, 2)})

        preds.sort(key=lambda d: d["yield"], reverse=True)
        return preds[:5]
