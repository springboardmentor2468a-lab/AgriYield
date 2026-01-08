import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load correct dataset
df = pd.read_csv("Crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "rainfall", "ph"]

# ---- DEMO YIELD CREATION ----
# (Hackathon / academic purpose)
df["yield"] = (
    df["N"] * 0.3 +
    df["P"] * 0.2 +
    df["K"] * 0.2 +
    df["rainfall"] * 0.1 +
    df["temperature"] * 0.2
)

X = df[FEATURES]
y = df["yield"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "yield_model_user_based.pkl")
joblib.dump(scaler, "yield_scaler_user_based.pkl")

print("✅ YIELD MODEL TRAINED SUCCESSFULLY")
print("Features used:", model.n_features_in_)
