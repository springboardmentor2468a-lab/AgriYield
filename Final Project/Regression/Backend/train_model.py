print("=== TRAINING STARTED ===")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pickle

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Clean_dataset_15 (1).csv")
df = df.drop_duplicates().dropna()

num_cols = ["N","P","K","temperature","humidity","ph","rainfall","Value"]
df[num_cols] = df[num_cols].apply(pd.to_numeric)

# ======================
# FEATURE ENGINEERING
# ======================
df["NPK_sum"] = df["N"] + df["P"] + df["K"]
df["NP_ratio"] = df["N"] / (df["P"] + 1)
df["NK_ratio"] = df["N"] / (df["K"] + 1)
df["PK_ratio"] = df["P"] / (df["K"] + 1)
df["temp_humidity"] = df["temperature"] * df["humidity"]
df["temp_ph"] = df["temperature"] * df["ph"]
df["rainfall_ph"] = df["rainfall"] * df["ph"]

# ======================
# ONE HOT ENCODE CROP
# ======================
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
crop_encoded = ohe.fit_transform(df[["crop"]])

crop_df = pd.DataFrame(
    crop_encoded,
    columns=ohe.get_feature_names_out(["crop"])
)

df = pd.concat([df.reset_index(drop=True), crop_df], axis=1)
df.drop(columns=["crop"], inplace=True)

# ======================
# FEATURES / TARGET
# ======================
X = df.drop(columns=["Value"])
y = df["Value"]

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ======================
# MODEL
# ======================
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================
print("Train R²:", r2_score(y_train, model.predict(X_train)))
print("Test  R²:", r2_score(y_test, model.predict(X_test)))

# ======================
# SAVE
# ======================
pickle.dump(model, open("yield_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(ohe, open("crop_ohe.pkl", "wb"))

print("=== TRAINING COMPLETED ===")
