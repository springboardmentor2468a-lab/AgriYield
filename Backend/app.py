from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
from flask import Flask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static"
)


# Global model + encoder
clf = None
le = None
CROP15 = ['banana','chickpea','coconut','coffee','cotton','jute','lentil','maize',
          'mango','mothbeans','muskmelon','orange','papaya','pigeonpeas','rice','watermelon']

def load_model():
    """Week-4 exact model training on startup"""
    global clf, le
    
    print("🚀 Loading Week-4 Crop Recommendation Model...")
    crop_df = pd.read_csv('Crop_recommendation.csv')
    X = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = crop_df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 80/20 stratified split (Week-4 style)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # RandomForest (Week-4 model)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"✅ Model loaded! Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"📊 CROP15 crops ready: {len(CROP15)}")

# Load on startup
load_model()

# ---------- ROUTES (UNCHANGED UI COMPATIBLE) ----------

@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")

@app.route("/predictor", methods=["GET"])
def predictor_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # EXACT Week-4 input format
    user_input = {
        'N': float(data["N"]),
        'P': float(data["P"]),
        'K': float(data["K"]),
        'temperature': float(data["temperature"]),
        'humidity': float(data["humidity"]),
        'ph': float(data["ph"]),
        'rainfall': float(data["rainfall"])
    }
    
    requested_crop = data.get("requested_crop", "coffee").lower()
    
    # Week-4 EXACT prediction
    input_df = pd.DataFrame([user_input])
    probs = clf.predict_proba(input_df)[0]
    
    # Filter CROP15 only (screenshot crops)
    crop_indices = [i for i, crop in enumerate(le.classes_) if crop in CROP15]
    crop_probs = [(le.classes_[i], round(probs[i] * 100, 2)) for i in crop_indices]
    
    # Sort DESC (coffee 92.67% top for test input)
    results = sorted(crop_probs, key=lambda x: x[1], reverse=True)
    
    # Frontend compatible response (your JS expects this)
    response = {
        "requested_crop": requested_crop,
        "results": [{"crop": crop, "probability": prob} for crop, prob in results],
        "top_crop": results[0][0],
        "top_prob": results[0][1]
    }
    
    return jsonify(response)

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

