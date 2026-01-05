from flask import Flask,jsonify,request,render_template
import pickle
import os
import numpy as np

BASE_DIR=os.getcwd()

model_path=os.path.join(BASE_DIR,"models","model.pkl")
encoder_path=os.path.join(BASE_DIR,"models","encoder.pkl")
scaler_path=os.path.join(BASE_DIR,"models","scaler.pkl")

with open(model_path, "rb") as m:
    model = pickle.load(m)
with open(encoder_path, "rb") as e:
    encoder = pickle.load(e)
with open(scaler_path, "rb") as s:
    scaler = pickle.load(s)

app=Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    try:
        data=request.get_json()
        N = data.get("N")
        P = data.get("P")
        K = data.get("K")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        ph = data.get("ph")
        rainfall = data.get("rainfall")
        required_colums=[N,P,K,temperature,humidity,ph,rainfall]
        if any(col is None for col in required_colums):
            return jsonify({"error":"All fields are required"}),400
        features=np.array(required_colums).reshape(1,-1)
        scaled_feature=scaler.transform(features)
        result=encoder.inverse_transform(model.predict(scaled_feature))
        return jsonify({"result":result[0]}),200
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(debug=True)

