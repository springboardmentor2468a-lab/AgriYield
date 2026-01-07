from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000/predict_yield"

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    top5 = None
    error = None

    if request.method == "POST":
        try:
            payload = {
                "crop": request.form["crop"],
                "N": float(request.form["N"]),
                "P": float(request.form["P"]),
                "K": float(request.form["K"]),
                "temperature": float(request.form["temperature"]),
                "humidity": float(request.form["humidity"]),
                "ph": float(request.form["ph"]),
                "rainfall": float(request.form["rainfall"])
            }

            res = requests.post(API_URL, json=payload)

            if res.status_code == 200:
                data = res.json()
                result = data["user_crop_yield"]
                top5 = data["top_recommendations"]
            else:
                error = res.text
        except Exception as e:
            error = str(e)

    return render_template("predictor.html", result=result, top5=top5, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
