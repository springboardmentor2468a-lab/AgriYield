from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import io
import base64
from matplotlib.figure import Figure
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import json

app = Flask(__name__)

def load_safe_model(path, loader_type="joblib"):
    try:
        if loader_type == "joblib":
            return joblib.load(path)
        elif loader_type == "keras":
            return tf.keras.models.load_model(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


DISEASE_INFO = {}
try:
    with open('diseases.json', 'r') as f:
        DISEASE_INFO = json.load(f)
except Exception as e:
    print(f"Warning: diseases.json not found or corrupted. {e}")


model = load_safe_model("models/random_forest_crop_yield_model.pkl")
scaler = load_safe_model("models/scaler.pkl")
disease_model = load_safe_model('models/Trained_model.keras', loader_type="keras")

disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

try:
    trained_columns = joblib.load("models/rf_trained_columns.pkl")
except:
    trained_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                       'Crop_banana', 'Crop_chickpea', 'Crop_coconut', 'Crop_coffee', 
                       'Crop_cotton', 'Crop_jute', 'Crop_lentil', 'Crop_maize', 
                       'Crop_mango', 'Crop_mothbeans', 'Crop_muskmelon', 'Crop_orange', 
                       'Crop_papaya', 'Crop_pigeonpeas', 'Crop_watermelon']

CROP_LIST = [col.replace("Crop_", "") for col in trained_columns if col.startswith("Crop_")]


UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/yield", methods=["GET", "POST"])
def yield_prediction():
    if model is None or scaler is None:
        return "Model files are missing on server. Check logs.", 500
    
    prediction = None
    top_5_crops = []
    plot_url = None

    if request.method == "POST":
        try:
            n = float(request.form["N"])
            p = float(request.form["P"])
            k = float(request.form["K"])
            temp = float(request.form["temperature"])
            hum = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rain = float(request.form["rainfall"])
            selected_crop = request.form["crop"].lower()

            crop_predictions = {}
            num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

            for crop in CROP_LIST:
                row = pd.DataFrame(0, index=[0], columns=trained_columns)
                row["N"], row["P"], row["K"] = n, p, k
                row["temperature"], row["humidity"] = temp, hum
                row["ph"], row["rainfall"] = ph, rain
                
                if f"Crop_{crop}" in trained_columns:
                    row[f"Crop_{crop}"] = 1
                
                row[num_cols] = scaler.transform(row[num_cols])
                log_pred = model.predict(row)[0]
                real_production = np.expm1(log_pred) 
                crop_predictions[crop] = round(float(real_production), 2)

            top_5_crops = sorted(crop_predictions.items(), key=lambda x: x[1], reverse=True)[:5]
            prediction = crop_predictions.get(selected_crop)

            if top_5_crops:
                crop_names = [c[0].capitalize() for c in top_5_crops]
                productions = [c[1] for c in top_5_crops]
                fig = Figure(figsize=(8, 5))
                ax = fig.subplots()
                bars = ax.bar(crop_names, productions, color=['#2ecc71', '#27ae60', '#16a085', '#1abc9c', '#34495e'])
                ax.ticklabel_format(style='plain', axis='y')   # scientific notation off
                ax.get_yaxis().set_major_formatter(lambda x, _: f'{int(x):,}')
                ax.set_ylabel("Estimated Production (tons)")
                ax.set_title("Top 5 Crops – Total Estimated Production (tons)")
                
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:,.0f}",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
                
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png")
                buf.seek(0)
                plot_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            print(f"Prediction Error: {e}")
            prediction = "Error in calculation"

    return render_template("yield.html", prediction=prediction, top_5_crops=top_5_crops, plot_url=plot_url)

@app.route("/disease", methods=["GET", "POST"])
def disease():
    if disease_model is None:
        return "Disease detection model not loaded.", 500
        
    prediction = None
    info = None
    image_path = None
    
    if request.method == "POST":
        try:
            file = request.files['image']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                image_path = filepath.replace('\\', '/')

                from tensorflow.keras.preprocessing.image import load_img, img_to_array
                image = load_img(filepath, target_size=(128, 128))
                input_arr = img_to_array(image)
                input_arr = np.array([input_arr], dtype=np.float32)

                preds = disease_model.predict(input_arr)
                result_index = np.argmax(preds)
                prediction = disease_classes[result_index]
                info = DISEASE_INFO.get(prediction, "No information available for this disease.")
        except Exception as e:
            print(f"Disease Detection Error: {e}")
            prediction = "Error processing image"

    return render_template("disease.html", prediction=prediction, image_path=image_path, info=info)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)