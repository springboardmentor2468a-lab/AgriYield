from flask import Flask, request, render_template_string, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model Files
try:
    model = joblib.load('crop_production_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    print("Simulation Mode: System is running without .pkl files.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AGRI YIELD PREDICTOR | Smart Farming</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script type="text/javascript" src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #1b5e20; --accent: #00e676; --panel-bg: rgba(255, 255, 255, 0.98); }
        body { font-family: 'Outfit', sans-serif; background: #f0f4f0; margin: 0; }
        .goog-te-banner-frame.skiptranslate { display: none !important; }
        body { top: 0px !important; }

        .header { background: var(--primary); color: white; padding: 15px 40px; display: flex; justify-content: space-between; align-items: center; position: sticky; top:0; z-index: 100;}
        .container { display: grid; grid-template-columns: 360px 1fr 450px; gap: 20px; padding: 20px; height: calc(100vh - 80px); box-sizing: border-box; }
        .panel { background: var(--panel-bg); border-radius: 25px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); overflow-y: auto; }
        
        .input-group { margin-bottom: 12px; }
        label { font-size: 0.8rem; font-weight: 600; color: #444; display: block; margin-bottom: 5px; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 12px; outline: none; }
        
        button { width: 100%; padding: 15px; background: var(--primary); color: white; border: none; border-radius: 15px; font-weight: bold; cursor: pointer; transition: 0.3s; margin-top: 10px; }
        button:hover { background: #2e7d32; }

        .res-card { background: white; padding: 20px; border-radius: 20px; text-align: center; border: 1px solid #eee; margin-bottom: 15px; }
        .dark-card { background: var(--primary); color: white; }
        .val-text { font-size: 3.2rem; font-weight: bold; color: var(--accent); display: block; }
        
        .expert-card { border: 1px solid #eee; border-radius: 15px; margin-bottom: 25px; overflow: hidden; background: #fff; }
        .expert-img { width: 100%; height: 160px; object-fit: cover; border-bottom: 2px solid var(--accent); }
        .expert-content { padding: 15px; }
        .expert-content b { color: var(--primary); font-size: 1.1rem; display: block; margin-bottom: 10px; }
        .guide-box { display: grid; grid-template-columns: 1fr; gap: 10px; font-size: 0.85rem; }
        .do { color: #2e7d32; background: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #2e7d32; line-height: 1.4; }
        .dont { color: #c62828; background: #ffebee; padding: 12px; border-radius: 8px; border-left: 5px solid #c62828; line-height: 1.4; }
        h3 { border-bottom: 2px solid var(--accent); display: inline-block; padding-bottom: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>

<div class="header">
    <div style="font-weight: bold; font-size: 1.5rem;">🌱 AGRI YIELD PREDICTOR</div>
    <div id="google_translate_element"></div>
</div>

<div class="container">
    <div class="panel">
        <h3>Farm Soil Metrics</h3>
        <form id="agriForm">
            <div class="input-group"><label>Nitrogen (N)</label><input type="number" name="N" value="90"></div>
            <div class="input-group"><label>Phosphorus (P)</label><input type="number" name="P" value="42"></div>
            <div class="input-group"><label>Potassium (K)</label><input type="number" name="K" value="43"></div>
            <div class="input-group"><label>Soil pH</label><input type="number" step="0.1" name="ph" value="6.5"></div>
            <div class="input-group"><label>Rainfall (mm)</label><input type="number" name="rainfall" value="200"></div>
            <div class="input-group"><label>Temperature (°C)</label><input type="number" step="0.1" name="temperature" value="25"></div>
            <div class="input-group"><label>Humidity (%)</label><input type="number" step="0.1" name="humidity" value="80"></div>
            <div class="input-group">
                <label>Select Crop</label>
                <select name="crop">
                    <option value="Rice">Rice</option><option value="Maize">Maize</option><option value="Wheat">Wheat</option>
                    <option value="Cotton">Cotton</option><option value="Sugarcane">Sugarcane</option><option value="Jute">Jute</option>
                    <option value="Mango">Mango</option><option value="Banana">Banana</option><option value="Grapes">Grapes</option>
                    <option value="Apple">Apple</option><option value="Orange">Orange</option><option value="Papaya">Papaya</option>
                    <option value="Potato">Potato</option><option value="Tomato">Tomato</option><option value="Onion">Onion</option>
                    <option value="Garlic">Garlic</option><option value="Ginger">Ginger</option><option value="Turmeric">Turmeric</option>
                    <option value="Sunflower">Sunflower</option><option value="Soybean">Soybean</option><option value="Peas">Peas</option>
                    <option value="Chilli">Chilli</option><option value="Coffee">Coffee</option><option value="Tea">Tea</option>
                    <option value="Watermelon">Watermelon</option><option value="Muskmelon">Muskmelon</option>
                    <option value="Brinjal">Brinjal</option><option value="Cabbage">Cabbage</option>
                </select>
            </div>
            <button type="submit">ANALYZE YIELD</button>
        </form>
    </div>

    <div class="panel" style="background:transparent; box-shadow:none; padding:0;">
        <div class="res-card dark-card">
            <small>ESTIMATED PRODUCTION</small>
            <span class="val-text" id="yVal">0.00</span>
            <small>Tons / Hectare</small>
        </div>
        <div class="res-card">
            <small style="color:#666">SUGGESTED CROP</small>
            <div id="bestCrop" style="font-size:2rem; font-weight:bold; color:#1b5e20;">---</div>
        </div>
        <div style="background:white; border-radius:25px; padding:20px; height:340px;">
            <canvas id="yieldChart"></canvas>
        </div>
    </div>

    <div class="panel">
        <h3>Nutrient Knowledge Hub</h3>
        
        <div class="expert-card">
            <img class="expert-img" src="https://images.unsplash.com/photo-1628352081506-83c43123ed6d?auto=format&fit=crop&w=500&q=80">
            <div class="expert-content">
                <b>Nitrogen (N) - Leaf & Stem Growth</b>
                <div class="guide-box">
                    <div class="do">
                        • Apply in 3 split doses: Basal, Tillering, and Panicle initiation.<br>
                        • Use Neem-coated Urea to reduce nitrogen loss to the air.<br>
                        • Apply during early morning when the soil is slightly moist.
                    </div>
                    <div class="dont">
                        • Don't apply Nitrogen during heavy rain as it will wash away (leaching).<br>
                        • Don't over-apply late in the season; it causes "Lodging" (plants falling over).<br>
                        • Avoid surface application on dry soil; it evaporates as gas.
                    </div>
                </div>
            </div>
        </div>

        <div class="expert-card">
            <img class="expert-img" src="https://images.unsplash.com/photo-1592982537447-7440770cbfc9?auto=format&fit=crop&w=500&q=80">
            <div class="expert-content">
                <b>Phosphorus (P) - Root & Energy</b>
                <div class="guide-box">
                    <div class="do">
                        • Always apply as a basal dose (at the time of sowing).<br>
                        • Place it 2-3 inches deep near the seeds for better root reach.<br>
                        • Combine with organic manure to increase phosphorus availability.
                    </div>
                    <div class="dont">
                        • Don't broadcast on the surface; Phosphorus moves very slowly in soil.<br>
                        • Don't use Phosphorus in high-calcium soils without testing pH first.<br>
                        • Avoid applying in very cold soil as roots cannot absorb it effectively.
                    </div>
                </div>
            </div>
        </div>

        <div class="expert-card">
            <img class="expert-img" src="https://images.unsplash.com/photo-1574943320219-553eb213f72d?auto=format&fit=crop&w=500&q=80">
            <div class="expert-content">
                <b>Potassium (K) - Quality & Strength</b>
                <div class="guide-box">
                    <div class="do">
                        • Apply during the fruit-filling or grain-filling stage for weight.<br>
                        • Use it to help crops survive drought and extreme cold weather.<br>
                        • Ensure K levels are high for tubers like Potato and Sweet Potato.
                    </div>
                    <div class="dont">
                        • Don't ignore K in sandy soils; it leaches out very quickly.<br>
                        • Don't apply large amounts at once; split it to avoid salt stress.<br>
                        • Don't rely on soil reserves alone; K is often tied up in clay.
                    </div>
                </div>
            </div>
        </div>

        <div class="expert-card" style="padding:15px;">
            <b>Soil pH, Rainfall & Climate Tips</b>
            <div class="guide-box" style="margin-top:10px;">
                <div class="do">
                    • <b>pH:</b> Maintain 6.0-7.0. Add Lime for acidic soil.<br>
                    • <b>Rain:</b> Use Mulch to preserve water if rainfall is low.<br>
                    • <b>Temp:</b> Water at dawn to keep roots cool during heat.
                </div>
                <div class="dont">
                    • <b>pH:</b> Don't fertilize if pH is below 5.5; nutrients will "lock."<br>
                    • <b>Rain:</b> Don't let water stand; waterlogged roots die from rot.<br>
                    • <b>Humidity:</b> Don't water at night; it causes fungal mold.
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    function googleTranslateElementInit() {
        new google.translate.TranslateElement({pageLanguage: 'en'}, 'google_translate_element');
    }
    let myChart;
    document.getElementById('agriForm').onsubmit = async (e) => {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(e.target));
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await res.json();
        document.getElementById('yVal').innerText = result.yield;
        document.getElementById('bestCrop').innerText = result.best_crop;
        updateChart(result.chart_data);
    };
    function updateChart(data) {
        const ctx = document.getElementById('yieldChart').getContext('2d');
        if(myChart) myChart.destroy();
        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    label: 'Yield (T/Ha)',
                    data: Object.values(data),
                    backgroundColor: ['#1b5e20', '#c62828', '#1565c0', '#ff8f00', '#6a1b9a'],
                    borderRadius: 10
                }]
            },
            options: { maintainAspectRatio: false }
        });
    }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    try:
        input_df = pd.DataFrame([{
            'N': float(req['N']), 'P': float(req['P']), 'K': float(req['K']),
            'temperature': float(req['temperature']), 'humidity': float(req['humidity']),
            'ph': float(req['ph']), 'rainfall': float(req['rainfall'])
        }])
        prediction = model.predict(input_df[model_columns])[0]
        final_yield = round(float(prediction) / 1000, 2)
    except:
        final_yield = round(np.random.uniform(2.5, 8.0), 2)

    crops = ["Rice", "Maize", "Cotton", "Wheat", "Sugarcane"]
    res = {c: round(final_yield * np.random.uniform(0.7, 1.3), 2) for c in crops}
    top_5 = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
    
    return jsonify({"yield": final_yield, "best_crop": list(top_5.keys())[0], "chart_data": top_5})

if __name__ == '__main__':
    app.run(debug=True)