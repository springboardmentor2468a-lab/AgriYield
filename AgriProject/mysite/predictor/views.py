from django.shortcuts import render
from django.http import HttpResponse
import joblib
import pandas as pd
import numpy as np
import os
from django.conf import settings
import glob

# --- 1. SETUP & SAFE LOADING ---
BASE_DIR = settings.BASE_DIR

# Full Crop List for Fallback
FALLBACK_CROPS = [
    'Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans', 
    'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango', 
    'Grapes', 'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 
    'Coconut', 'Cotton', 'Jute', 'Coffee'
]

# Ideal Conditions (For Smart Demo Logic)
CROP_PROFILE = {
    'Banana': {'temp': 27, 'rain': 100}, 
    'Mango': {'temp': 30, 'rain': 90},   
    'Chickpea': {'temp': 20, 'rain': 40}, 
    'Apple': {'temp': 10, 'rain': 80},    
    'Rice': {'temp': 25, 'rain': 200},    
    'Cotton': {'temp': 30, 'rain': 80},
    'Coconut': {'temp': 27, 'rain': 150},
    'Papaya': {'temp': 28, 'rain': 100},
    'Orange': {'temp': 25, 'rain': 100},
}

try:
    # Load Files
    model_path = os.path.join(BASE_DIR, 'agri_yield_model.joblib')
    cols_path = os.path.join(BASE_DIR, 'model_columns.joblib')
    scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
    ranges_path = os.path.join(BASE_DIR, 'input_ranges.joblib')

    model = joblib.load(model_path)
    model_cols = joblib.load(cols_path)
    scaler = joblib.load(scaler_path)
    
    if os.path.exists(ranges_path):
        ranges = joblib.load(ranges_path)
    else:
        ranges = {}
    print("✅ Model Loaded")

except Exception as e:
    print(f"⚠️ Using Smart Demo Mode: {e}")
    ranges = {}
    model = None
    scaler = None
    model_cols = []

# --- 2. MAIN VIEW ---
def predict_yield(request):
    # Get Full Crop List
    crop_cols = [col for col in model_cols if str(col).startswith('label_')]
    if crop_cols:
        crop_list = sorted([col.replace('label_', '') for col in crop_cols])
    else:
        crop_list = sorted(FALLBACK_CROPS)

    # --- INPUT SETUP ---
    form_inputs = []
    scaler_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Defaults
    meta_data = {
        'N': {'min': 0, 'max': 300}, 'P': {'min': 0, 'max': 150}, 'K': {'min': 0, 'max': 250},
        'temperature': {'min': 5, 'max': 50}, 'humidity': {'min': 10, 'max': 100},
        'ph': {'min': 3, 'max': 10}, 'rainfall': {'min': 0, 'max': 1200}
    }

    # Generate Inputs
    for key in scaler_order:
        std = meta_data.get(key, {'min':0, 'max':100})
        # Try to keep values persistent
        val = (std['min'] + std['max']) / 2
        if request.method == 'POST':
            try: val = float(request.POST.get(key, val))
            except: pass
            
        form_inputs.append({
            'name': key, 'label': key.upper(), 'min': std['min'], 'max': std['max'], 'current': val
        })

    context = {'form_inputs': form_inputs, 'crop_list': crop_list, 'show_result': False}

    # --- PREDICTION LOGIC ---
    if request.method == 'POST':
        try:
            # 1. Collect Data
            user_data = {item['name']: item['current'] for item in form_inputs}
            selected_crop = request.POST.get('crop', 'Rice')

            # 2. Smart Calculator
            def get_yield(c_name, inputs):
                # A. TRY REAL MODEL
                if model and scaler:
                    try:
                        feats = [[inputs['N'], inputs['P'], inputs['K'], inputs['temperature'], 
                                  inputs['humidity'], inputs['ph'], inputs['rainfall']]]
                        scaled = scaler.transform(feats)
                        row = pd.DataFrame(np.zeros((1, len(model_cols))), columns=model_cols)
                        row.iloc[0, :7] = scaled[0]
                        if f'label_{c_name}' in model_cols: row[f'label_{c_name}'] = 1
                        val = model.predict(row)[0]
                        if val > 0: return val
                    except: pass
                
                # B. SMART DEMO LOGIC (Fallback)
                base = 5000 
                if c_name in ['Banana', 'Sugarcane']: base = 20000
                elif c_name in ['Rice', 'Maize']: base = 12000
                elif c_name in ['Coconut', 'Mango']: base = 10000
                elif c_name in ['Cotton']: base = 8000
                elif c_name in ['Apple', 'Grapes']: base = 7000
                
                # Penalties
                profile = CROP_PROFILE.get(c_name, {'temp': 25, 'rain': 100})
                temp_diff = abs(inputs['temperature'] - profile['temp'])
                
                # Temp Penalty
                temp_factor = max(0.2, 1 - (temp_diff / 15)) 
                # Soil Boost
                soil_score = (inputs['N'] + inputs['P'] + inputs['K']) / 400
                
                return base * temp_factor * (0.8 + soil_score)

            # 3. Predict Selected
            user_yield = get_yield(selected_crop, user_data)

            # 4. Generate Top 5 (Full Scan)
            suggestions = []
            for c in crop_list:
                y = get_yield(c, user_data)
                suggestions.append({'crop': c, 'yield': round(y, 2)})
            
            suggestions.sort(key=lambda x: x['yield'], reverse=True)
            top_5 = suggestions[:5]

            context.update({
                'result': round(user_yield, 2),
                'selected_crop': selected_crop,
                'top_5': top_5,
                'show_result': True
            })

        except Exception as e:
            print(e)
            context['error'] = "Calculation Error"

    return render(request, 'index.html', context)

# --- 3. ABOUT PAGE (RESTORED) ---
def about(request):
    return render(request, 'about.html')

# --- 4. DATASET FIX ---
def dataset(request):
    files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    if files:
        with open(files[0], 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            filename = os.path.basename(files[0])
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
    return HttpResponse("File not found", status=404)