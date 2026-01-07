import os
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.paginator import Paginator
from django.http import FileResponse, Http404
from predictor.models import PredictionHistory, CropRecommendationHistory

# --- 1. ROBUST PATH FINDER ---
BASE_DIR = settings.BASE_DIR

def find_file(filename):
    search_paths = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, 'predictor', filename),
        os.path.join(BASE_DIR, 'Datasets', filename),
        filename
    ]
    for path in search_paths:
        if os.path.exists(path): return path
    return None

# --- 2. LOAD MODELS ---
files = {
    'yield_model': 'agri_yield_model.joblib',
    'scaler': 'scaler.joblib',
    'yield_le': 'label_encoder.joblib',
    'rec_model': 'crop_recommendation_model.joblib',
    'rec_le': 'crop_label_encoder.joblib'
}

loaded = {}
for key, name in files.items():
    path = find_file(name)
    if path:
        try: loaded[key] = joblib.load(path)
        except: loaded[key] = None
    else: loaded[key] = None

yield_model = loaded['yield_model']
scaler = loaded['scaler']
label_encoder = loaded['yield_le']
rec_model = loaded['rec_model']
rec_encoder = loaded['rec_le']

# --- HELPER: Pagination ---
def get_smart_pagination(current, total):
    if total <= 7: return range(1, total + 1)
    if current <= 4: return list(range(1, 6)) + ['...', total]
    if current >= total - 3: return [1, '...'] + list(range(total - 4, total + 1))
    return [1, '...'] + list(range(current - 2, current + 3)) + ['...', total]

# ==========================
# 3. YIELD PREDICTOR
# ==========================
def predict_yield(request):
    prediction = None
    error_msg = None
    top_5, bottom_5 = [], []
    
    form_inputs = [
        {'name': 'N', 'label': 'Nitrogen (N)', 'min': 0, 'max': 200, 'current': 50, 'unit': 'kg/ha', 'icon': '🧪'},
        {'name': 'P', 'label': 'Phosphorus (P)', 'min': 0, 'max': 200, 'current': 50, 'unit': 'kg/ha', 'icon': '⚗️'},
        {'name': 'K', 'label': 'Potassium (K)', 'min': 0, 'max': 200, 'current': 50, 'unit': 'kg/ha', 'icon': '🧂'},
        {'name': 'temperature', 'label': 'Temperature', 'min': -10, 'max': 60, 'current': 25, 'unit': '°C', 'icon': '🌡️'},
        {'name': 'humidity', 'label': 'Humidity', 'min': 0, 'max': 100, 'current': 50, 'unit': '%', 'icon': '💧'},
        {'name': 'ph', 'label': 'Soil pH', 'min': 0, 'max': 14, 'current': 6.5, 'unit': 'pH', 'icon': '🧪'},
        {'name': 'rainfall', 'label': 'Rainfall', 'min': 0, 'max': 300, 'current': 100, 'unit': 'mm', 'icon': '🌧️'},
    ]

    crop_list = sorted([c.title() for c in label_encoder.classes_]) if label_encoder else []

    if request.method == 'POST' and 'clear_history' in request.POST:
        PredictionHistory.objects.all().delete()
        return redirect('predict_yield')

    if request.method == 'POST' and 'predict' in request.POST:
        try:
            inputs = {}
            for item in form_inputs:
                val = float(request.POST.get(item['name'], item['current']))
                item['current'] = val 
                inputs[item['name']] = val
            
            selected_crop = request.POST.get('crop')
            t, h, ph, r = inputs['temperature'], inputs['humidity'], inputs['ph'], inputs['rainfall']
            n, p, k = inputs['N'], inputs['P'], inputs['K']

            # Validation
            is_impossible = False
            if t > 48: is_impossible = True; error_msg = "Extreme Heat (>48°C). Yield impossible."
            elif t < 5: is_impossible = True; error_msg = "Extreme Cold (<5°C). Yield impossible."
            elif ph < 3.5: is_impossible = True; error_msg = "Soil Too Acidic (pH < 3.5)."
            elif ph > 9.5: is_impossible = True; error_msg = "Soil Too Alkaline (pH > 9.5)."
            elif r < 10 and h < 15: is_impossible = True; error_msg = "Severe Drought detected."

            def get_yield(c_name):
                if is_impossible or not yield_model: return 0
                season, stress = t * h, r / t if t != 0 else 0
                feats = np.array([[n, p, k, t, h, ph, r, season, stress]])
                scaled = scaler.transform(feats)
                try: c_enc = label_encoder.transform([c_name])[0]
                except: 
                    try: c_enc = label_encoder.transform([c_name.lower()])[0]
                    except: return 0
                val = yield_model.predict(np.append(scaled[0], c_enc).reshape(1, -1))[0]
                return max(0, np.expm1(val) if val < 20 else val)

            if not is_impossible:
                user_yield = get_yield(selected_crop)
                prediction = f"{user_yield:,.2f} Kg/Ha"
                
                # Context Logic (Top 5 & Bottom 5)
                original_classes = label_encoder.classes_ if label_encoder is not None else []
                all_preds = []
                for c in original_classes:
                    y = get_yield(c)
                    if y > 0: all_preds.append({'crop': c.title(), 'yield': round(y, 2)})
                
                all_preds.sort(key=lambda x: x['yield'], reverse=True)
                
                top_5 = all_preds[:5]
                bottom_5 = all_preds[-5:] # Smallest yields
                bottom_5.sort(key=lambda x: x['yield']) # Sort for chart display (low to high)
                
                PredictionHistory.objects.create(crop=selected_crop, N=n, P=p, K=k, temperature=t, humidity=h, ph=ph, rainfall=r, predicted_yield=prediction)

        except Exception as e: error_msg = str(e)

    history_list = PredictionHistory.objects.all().order_by('-created_at')
    paginator = Paginator(history_list, 5) 
    page_obj = paginator.get_page(request.GET.get('page', 1))

    return render(request, 'YieldPredictor.html', {
        'form_inputs': form_inputs, 'crop_list': crop_list, 'prediction': prediction,
        'selected_crop': request.POST.get('crop'), 'top_5': top_5, 'bottom_5': bottom_5,
        'error': error_msg, 'history': page_obj, 'custom_range': get_smart_pagination(page_obj.number, paginator.num_pages),
        'show_result': prediction is not None
    })

# ==========================
# 4. CROP RECOMMENDER
# ==========================
def recommend_crop(request):
    result = None
    error_msg = None
    top_5, bottom_5 = [], []
    
    form_inputs = [
        {'name': 'N', 'label': 'Nitrogen (N)', 'min': 0, 'max': 140, 'current': 90, 'unit': 'kg/ha', 'icon': '🧪'},
        {'name': 'P', 'label': 'Phosphorus (P)', 'min': 0, 'max': 145, 'current': 42, 'unit': 'kg/ha', 'icon': '⚗️'},
        {'name': 'K', 'label': 'Potassium (K)', 'min': 0, 'max': 205, 'current': 43, 'unit': 'kg/ha', 'icon': '🧂'},
        {'name': 'temperature', 'label': 'Temperature', 'min': 0, 'max': 50, 'current': 20.8, 'unit': '°C', 'icon': '🌡️'},
        {'name': 'humidity', 'label': 'Humidity', 'min': 0, 'max': 100, 'current': 82.0, 'unit': '%', 'icon': '💧'},
        {'name': 'ph', 'label': 'Soil pH', 'min': 0, 'max': 14, 'current': 6.5, 'unit': 'pH', 'icon': '🧪'},
        {'name': 'rainfall', 'label': 'Rainfall', 'min': 0, 'max': 300, 'current': 202.9, 'unit': 'mm', 'icon': '🌧️'},
    ]

    if request.method == 'POST' and 'clear_history' in request.POST:
        CropRecommendationHistory.objects.all().delete()
        return redirect('recommend_crop')

    if request.method == 'POST' and 'recommend' in request.POST:
        try:
            inputs = []
            d = {}
            for item in form_inputs:
                val = float(request.POST.get(item['name'], item['current']))
                item['current'] = val; inputs.append(val); d[item['name']] = val
            
            t, h, ph, r = d['temperature'], d['humidity'], d['ph'], d['rainfall']

            if t > 48: error_msg = "Extreme Heat Detected (>48°C)."
            elif t < 5: error_msg = "Extreme Cold Detected (<5°C)."
            elif ph < 3.5: error_msg = "Soil Too Acidic (pH < 3.5)."
            elif ph > 9.5: error_msg = "Soil Too Alkaline (pH > 9.5)."
            elif r < 10 and h < 15: error_msg = "Severe Drought Detected."
            elif rec_model and rec_encoder:
                probs = rec_model.predict_proba([inputs])[0]
                classes = rec_encoder.classes_
                prob_list = sorted([{'crop': classes[i].title(), 'prob': probs[i] * 100} for i in range(len(probs))], key=lambda x: x['prob'], reverse=True)
                
                result = {'crop': prob_list[0]['crop'], 'confidence': round(prob_list[0]['prob'], 2)}
                top_5 = prob_list[:5]
                bottom_5 = [p for p in prob_list if p['prob'] < 1][-5:]
                
                CropRecommendationHistory.objects.create(N=d['N'], P=d['P'], K=d['K'], temperature=t, humidity=h, ph=ph, rainfall=r, recommended_crop=result['crop'], confidence_score=f"{result['confidence']}%")
            else:
                error_msg = "Recommender Models not loaded."

        except Exception as e: error_msg = str(e)

    history_list = CropRecommendationHistory.objects.all().order_by('-created_at')
    paginator = Paginator(history_list, 5) 
    page_obj = paginator.get_page(request.GET.get('page', 1))

    return render(request, 'CropRecommender.html', {
        'form_inputs': form_inputs, 'result': result, 'top_5': top_5, 'bottom_5': bottom_5,
        'error': error_msg, 'history': page_obj, 'custom_range': get_smart_pagination(page_obj.number, paginator.num_pages)
    })

# --- OTHER VIEWS ---
def dashboard(request): return render(request, 'index.html')
def about(request): return render(request, 'about.html')
def dataset(request):
    ctx = {'total_rows': 0, 'preview_data': [], 'error': None}
    try:
        path = find_file('Final_Agri_Data.csv')
        if path:
            df = pd.read_csv(path)
            ctx = {'total_rows': len(df), 'total_cols': df.shape[1], 'columns': df.columns.tolist(), 'preview_data': df.head(50).to_dict(orient='records'), 'unique_crops': df['label'].nunique() if 'label' in df else 0, 'avg_yield': round(df['Value'].mean(), 2) if 'Value' in df else 0}
        else: ctx['error'] = "File not found."
    except Exception as e: ctx['error'] = str(e)
    return render(request, 'dataset.html', ctx)
def download_dataset(request):
    path = find_file('Final_Agri_Data.csv')
    if path: return FileResponse(open(path, 'rb'), as_attachment=True, filename='Final_Agri_Data.csv')
    raise Http404("File not found")