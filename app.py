import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI AgriYield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_models():
    model = joblib.load('best_yield_model.pkl')
    le = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, le, scaler

model, le, scaler = load_models()

available_crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
                   'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
                   'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
                   'coconut', 'cotton', 'jute', 'coffee']

numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: #001f3f;
    }
    
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem 1.5rem;
    }
    
    .header {
        text-align: center;
        padding: 2rem 0 3rem;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .input-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .input-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #ffffff;
        min-width: 120px;
    }
    
    .input-unit {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.6);
        min-width: 50px;
    }
    
    .value-display {
        font-size: 1.1rem;
        font-weight: 600;
        color: #6bb6ff;
        min-width: 80px;
        text-align: right;
    }
    
    .result-card {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .result-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-crop {
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .result-unit {
        font-size: 1.25rem;
        opacity: 0.9;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .recommendations-title {
        font-size: 1rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin: 2rem 0 1rem;
    }
    
    .recommendation-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin-bottom: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .recommendation-crop {
        font-weight: 500;
        color: #ffffff;
    }
    
    .recommendation-yield {
        font-weight: 700;
        color: #6bb6ff;
        font-family: monospace;
    }
    
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: rgba(255, 255, 255, 0.5);
    }
    
    .empty-icon {
        font-size: 4rem;
        opacity: 0.3;
        margin-bottom: 1rem;
    }
    
    .empty-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 0.5rem;
    }
    
    .stSlider > div > div > div {
        background: #4a90e2;
    }
    
    .stSlider > div > div > div > div {
        background: #ffffff;
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: #ffffff;
        font-weight: 500;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #4a90e2;
        background: rgba(255, 255, 255, 0.15);
    }
    
    .stSelectbox > div > div > select option {
        background: #001f3f;
        color: #ffffff;
    }
    
    .stButton > button {
        background: #4a90e2;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #5aa0f2;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="container">
        <div class="header">
            <h1>🌾 AI AgriYield Predictor</h1>
            <p>Intelligent crop yield prediction powered by machine learning</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Soil & Environmental Parameters</div>', unsafe_allow_html=True)

defaults = {'N': 50, 'P': 50, 'K': 50, 'temperature': 25.0, 'humidity': 50.0, 'ph': 7.0, 'rainfall': 100.0}
for key in defaults:
    if key not in st.session_state:
        st.session_state[key] = defaults[key]

param_configs = [
    {'name': 'N', 'label': 'Nitrogen (N)', 'unit': 'ppm', 'min': 0, 'max': 300, 'step': 1, 'type': 'int'},
    {'name': 'P', 'label': 'Phosphorus (P)', 'unit': 'ppm', 'min': 0, 'max': 150, 'step': 1, 'type': 'int'},
    {'name': 'K', 'label': 'Potassium (K)', 'unit': 'ppm', 'min': 0, 'max': 250, 'step': 1, 'type': 'int'},
    {'name': 'temperature', 'label': 'Temperature', 'unit': '°C', 'min': 5.0, 'max': 50.0, 'step': 0.1, 'type': 'float'},
    {'name': 'humidity', 'label': 'Humidity', 'unit': '%', 'min': 10.0, 'max': 100.0, 'step': 0.1, 'type': 'float'},
    {'name': 'ph', 'label': 'pH Level', 'unit': 'pH', 'min': 3.0, 'max': 10.0, 'step': 0.1, 'type': 'float'},
    {'name': 'rainfall', 'label': 'Rainfall', 'unit': 'mm', 'min': 0.0, 'max': 1200.0, 'step': 0.1, 'type': 'float'}
]

input_values = {}

for param in param_configs:
    col1, col2, col3, col4 = st.columns([2, 4, 1, 1])
    
    with col1:
        st.markdown(f'<div class="input-label">{param["label"]}</div>', unsafe_allow_html=True)
    
    with col2:
        if param['type'] == 'int':
            value = st.slider(
                param['name'],
                min_value=param['min'],
                max_value=param['max'],
                value=int(st.session_state[param['name']]),
                step=param['step'],
                label_visibility="collapsed",
                key=f"{param['name']}_slider"
            )
        else:
            value = st.slider(
                param['name'],
                min_value=param['min'],
                max_value=param['max'],
                value=float(st.session_state[param['name']]),
                step=param['step'],
                label_visibility="collapsed",
                key=f"{param['name']}_slider"
            )
    
    with col3:
        if param['type'] == 'int':
            value = st.number_input(
                param['name'],
                min_value=param['min'],
                max_value=param['max'],
                value=int(value),
                step=param['step'],
                label_visibility="collapsed",
                key=f"{param['name']}_num"
            )
        else:
            value = st.number_input(
                param['name'],
                min_value=param['min'],
                max_value=param['max'],
                value=float(value),
                step=param['step'],
                label_visibility="collapsed",
                key=f"{param['name']}_num"
            )
    
    with col4:
        st.markdown(f'<div class="input-unit">{param["unit"]}</div>', unsafe_allow_html=True)
    
    st.session_state[param['name']] = value
    input_values[param['name']] = value

st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Crop Selection</div>', unsafe_allow_html=True)
crop = st.selectbox("Select Target Crop", available_crops, label_visibility="collapsed", key="crop_select")

if st.button("Predict Yield", type="primary", use_container_width=True):
    with st.spinner("Analyzing parameters..."):
        crop_encoded = le.transform([crop])[0]
        input_data = pd.DataFrame([[
            input_values['N'],
            input_values['P'],
            input_values['K'],
            input_values['temperature'],
            input_values['humidity'],
            input_values['ph'],
            input_values['rainfall'],
            crop_encoded
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label_encoded'])
        
        input_scaled = input_data.copy()
        input_scaled[numeric_cols] = scaler.transform(input_data[numeric_cols])
        
        prediction = model.predict(input_scaled)[0]
        prediction_kt = prediction / 1000
        
        recommendations = []
        for test_crop in available_crops:
            try:
                test_crop_encoded = le.transform([test_crop])[0]
                test_input = pd.DataFrame([[
                    input_values['N'],
                    input_values['P'],
                    input_values['K'],
                    input_values['temperature'],
                    input_values['humidity'],
                    input_values['ph'],
                    input_values['rainfall'],
                    test_crop_encoded
                ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label_encoded'])
                test_input_scaled = test_input.copy()
                test_input_scaled[numeric_cols] = scaler.transform(test_input[numeric_cols])
                test_pred = model.predict(test_input_scaled)[0]
                test_pred_kt = test_pred / 1000
                recommendations.append({'crop': test_crop.title(), 'yield': round(test_pred_kt, 2)})
            except:
                continue
        
        recommendations.sort(key=lambda x: x['yield'], reverse=True)
        top_5 = recommendations[:5]
        
        st.session_state['prediction'] = prediction_kt
        st.session_state['crop'] = crop
        st.session_state['top_5'] = top_5
        st.session_state['show_result'] = True

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

if 'show_result' in st.session_state and st.session_state['show_result']:
    st.markdown(f"""
        <div class="result-card">
            <div class="result-badge">Prediction Complete</div>
            <div class="result-crop">{st.session_state['crop'].title()}</div>
            <div class="result-value">{st.session_state['prediction']:,.2f}</div>
            <div class="result-unit">Kilotonnes (kT)</div>
        </div>
    """, unsafe_allow_html=True)
    
    top_5 = st.session_state['top_5']
    crop_names = [item['crop'] for item in top_5]
    crop_yields = [item['yield'] for item in top_5]
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=crop_names,
            y=crop_yields,
            marker=dict(
                color='#4a90e2',
                line=dict(color='#357abd', width=1.5)
            ),
            text=[f"{y:.2f} kT" for y in crop_yields],
            textposition='outside',
            textfont=dict(color='#ffffff', size=12)
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color='rgba(255, 255, 255, 0.8)')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=11, color='rgba(255, 255, 255, 0.8)'),
            title='Yield (kT)'
        ),
        height=350,
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="recommendations-title">Alternative Crop Recommendations</div>', unsafe_allow_html=True)
    
    medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    for i, item in enumerate(top_5):
        st.markdown(f"""
            <div class="recommendation-item">
                <span class="recommendation-crop">{medals[i]} {item['crop']}</span>
                <span class="recommendation-yield">{item['yield']} kT</span>
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">Analytics Dashboard</div>
            <div style="color: rgba(255, 255, 255, 0.5);">Results will appear here after running a prediction</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        <p>&copy; 2025 <strong>AI AgriYield Predictor</strong>. All rights reserved.</p>
        <p style="margin-top: 0.5rem; font-size: 0.85rem;">XGBoost Regression Model | Accuracy: 96.33% (R² Score)</p>
    </div>
""", unsafe_allow_html=True)
