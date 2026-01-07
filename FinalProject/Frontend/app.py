import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Page Configuration
st.set_page_config(
    page_title="AI AgriYield & Crop Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - ECO-FUTURISM THEME WITH GLASSMORPHISM
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 50%, #0E1117 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(20, 25, 35, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(0, 255, 163, 0.2);
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 255, 163, 0.1),
                    0 0 0 1px rgba(0, 255, 163, 0.05) inset;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(0, 255, 163, 0.4);
        box-shadow: 0 12px 48px rgba(0, 255, 163, 0.2),
                    0 0 0 1px rgba(0, 255, 163, 0.1) inset;
        transform: translateY(-2px);
    }
    
    /* Hero Card - Winner Display */
    .hero-card {
        background: linear-gradient(135deg, 
            rgba(0, 255, 163, 0.15) 0%, 
            rgba(46, 139, 87, 0.15) 100%);
        backdrop-filter: blur(30px);
        border-radius: 32px;
        border: 2px solid rgba(0, 255, 163, 0.3);
        padding: 4rem 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(0, 255, 163, 0.2),
                    0 0 40px rgba(0, 255, 163, 0.1) inset;
        animation: glowPulse 3s ease-in-out infinite;
    }
    
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 20px 60px rgba(0, 255, 163, 0.2), 0 0 40px rgba(0, 255, 163, 0.1) inset; }
        50% { box-shadow: 0 20px 80px rgba(0, 255, 163, 0.3), 0 0 60px rgba(0, 255, 163, 0.15) inset; }
    }
    
    /* Alternative Cards */
    .alt-card {
        background: rgba(20, 25, 35, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(0, 255, 163, 0.15);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .alt-card:hover {
        border-color: rgba(0, 255, 163, 0.3);
        transform: translateX(5px);
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00FFA3 0%, #2E8B57 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .section-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Input Group Labels */
    .input-group-label {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #00FFA3;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(0, 255, 163, 0.3);
    }
    
    /* Parameter Label */
    .param-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0.5rem;
    }
    
    /* Parameter Value Display */
    .param-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #00FFA3;
        text-align: right;
    }
    
    /* Unit Label */
    .param-unit {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 500;
    }
    
    /* KPI Metric Display */
    .kpi-metric {
        font-family: 'Poppins', sans-serif;
        font-size: 5.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00FFA3 0%, #2E8B57 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin: 1.5rem 0;
    }
    
    .kpi-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        background: rgba(0, 255, 163, 0.15);
        border: 1px solid rgba(0, 255, 163, 0.3);
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #00FFA3;
        margin-top: 1rem;
    }
    
    /* Status Indicator */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #00FFA3;
        box-shadow: 0 0 10px rgba(0, 255, 163, 0.6);
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    /* Custom Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00FFA3 0%, #2E8B57 100%);
        color: #0E1117;
        border: none;
        border-radius: 16px;
        padding: 1.2rem 3rem;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(0, 255, 163, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(0, 255, 163, 0.4);
        background: linear-gradient(135deg, #2E8B57 0%, #00FFA3 100%);
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: rgba(0, 255, 163, 0.2);
    }
    
    .stSlider > div > div > div > div {
        background: #00FFA3;
        box-shadow: 0 0 10px rgba(0, 255, 163, 0.5);
    }
    
    /* Number Input Styling */
    .stNumberInput > div > div > input {
        background: rgba(20, 25, 35, 0.8);
        border: 2px solid rgba(0, 255, 163, 0.2);
        border-radius: 12px;
        color: #ffffff;
        font-weight: 600;
        padding: 0.75rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #00FFA3;
        box-shadow: 0 0 0 3px rgba(0, 255, 163, 0.2);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div > select {
        background: rgba(20, 25, 35, 0.8);
        border: 2px solid rgba(0, 255, 163, 0.2);
        border-radius: 12px;
        color: #ffffff;
        font-weight: 600;
        padding: 0.75rem;
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: #00FFA3;
        box-shadow: 0 0 0 3px rgba(0, 255, 163, 0.2);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(0, 255, 163, 0.1);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(0, 255, 163, 0.1);
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: rgba(255, 255, 255, 0.5);
    }
    
    .empty-icon {
        font-size: 6rem;
        opacity: 0.3;
        margin-bottom: 1.5rem;
    }
    
    /* Arrow Indicator */
    .arrow-up {
        color: #00FFA3;
        font-size: 2rem;
        animation: bounce 2s infinite;
    }
    
    .arrow-down {
        color: #ff6b6b;
        font-size: 2rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Grid Container */
    .grid-container {
        display: grid;
        gap: 1.5rem;
    }
    
    /* Crop Icon Mapping */
    .crop-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
@st.cache_resource
def load_yield_models():
    """Load yield prediction models"""
    try:
        base = os.path.dirname(__file__)
        model_dir = os.path.normpath(os.path.join(base, '..', 'Backend'))
        model = joblib.load(os.path.join(model_dir, 'best_yield_model.pkl'))
        le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        return model, le, scaler, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"Error loading yield models: {str(e)}"

@st.cache_resource
def load_crop_classifier():
    """Load crop classification models"""
    try:
        base = os.path.dirname(__file__)
        model_dir = os.path.normpath(os.path.join(base, '..', 'Backend'))
        classifier = joblib.load(os.path.join(model_dir, 'crop_classifier_model.pkl'))
        crop_encoder = joblib.load(os.path.join(model_dir, 'crop_name_encoder.pkl'))
        crop_scaler = joblib.load(os.path.join(model_dir, 'crop_scaler.pkl'))
        return classifier, crop_encoder, crop_scaler, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, f"Error loading crop classifier: {str(e)}"

# Load models
yield_model, yield_le, yield_scaler, yield_error = load_yield_models()
crop_classifier, crop_encoder, crop_scaler, crop_error = load_crop_classifier()

# ============================================================================
# CONSTANTS
# ============================================================================
available_crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
                   'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
                   'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
                   'coconut', 'cotton', 'jute', 'coffee']

crop_icons = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫘', 'mothbeans': '🫘', 'mungbean': '🫘', 'blackgram': '🫘',
    'lentil': '🫘', 'pomegranate': '🍎', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍎',
    'orange': '🍊', 'papaya': '🥭', 'coconut': '🥥', 'cotton': '🌾',
    'jute': '🌾', 'coffee': '☕'
}

numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    # Logo and Title
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🌾</h1>
            <h2 style="font-family: 'Poppins', sans-serif; font-size: 1.5rem; font-weight: 800; 
                       background: linear-gradient(135deg, #00FFA3 0%, #2E8B57 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       margin-bottom: 0.5rem;">AI AgriYield</h2>
            <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">Crop Predictor</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Status
    st.markdown("### System Status")
    yield_status = "✅ Ready" if yield_model is not None else "❌ Not Available"
    crop_status = "✅ Ready" if crop_classifier is not None else "❌ Not Available"
    
    st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0;">
                <span style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">Yield Model:</span>
                <span style="color: {'#00FFA3' if yield_model is not None else '#ff6b6b'}; font-weight: 700;">{yield_status}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0;">
                <span style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">Crop Model:</span>
                <span style="color: {'#00FFA3' if crop_classifier is not None else '#ff6b6b'}; font-weight: 700;">{crop_status}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    mode = st.radio(
        "",
        ["🌱 Smart Crop Recommender", "📊 Yield Precision Tool"],
        key="mode_selector",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # How to Use
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Smart Crop Recommender:**
        - Enter your soil and weather conditions
        - Get AI-powered crop recommendations
        - View confidence scores for each option
        
        **Yield Precision Tool:**
        - Select a crop and enter conditions
        - Get precise yield predictions in Kilotons per Hectare (kt/Ha)
        - Compare with optimal conditions
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: rgba(255, 255, 255, 0.5); font-size: 0.85rem; padding-top: 2rem;">
            <p>AI AgriYield & Crop Predictor</p>
            <p style="margin-top: 0.5rem;">© 2025</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
defaults = {'N': 50, 'P': 50, 'K': 50, 'temperature': 25.0, 'humidity': 50.0, 'ph': 7.0, 'rainfall': 100.0}
for key in defaults:
    if key not in st.session_state:
        st.session_state[key] = defaults[key]

# ============================================================================
# PAGE 1: SMART CROP RECOMMENDER
# ============================================================================
if mode == "🌱 Smart Crop Recommender":
    if crop_classifier is None or crop_encoder is None or crop_scaler is None:
        st.error("⚠️ Crop recommendation models not available. Please ensure model files are present.")
    else:
        # Header
        st.markdown("""
            <div class="section-header">Find the Perfect Crop for Your Soil</div>
            <div class="section-subtitle">Enter your field conditions and get AI-powered crop recommendations</div>
        """, unsafe_allow_html=True)
        
        # Input Section in Grid Layout
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Group 1: Soil Health
        st.markdown('<div class="input-group-label">🌍 Soil Health Parameters</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="param-label">Nitrogen (N)</div>', unsafe_allow_html=True)
            n_val = st.slider("N", 0, 300, int(st.session_state['N']), key="crop_n_slider", label_visibility="collapsed")
            n_input = st.number_input("N", 0, 300, n_val, key="crop_n_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">ppm</div>', unsafe_allow_html=True)
            st.session_state['N'] = n_input
        
        with col2:
            st.markdown('<div class="param-label">Phosphorus (P)</div>', unsafe_allow_html=True)
            p_val = st.slider("P", 0, 150, int(st.session_state['P']), key="crop_p_slider", label_visibility="collapsed")
            p_input = st.number_input("P", 0, 150, p_val, key="crop_p_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">ppm</div>', unsafe_allow_html=True)
            st.session_state['P'] = p_input
        
        with col3:
            st.markdown('<div class="param-label">Potassium (K)</div>', unsafe_allow_html=True)
            k_val = st.slider("K", 0, 250, int(st.session_state['K']), key="crop_k_slider", label_visibility="collapsed")
            k_input = st.number_input("K", 0, 250, k_val, key="crop_k_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">ppm</div>', unsafe_allow_html=True)
            st.session_state['K'] = k_input
        
        with col4:
            st.markdown('<div class="param-label">pH Level</div>', unsafe_allow_html=True)
            ph_val = st.slider("pH", 3.0, 10.0, float(st.session_state['ph']), 0.1, key="crop_ph_slider", label_visibility="collapsed")
            ph_input = st.number_input("pH", 3.0, 10.0, ph_val, 0.1, key="crop_ph_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">pH</div>', unsafe_allow_html=True)
            st.session_state['ph'] = ph_input
        
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        
        # Group 2: Atmosphere
        st.markdown('<div class="input-group-label">🌤️ Atmospheric Conditions</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="param-label">Temperature</div>', unsafe_allow_html=True)
            temp_val = st.slider("Temp", 5.0, 50.0, float(st.session_state['temperature']), 0.1, key="crop_temp_slider", label_visibility="collapsed")
            temp_input = st.number_input("Temp", 5.0, 50.0, temp_val, 0.1, key="crop_temp_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">°C</div>', unsafe_allow_html=True)
            st.session_state['temperature'] = temp_input
        
        with col2:
            st.markdown('<div class="param-label">Humidity</div>', unsafe_allow_html=True)
            hum_val = st.slider("Humidity", 10.0, 100.0, float(st.session_state['humidity']), 0.1, key="crop_hum_slider", label_visibility="collapsed")
            hum_input = st.number_input("Humidity", 10.0, 100.0, hum_val, 0.1, key="crop_hum_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">%</div>', unsafe_allow_html=True)
            st.session_state['humidity'] = hum_input
        
        with col3:
            st.markdown('<div class="param-label">Rainfall</div>', unsafe_allow_html=True)
            rain_val = st.slider("Rainfall", 0.0, 1200.0, float(st.session_state['rainfall']), 0.1, key="crop_rain_slider", label_visibility="collapsed")
            rain_input = st.number_input("Rainfall", 0.0, 1200.0, rain_val, 0.1, key="crop_rain_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">mm</div>', unsafe_allow_html=True)
            st.session_state['rainfall'] = rain_input
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Button
        if st.button("Analyze Soil Conditions 🔍", type="primary", use_container_width=True, key="crop_analyze_btn"):
            with st.spinner("🤖 Analyzing conditions and generating recommendations..."):
                try:
                    input_data = np.array([[
                        st.session_state['N'],
                        st.session_state['P'],
                        st.session_state['K'],
                        st.session_state['temperature'],
                        st.session_state['humidity'],
                        st.session_state['ph'],
                        st.session_state['rainfall']
                    ]])
                    
                    input_scaled = crop_scaler.transform(input_data)
                    probabilities = crop_classifier.predict_proba(input_scaled)[0]
                    
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    
                    top_3_crops = []
                    for idx in top_indices:
                        crop_name = crop_encoder.inverse_transform([idx])[0]
                        probability = probabilities[idx]
                        top_3_crops.append({'crop': crop_name, 'confidence': probability})
                    
                    st.session_state['top_3_crops'] = top_3_crops
                    st.session_state['crop_show_result'] = True
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Output Section
        if 'crop_show_result' in st.session_state and st.session_state['crop_show_result']:
            top_3 = st.session_state['top_3_crops']
            best_crop = top_3[0]
            
            # Winner Card (Hero Section)
            st.markdown(f"""
                <div class="hero-card">
                    <h2 style="font-size: 1.5rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 2px;">🏆 Best Match</h2>
                    <h1 style="font-size: 4rem; font-weight: 900; margin: 1.5rem 0; color: #00FFA3;">{best_crop['crop'].title()}</h1>
                    <div class="confidence-badge">{best_crop['confidence']*100:.1f}% Match</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Alternatives
            st.markdown('<div class="input-group-label" style="margin-top: 3rem;">🥈 Alternative Options</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                alt1 = top_3[1]
                st.markdown(f"""
                    <div class="alt-card">
                        <h3 style="font-size: 1.8rem; color: #00FFA3; margin-bottom: 0.5rem;">🥈 {alt1['crop'].title()}</h3>
                        <div class="confidence-badge">{alt1['confidence']*100:.1f}% Match</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                alt2 = top_3[2]
                st.markdown(f"""
                    <div class="alt-card">
                        <h3 style="font-size: 1.8rem; color: #00FFA3; margin-bottom: 0.5rem;">🥉 {alt2['crop'].title()}</h3>
                        <div class="confidence-badge">{alt2['confidence']*100:.1f}% Match</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Visualization - Bar Chart
            st.markdown('<div class="glass-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
            st.markdown('<div class="input-group-label">📊 Confidence Comparison</div>', unsafe_allow_html=True)
            
            crop_names = [item['crop'].title() for item in top_3]
            confidences = [item['confidence'] * 100 for item in top_3]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=crop_names,
                    y=confidences,
                    marker=dict(
                        color=['#00FFA3', '#2E8B57', '#1a5f3f'],
                        line=dict(color='#00FFA3', width=2),
                        opacity=0.9
                    ),
                    text=[f"{c:.1f}%" for c in confidences],
                    textposition='outside',
                    textfont=dict(color='#ffffff', size=16, family='Poppins', weight='bold')
                )
            ])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=14, family='Inter'),
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=14, color='rgba(255, 255, 255, 0.9)', family='Poppins'),
                    title=dict(
                        text='Recommended Crops',
                        font=dict(size=16, family='Poppins', weight='bold', color='rgba(255, 255, 255, 0.9)')
                    )
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0, 255, 163, 0.2)',
                    tickfont=dict(size=14, color='rgba(255, 255, 255, 0.9)', family='Inter'),
                    title=dict(
                        text='Confidence (%)',
                        font=dict(size=16, family='Poppins', weight='bold', color='rgba(255, 255, 255, 0.9)')
                    ),
                    range=[0, 100]
                ),
                height=400,
                margin=dict(l=60, r=30, t=30, b=60),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 2: YIELD PRECISION TOOL
# ============================================================================
elif mode == "📊 Yield Precision Tool":
    if yield_model is None or yield_le is None or yield_scaler is None:
        st.error("⚠️ Yield prediction models not available. Please ensure model files are present.")
    else:
        # Header
        st.markdown("""
            <div class="section-header">Predict Harvest Efficiency</div>
            <div class="section-subtitle">Select a crop and enter conditions to get precise yield predictions</div>
        """, unsafe_allow_html=True)
        
        # Input Section in Grid Layout
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Group 1: Soil Health
        st.markdown('<div class="input-group-label">🌍 Soil Health Parameters</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="param-label">Nitrogen (N)</div>', unsafe_allow_html=True)
            n_val = st.slider("N", 0, 300, int(st.session_state['N']), key="yield_n_slider", label_visibility="collapsed")
            n_input = st.number_input("N", 0, 300, n_val, key="yield_n_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">ppm</div>', unsafe_allow_html=True)
            st.session_state['N'] = n_input
        
        with col2:
            st.markdown('<div class="param-label">Phosphorus (P)</div>', unsafe_allow_html=True)
            p_val = st.slider("P", 0, 150, int(st.session_state['P']), key="yield_p_slider", label_visibility="collapsed")
            p_input = st.number_input("P", 0, 150, p_val, key="yield_p_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">ppm</div>', unsafe_allow_html=True)
            st.session_state['P'] = p_input
        
        with col3:
            st.markdown('<div class="param-label">Potassium (K)</div>', unsafe_allow_html=True)
            k_val = st.slider("K", 0, 250, int(st.session_state['K']), key="yield_k_slider", label_visibility="collapsed")
            k_input = st.number_input("K", 0, 250, k_val, key="yield_k_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">ppm</div>', unsafe_allow_html=True)
            st.session_state['K'] = k_input
        
        with col4:
            st.markdown('<div class="param-label">pH Level</div>', unsafe_allow_html=True)
            ph_val = st.slider("pH", 3.0, 10.0, float(st.session_state['ph']), 0.1, key="yield_ph_slider", label_visibility="collapsed")
            ph_input = st.number_input("pH", 3.0, 10.0, ph_val, 0.1, key="yield_ph_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">pH</div>', unsafe_allow_html=True)
            st.session_state['ph'] = ph_input
        
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        
        # Group 2: Atmosphere
        st.markdown('<div class="input-group-label">🌤️ Atmospheric Conditions</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="param-label">Temperature</div>', unsafe_allow_html=True)
            temp_val = st.slider("Temp", 5.0, 50.0, float(st.session_state['temperature']), 0.1, key="yield_temp_slider", label_visibility="collapsed")
            temp_input = st.number_input("Temp", 5.0, 50.0, temp_val, 0.1, key="yield_temp_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">°C</div>', unsafe_allow_html=True)
            st.session_state['temperature'] = temp_input
        
        with col2:
            st.markdown('<div class="param-label">Humidity</div>', unsafe_allow_html=True)
            hum_val = st.slider("Humidity", 10.0, 100.0, float(st.session_state['humidity']), 0.1, key="yield_hum_slider", label_visibility="collapsed")
            hum_input = st.number_input("Humidity", 10.0, 100.0, hum_val, 0.1, key="yield_hum_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">%</div>', unsafe_allow_html=True)
            st.session_state['humidity'] = hum_input
        
        with col3:
            st.markdown('<div class="param-label">Rainfall</div>', unsafe_allow_html=True)
            rain_val = st.slider("Rainfall", 0.0, 1200.0, float(st.session_state['rainfall']), 0.1, key="yield_rain_slider", label_visibility="collapsed")
            rain_input = st.number_input("Rainfall", 0.0, 1200.0, rain_val, 0.1, key="yield_rain_input", label_visibility="collapsed")
            st.markdown(f'<div class="param-unit">mm</div>', unsafe_allow_html=True)
            st.session_state['rainfall'] = rain_input
        
        # Crop Selector
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="input-group-label">🌾 Crop Selection</div>', unsafe_allow_html=True)
        
        crop_options = [f"{crop_icons.get(crop, '🌾')} {crop.title()}" for crop in available_crops]
        selected_crop_display = st.selectbox("", crop_options, key="yield_crop_select", label_visibility="collapsed")
        selected_crop = selected_crop_display.split(' ', 1)[1].lower() if ' ' in selected_crop_display else selected_crop_display.lower()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Button
        if st.button("Calculate Yield Potential 🚀", type="primary", use_container_width=True, key="yield_calc_btn"):
            with st.spinner("🔍 Calculating yield prediction..."):
                try:
                    crop_encoded = yield_le.transform([selected_crop])[0]
                    input_data = pd.DataFrame([[
                        st.session_state['N'],
                        st.session_state['P'],
                        st.session_state['K'],
                        st.session_state['temperature'],
                        st.session_state['humidity'],
                        st.session_state['ph'],
                        st.session_state['rainfall'],
                        crop_encoded
                    ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label_encoded'])
                    
                    input_scaled = input_data.copy()
                    input_scaled[numeric_cols] = yield_scaler.transform(input_data[numeric_cols])
                    
                    prediction = yield_model.predict(input_scaled)[0]
                    
                    st.session_state['yield_prediction'] = prediction
                    st.session_state['yield_crop'] = selected_crop
                    st.session_state['yield_show_result'] = True
                    st.success("✅ Prediction complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Output Section
        if 'yield_show_result' in st.session_state and st.session_state['yield_show_result']:
            prediction_kg_ha = st.session_state['yield_prediction']
            # Convert kilograms per hectare to kilotons per hectare (1 kiloton = 1,000,000 kg)
            prediction_kt_ha = prediction_kg_ha / 1_000_000
            
            # Determine if high or low yield (dummy logic - can be improved)
            # Assuming average yield is around 3000-5000 Kg/Ha (~0.003-0.005 kt/Ha) for most crops
            is_high_yield = prediction_kg_ha > 4000
            arrow_icon = "⬆️" if is_high_yield else "⬇️"
            arrow_class = "arrow-up" if is_high_yield else "arrow-down"
            yield_status = "High Yield" if is_high_yield else "Low Yield"
            status_color = "#00FFA3" if is_high_yield else "#ff6b6b"
            
            # KPI Metric Display
            st.markdown(f"""
                <div class="hero-card">
                    <h2 style="font-size: 1.5rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 2px;">Predicted Yield</h2>
                    <h1 style="font-size: 4rem; font-weight: 900; margin: 1.5rem 0; color: #00FFA3;">{st.session_state['yield_crop'].title()}</h1>
                    <div class="kpi-metric">{prediction_kt_ha:,.6f}</div>
                    <div class="kpi-label">Kilotons per Hectare (kt/Ha)</div>
                    <div style="margin-top: 1.5rem;">
                        <span class="{arrow_class}">{arrow_icon}</span>
                        <span style="color: {status_color}; font-size: 1.3rem; font-weight: 700; margin-left: 0.5rem;">{yield_status}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
