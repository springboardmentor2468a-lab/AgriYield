import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== CONFIG ==================
st.set_page_config(
    page_title="AI AgriYield Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== LOAD MODEL ==================
model = joblib.load("crop_yield_pipeline.pkl")

# Crop list
CROPS = [
    "rice", "banana", "mango", "orange", "papaya",
    "grapes", "watermelon", "muskmelon", "apple",
    "coffee", "cotton", "jute", "lentil",
    "chickpea", "pigeonpeas", "mothbeans", "coconut"
]

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global styles */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
}

/* Remove empty space at top */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* Card styling */
.card {
    background: linear-gradient(145deg, #1a2332 0%, #14192a 100%);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), 
                0 0 0 1px rgba(255,255,255,0.03);
    border: 1px solid rgba(74, 222, 128, 0.1);
    backdrop-filter: blur(10px);
}

/* Welcome page card */
.welcome-card {
    background: linear-gradient(145deg, #1a2332 0%, #14192a 100%);
    padding: 50px;
    border-radius: 24px;
    box-shadow: 0 25px 70px rgba(0,0,0,0.6);
    border: 2px solid rgba(74, 222, 128, 0.15);
    margin: 20px 0;
}

/* Feature card */
.feature-card {
    background: rgba(30, 41, 59, 0.5);
    padding: 25px;
    border-radius: 16px;
    border-left: 4px solid #4ade80;
    margin: 15px 0;
    transition: all 0.3s ease;
}

.feature-card:hover {
    background: rgba(30, 41, 59, 0.8);
    transform: translateX(5px);
    box-shadow: 0 10px 30px rgba(74, 222, 128, 0.2);
}

/* Result card with gradient */
.result-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%);
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0 25px 70px rgba(34, 197, 94, 0.15),
                0 0 0 1px rgba(74, 222, 128, 0.2);
    border: 2px solid rgba(74, 222, 128, 0.15);
    margin-bottom: 25px;
}

/* Typography */
h1, h2, h3, h4 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

.main-title {
    font-size: 52px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px !important;
}

.subtitle {
    color: #94a3b8 !important;
    font-size: 18px !important;
    margin-bottom: 40px !important;
}

/* Section headers with icons */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
    padding-bottom: 15px;
    border-bottom: 2px solid rgba(74, 222, 128, 0.2);
}

.section-header h3 {
    margin: 0 !important;
    font-size: 24px !important;
}

/* Slider styling */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #22c55e 0%, #4ade80 100%);
}

.stSlider > div > div > div {
    background: rgba(74, 222, 128, 0.1);
}

/* Number input styling */
.stNumberInput > div > div > input {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(74, 222, 128, 0.2) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Select box styling */
.stSelectbox > div > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(74, 222, 128, 0.2) !important;
    border-radius: 10px !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    box-shadow: 0 15px 40px rgba(34, 197, 94, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* Result display */
.yield-value {
    font-size: 56px !important;
    font-weight: 800 !important;
    color: #4ade80 !important;
    text-shadow: 0 0 30px rgba(74, 222, 128, 0.5);
    margin: 20px 0 !important;
}

.crop-name {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin: 15px 0 !important;
}

.label-text {
    color: #94a3b8 !important;
    font-size: 14px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-bottom: 5px !important;
}

/* Success badge */
.success-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
    padding: 10px 20px;
    border-radius: 50px;
    font-weight: 600;
    margin-bottom: 25px;
    border: 1px solid rgba(74, 222, 128, 0.3);
}

/* Recommendation items */
.rec-item {
    background: rgba(30, 41, 59, 0.5);
    padding: 15px 20px;
    border-radius: 12px;
    margin: 10px 0;
    border-left: 4px solid #4ade80;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.3s ease;
}

.rec-item:hover {
    background: rgba(30, 41, 59, 0.8);
    transform: translateX(5px);
}

.rec-crop {
    color: #f8fafc;
    font-weight: 600;
    font-size: 16px;
}

.rec-yield {
    color: #4ade80;
    font-weight: 700;
    font-size: 16px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    margin-top: 50px;
    padding: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* Remove spacing */
.element-container {
    margin: 0 !important;
    padding: 0 !important;
}

/* Chart styling */
.stPlotlyChart, .stPyplot {
    background: transparent !important;
}

/* Step number */
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    border-radius: 50%;
    color: white;
    font-weight: 700;
    font-size: 18px;
    margin-right: 15px;
    flex-shrink: 0;
}

/* Instruction item */
.instruction-item {
    display: flex;
    align-items: flex-start;
    margin: 20px 0;
    padding: 20px;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 12px;
    border-left: 3px solid #4ade80;
}

.instruction-content {
    flex: 1;
}

.instruction-content h4 {
    color: #f8fafc !important;
    margin: 0 0 10px 0 !important;
    font-size: 18px !important;
}

.instruction-content p {
    color: #94a3b8 !important;
    margin: 0 !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

# ================== WELCOME PAGE ==================
def show_welcome_page():
    # Header
    st.markdown("""
    <div style="text-align:center; margin-bottom: 50px;">
        <div style="font-size: 70px; margin-bottom: 20px;">🌾</div>
        <h1 class="main-title">Smart Yield Predictor</h1>
        <p class="subtitle">Maximize agricultural output using AI-driven soil & climate analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="font-size: 32px; color: #4ade80;">Welcome to AI AgriYield Predictor! 👋</h2>
            <p style="color: #94a3b8; font-size: 16px; line-height: 1.8;">
                Our advanced machine learning system helps farmers and agricultural professionals 
                make data-driven decisions by predicting crop yields based on soil parameters and climate conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Features
        st.markdown("""
        <div class="section-header" style="border: none; margin-top: 30px;">
            <span style="font-size: 28px;">✨</span>
            <h3>Key Features</h3>
        </div>
        """, unsafe_allow_html=True)

        features = [
            ("🎯", "Accurate Predictions", "AI-powered yield predictions for 17+ crop varieties"),
            ("📊", "Top 5 Rankings", "Get ranked recommendations for best crop choices"),
            ("🌱", "Soil Analysis", "Analyze NPK levels, pH, and soil composition"),
            ("🌤️", "Climate Factors", "Consider temperature, humidity, and rainfall"),
        ]

        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-card">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 32px;">{icon}</span>
                    <div>
                        <h4 style="margin: 0; color: #f8fafc; font-size: 18px;">{title}</h4>
                        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 14px;">{desc}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header" style="border: none;">
            <span style="font-size: 28px;">📖</span>
            <h3>How to Use</h3>
        </div>
        """, unsafe_allow_html=True)

        instructions = [
            ("1", "Input Soil Parameters", "Enter nitrogen (N), phosphorus (P), and potassium (K) levels from your soil test results."),
            ("2", "Set Climate Conditions", "Adjust temperature, humidity, pH level, and expected rainfall for your region."),
            ("3", "Select Target Crop", "Choose the crop you want to analyze from our database of 17 crop varieties."),
            ("4", "Get Predictions", "Click 'Predict Yield' to receive AI-generated yield predictions and recommendations."),
            ("5", "Review Results", "View your selected crop's yield prediction and explore top 5 alternative crop options."),
        ]

        for num, title, desc in instructions:
            st.markdown(f"""
            <div class="instruction-item">
                <div class="step-number">{num}</div>
                <div class="instruction-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # CTA Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Predicting Now", use_container_width=True, key="start_btn"):
            st.session_state.page = 'predictor'
            st.rerun()

    # Supported Crops
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header" style="border: none;">
        <span style="font-size: 28px;">🌾</span>
        <h3>Supported Crops</h3>
    </div>
    <p style="color: #94a3b8; margin-bottom: 20px;">Our AI model supports yield predictions for the following crops:</p>
    """, unsafe_allow_html=True)

    # Display crops in a grid
    crop_cols = st.columns(6)
    crop_icons = ["🌾", "🍌", "🥭", "🍊", "🫐", "🍇", "🍉", "🍈", "🍎", "☕", "🌸", "🧵", "🫘", "🫛", "🫘", "🫘", "🥥"]
    
    for idx, (crop, icon) in enumerate(zip(CROPS, crop_icons)):
        with crop_cols[idx % 6]:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: rgba(30, 41, 59, 0.3); 
                        border-radius: 12px; margin: 5px; transition: all 0.3s ease;">
                <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
                <div style="color: #f8fafc; font-weight: 600; font-size: 13px;">{crop.capitalize()}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2026 AI AgriYield Predictor | Built with ❤️ using Streamlit & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


# ================== PREDICTOR PAGE ==================
def show_predictor_page():
    # Back button
    if st.button("← Back to Home", key="back_btn"):
        st.session_state.page = 'welcome'
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="text-align:center; margin-bottom: 40px;">
        <div style="font-size: 50px; margin-bottom: 15px;">🌾</div>
        <h1 class="main-title" style="font-size: 42px;">Smart Yield Predictor</h1>
        <p class="subtitle">Enter your parameters to get AI-powered yield predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout
    left, right = st.columns([1, 1.5], gap="large")

    # Input Panel
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <span style="font-size: 28px;">🌱</span>
            <h3>Soil Parameters</h3>
        </div>
        """, unsafe_allow_html=True)

        N = st.slider("Nitrogen (N)", 0, 150, 90, help="Nitrogen content in soil")
        P = st.slider("Phosphorus (P)", 0, 150, 42, help="Phosphorus content in soil")
        K = st.slider("Potassium (K)", 0, 200, 43, help="Potassium content in soil")

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <span style="font-size: 28px;">🌤️</span>
            <h3>Climate Conditions</h3>
        </div>
        """, unsafe_allow_html=True)

        temp = st.slider("Temperature (°C)", 0.0, 50.0, 26.5, help="Average temperature")
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0, help="Relative humidity")
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5, help="Soil acidity/alkalinity")
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 160.0, help="Average rainfall")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <span style="font-size: 28px;">📅</span>
            <h3>Additional Info</h3>
        </div>
        """, unsafe_allow_html=True)
        
        year = st.number_input("Year", value=2023, help="Year of cultivation")
        target_crop = st.selectbox("🎯 Select Target Crop", CROPS, help="Choose the crop to analyze")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🌿 Predict Yield", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Prediction
    with right:
        if predict_btn:
            predictions = {}

            for crop in CROPS:
                input_df = pd.DataFrame([{
                    "N": N,
                    "P": P,
                    "K": K,
                    "temperature": temp,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                    "Year": year,
                    "crop": crop
                }])

                pred_kg = model.predict(input_df)[0]
                predictions[crop] = pred_kg / 1000

            top5 = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5])
            selected_yield = predictions[target_crop]

            # Success Message
            st.markdown("""
            <div class="success-badge">
                <span>✅</span>
                <span>Prediction Complete</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Main Result Card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<p class="label-text">Estimated Yield for</p>', unsafe_allow_html=True)
            st.markdown(f'<h2 class="crop-name">{target_crop.capitalize()}</h2>', unsafe_allow_html=True)
            st.markdown(f'<div class="yield-value">{selected_yield:.2f} tonnes/ha</div>', unsafe_allow_html=True)
            st.markdown('<p class="label-text">AI-based prediction using machine learning</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Chart Card
            st.markdown('<div class="card" style="margin-top: 25px;">', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <span style="font-size: 28px;">📊</span>
                <h3>Top 5 Crop Predictions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            df = pd.DataFrame(top5.items(), columns=["Crop", "Yield"])

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a2332")
            ax.set_facecolor("#1a2332")

            bars = ax.barh(df["Crop"], df["Yield"], color="#4ade80", height=0.6)

            ax.set_xlabel("Tonnes / hectare", color="#94a3b8", fontsize=12, fontweight="600")
            ax.tick_params(colors="#94a3b8", labelsize=11)
            ax.grid(axis='x', alpha=0.1, color='#4ade80', linestyle='--')

            for spine in ax.spines.values():
                spine.set_visible(False)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(
                    width + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    f"{width:.2f}",
                    ha="left",
                    va="center",
                    color="#4ade80",
                    fontweight="700",
                    fontsize=11
                )

            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # Recommendations Card
            st.markdown('<div class="card" style="margin-top: 25px;">', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <span style="font-size: 28px;">🌟</span>
                <h3>Alternative Recommendations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (crop, val) in enumerate(top5.items(), 1):
                st.markdown(f"""
                <div class="rec-item">
                    <span class="rec-crop">#{i} {crop.capitalize()}</span>
                    <span class="rec-yield">{val:.2f} tonnes/ha</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show placeholder when no prediction
            st.markdown("""
            <div class="card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; text-align: center;">
                <div style="font-size: 80px; margin-bottom: 20px; opacity: 0.3;">🌾</div>
                <h3 style="color: #94a3b8; font-weight: 600;">Ready to Predict</h3>
                <p style="color: #64748b; font-size: 16px; max-width: 400px; margin-top: 10px;">
                    Enter your soil and climate parameters on the left, select your target crop, 
                    and click "Predict Yield" to see AI-powered predictions and recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2026 AI AgriYield Predictor</p>
    </div>
    """, unsafe_allow_html=True)


# ================== PAGE ROUTING ==================
if st.session_state.page == 'welcome':
    show_welcome_page()
else:
    show_predictor_page()