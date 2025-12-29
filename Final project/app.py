import base64
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import altair as alt
import streamlit as st

# ================== CONFIG ==================
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Crop Vision | AI Yield Predictor",
    layout="wide",
    page_icon="🌱",
)

if "started" not in st.session_state:
    st.session_state.started = False

# ================== BACKGROUND IMAGE (OPTIONAL) ==================
def set_bg_local(image_file: str):
    img_path = Path(image_file)
    if img_path.exists():
        with open(img_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

set_bg_local("welcome_bg.jpg")

# ================== GLOBAL + NEW WELCOME CSS (IMAGE 2 STYLE) ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

header {visibility: hidden;}

/* Welcome page background with farm field feel */
.stApp {
    background: linear-gradient(135deg, rgba(140, 180, 140, 0.3) 0%, rgba(200, 220, 200, 0.4) 100%),
                linear-gradient(to bottom, #a8c5a0 0%, #b8d4b0 50%, #c8e3c0 100%);
}

/* Floating tech decorations */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}

@keyframes float-delayed {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
}

.tech-icon {
    position: fixed;
    color: rgba(255, 255, 255, 0.4);
    font-size: 2rem;
    pointer-events: none;
    z-index: 1;
}

/* HERO CARD - IMAGE 2 STYLE */
.hero-card {
    position: relative;
    background: linear-gradient(135deg, rgba(245, 242, 235, 0.98) 0%, rgba(240, 237, 230, 0.95) 100%);
    backdrop-filter: blur(10px);
    border-radius: 28px;
    padding: 50px 70px 50px 70px;
    max-width: 1100px;
    margin: 6vh auto 0 auto;
    color: #2d4a2d;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15),
                inset 0 1px 0 rgba(255,255,255,0.8);
    overflow: visible;
    border: 1px solid rgba(200, 195, 185, 0.4);
}

/* Decorative agricultural icons around the card */
.deco-icon {
    position: absolute;
    font-size: 3rem;
    color: rgba(60, 80, 50, 0.6);
    pointer-events: none;
    filter: brightness(0.7) contrast(1.2);
}

.icon-brain { top: 80px; left: -120px; }
.icon-cloud { top: 60px; left: -20px; }
.icon-bulb { bottom: 100px; left: -110px; }
.icon-leaf { top: 60px; right: -80px; }
.icon-phone { top: 20px; right: -60px; }
.icon-plant { bottom: 80px; right: -100px; }
.icon-soil { bottom: 120px; right: -30px; }

/* Connecting lines decoration */
.hero-card::before {
    content: "";
    position: absolute;
    top: 50%;
    left: -150px;
    right: -150px;
    height: 1px;
    background: linear-gradient(90deg, 
        transparent 0%, 
        rgba(150, 170, 140, 0.3) 20%, 
        rgba(150, 170, 140, 0.3) 80%, 
        transparent 100%);
    pointer-events: none;
}

/* LOGO AREA - TOP LEFT */
.hero-logo {
    position: absolute;
    left: 28px;
    top: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    color: #5a7a5a;
}

.hero-logo-mark {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6b8e6b, #4a6f4a);
    box-shadow: 0 4px 12px rgba(90, 122, 90, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.hero-kicker {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8a9a8a;
    font-weight: 500;
    margin-bottom: 8px;
    opacity: 0.85;
}

.hero-title {
    font-size: 3.5rem;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 18px;
    color: #2d5a2d;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #5a6a5a;
    margin-bottom: 24px;
    font-weight: 400;
    font-style: italic;
    line-height: 1.5;
}

.hero-tagline {
    font-size: 1rem;
    color: #6a7a6a;
    margin-bottom: 36px;
    line-height: 1.6;
    max-width: 850px;
    margin-left: auto;
    margin-right: auto;
}

/* INFO CHIPS - SOFT COLORS */
.hero-chip-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    font-size: 0.88rem;
    margin-bottom: 32px;
}

.hero-chip {
    padding: 14px 28px;
    border-radius: 24px;
    font-weight: 600;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.6);
    display: flex;
    align-items: center;
    gap: 8px;
}

.hero-chip:nth-child(1) {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    color: #78350f;
}
.hero-chip:nth-child(2) {
    background: linear-gradient(135deg, #bfdbfe, #93c5fd);
    color: #1e3a8a;
}
.hero-chip:nth-child(3) {
    background: linear-gradient(135deg, #fecaca, #fca5a5);
    color: #7f1d1d;
}

.hero-chip:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

/* STATUS ROW - GREEN DOTS */
.hero-status-row {
    display: flex;
    justify-content: center;
    gap: 40px;
    font-size: 0.85rem;
    color: #5a6a5a;
    font-weight: 500;
    margin-top: 8px;
}

.hero-status-item {
    display: flex;
    align-items: center;
    gap: 10px;
}

.hero-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #4a7a4a;
    box-shadow: 0 0 0 3px rgba(74, 122, 74, 0.2);
}

/* FOOTER */
.app-footer {
    text-align: center;
    font-size: 0.8rem;
    color: rgba(90, 110, 90, 0.8);
    margin-top: 40px;
    margin-bottom: 20px;
    font-weight: 500;
}

/* ======= PREDICTION PAGE HEADER BAR ======= */
.pred-header {
    background: linear-gradient(90deg, #1b5e20, #2e7d32, #388e3c);
    padding: 20px 40px;
    margin: -1.5rem -1.5rem 20px -1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.pred-header-left h1 {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0;
    margin-bottom: 4px;
}

.pred-header-left p {
    color: rgba(255,255,255,0.9);
    font-size: 0.9rem;
    margin: 0;
    font-weight: 400;
}

.pred-header-badge {
    background: rgba(255,255,255,0.2);
    color: #ffffff;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.3);
    letter-spacing: 0.05em;
}

/* ======= PREDICTION PAGE LAYOUT ======= */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ================== WELCOME SCREEN - IMAGE 2 STYLE ==================
if not st.session_state.started:
    st.markdown("""
    <div class="tech-icon" style="top: 8%; left: 5%; animation: float 4s ease-in-out infinite;">🌐</div>
    <div class="tech-icon" style="top: 15%; right: 8%; animation: float-delayed 3.5s ease-in-out infinite;">⚡</div>
    <div class="tech-icon" style="bottom: 20%; left: 10%; animation: float 3s ease-in-out infinite;">📊</div>
    <div class="tech-icon" style="bottom: 15%; right: 12%; animation: float-delayed 4.5s ease-in-out infinite;">💡</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hero-card">
        <div class="hero-logo">
            <div class="hero-logo-mark">🌱</div>
            <span>CROP VISION</span>
        </div>
        <div class="deco-icon icon-brain">🧠</div>
        <div class="deco-icon icon-cloud">☁️</div>
        <div class="deco-icon icon-bulb">💡</div>
        <div class="deco-icon icon-leaf">🍃</div>
        <div class="deco-icon icon-phone">📱</div>
        <div class="deco-icon icon-plant">🌾</div>
        <div class="deco-icon icon-soil">🏔️</div>
        <div class="hero-kicker">WELCOME TO</div>
        <div class="hero-title">AgriSense AI</div>
        <div class="hero-subtitle">"Where farmer's wisdom meets machine insight."</div>
        <div class="hero-tagline">Turn soil, weather and crop data into confident yield predictions and smart crop recommendations.</div>
        <div class="hero-chip-row">
            <div class="hero-chip">📊 15 Crops Analyzed</div>
            <div class="hero-chip">☁️ Weather & Soil Aware</div>
            <div class="hero-chip">🌾 Smart Crop Recommendations</div>
        </div>
        <div class="hero-status-row">
            <div class="hero-status-item"><div class="hero-dot"></div><span>Live AI engine</span></div>
            <div class="hero-status-item"><div class="hero-dot"></div><span>Real‑time inputs</span></div>
            <div class="hero-status-item"><div class="hero-dot"></div><span>Smart farm insights</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 0.45, 1])
    with btn_col:
        if st.button("START", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.rerun()

    st.markdown("<div class='app-footer'>Crop Vision • © 2025 Shrthika • AI‑powered yield insights</div>", unsafe_allow_html=True)

# ================== PREDICTION PAGE – MATCHING REFERENCE IMAGE ==================
else:
    st.markdown(
        """
        <style>
        body, .stApp {
            background: linear-gradient(to right, #e3f2fd 0%, #e3f2fd 50%, #e8f5e9 50%, #e8f5e9 100%) !important;
        }
        .pred-shell {
            max-width: 1400px;
            margin: 20px auto;
            background: transparent;
            padding: 0;
        }
        
        /* Left column background - Light Blue */
        div[data-testid="column"]:first-child {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 0px;
        }
        
        /* Right column background - Light Green */
        div[data-testid="column"]:last-child {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 0px;
        }
        
        /* Left panel styling */
        .left-panel {
            background: #ffffff;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
            margin: 12px;
        }
        
        /* Right panel styling */
        .right-panel {
            background: transparent;
            border-radius: 0px;
            padding: 12px;
            box-shadow: none;
            border: none;
        }
        
        .panel-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1b5e20;
            margin-bottom: 8px;
        }
        
        .panel-subtitle {
            font-size: 0.85rem;
            color: #558b2f;
            margin-bottom: 20px;
        }
        
        /* Yield display card - light green */
        .yield-card {
            background: linear-gradient(135deg, #a5d6a7, #81c784);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        .yield-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #ffffff;
            font-weight: 600;
            margin-bottom: 8px;
            opacity: 0.95;
        }
        
        .yield-value {
            font-size: 2.8rem;
            font-weight: 900;
            color: #ffffff;
            margin-bottom: 4px;
            line-height: 1;
        }
        
        .yield-caption {
            font-size: 0.8rem;
            color: #ffffff;
            font-weight: 400;
            opacity: 0.9;
        }
        
        /* Recommendation card - dark green */
        .recom-card {
            background: linear-gradient(135deg, #2e7d32, #388e3c);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            color: white;
        }
        
        .recom-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 12px;
            opacity: 0.95;
        }
        
        .recom-stat {
            font-size: 0.85rem;
            margin-bottom: 8px;
            line-height: 1.6;
        }
        
        /* Crop list styling */
        .crop-list-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #1b5e20;
            margin-bottom: 12px;
        }
        
        .crop-item {
            margin-bottom: 12px;
        }
        
        .crop-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        
        .crop-name {
            font-size: 0.95rem;
            font-weight: 600;
            color: #1b5e20;
        }
        
        .crop-value {
            font-size: 0.85rem;
            color: #558b2f;
            font-weight: 600;
        }
        
        .crop-bar-bg {
            width: 100%;
            height: 10px;
            background: rgba(200, 230, 201, 0.5);
            border-radius: 5px;
            overflow: hidden;
        }
        
        /* original single color kept as base */
        .crop-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #66bb6a, #81c784);
            border-radius: 5px;
            transition: width 0.5s ease;
        }

        /* NEW: alternating bar colors */
        .crop-bar-fill-a {
            height: 100%;
            background: linear-gradient(90deg, #2e7d32, #4caf50);
            border-radius: 5px;
            transition: width 0.5s ease;
        }

        .crop-bar-fill-b {
            height: 100%;
            background: linear-gradient(90deg, #66bb6a, #a5d6a7);
            border-radius: 5px;
            transition: width 0.5s ease;
        }
        
        /* Chart section */
        .chart-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #558b2f;
            margin-top: 20px;
            margin-bottom: 8px;
        }
        
        /* Input styling */
        .stNumberInput label, .stSlider label, .stSelectbox label {
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #1b5e20 !important;
        }
        
        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Green header bar
    st.markdown(
        """
        <div class="pred-header">
            <div class="pred-header-left">
                <h1>PREDICTION OVERVIEW</h1>
                <p>Use your field conditions to estimate yield and reveal the best crops to plant.</p>
            </div>
            <div class="pred-header-badge">LIVE • AI Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main layout
    col_input, col_output = st.columns([1, 1], gap="large")

    # ========== LEFT COLUMN: INPUTS ==========
    with col_input:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-title'>Yield Input Panel</div>"
            "<div class='panel-subtitle'>Soil nutrients and weather features.</div>",
            unsafe_allow_html=True,
        )

        # NPK inputs
        c1, c2 = st.columns(2)
        with c1:
            n = st.number_input(
                "Nitrogen (N)", 
                value=90.0,
                min_value=0.0, 
                max_value=300.0, 
                step=1.0,
            )
        with c2:
            p = st.number_input(
                "Phosphorus (P)",
                value=40.0,
                min_value=0.0, 
                max_value=200.0, 
                step=1.0,
            )

        c3, c4 = st.columns(2)
        with c3:
            k = st.number_input(
                "Potassium (K)",
                value=40.0,
                min_value=0.0, 
                max_value=250.0, 
                step=1.0,
            )

        st.markdown("<hr style='border:none;border-top:1px solid #e0e0e0;margin:16px 0;'/>", unsafe_allow_html=True)
        
        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#1b5e20;margin-bottom:12px;'>"
            "Weather & soil conditions</div>",
            unsafe_allow_html=True,
        )

        # Weather inputs
        w1, w2 = st.columns(2)
        with w1:
            temp = st.slider("Temperature (°C)", 10.0, 45.0, 25.0, 0.5)
            hum = st.slider("Humidity (%)", 0, 100, 80)
        with w2:
            rain = st.slider("Rainfall (mm)", 0.0, 400.0, 200.0, 1.0)
            ph = st.slider("pH Level", 0.0, 14.0, 6.5, 0.1)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        crop_choice = st.selectbox(
            "Select Crop",
            [
                "banana", "chickpea", "coffee", "cotton", "grapes", "jute",
                "lentil", "maize", "mango", "mothbeans", "muskmelon",
                "papaya", "pigeonpeas", "rice", "watermelon",
            ],
        )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # Buttons
        cbtn1, cbtn2 = st.columns([1.4, 1])
        with cbtn1:
            predict_btn = st.button("PREDICT YIELD", use_container_width=True, type="primary")
        with cbtn2:
            reset_btn = st.button("Reset Fields", use_container_width=True)

        if reset_btn:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.started = True
            st.rerun()

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        if st.button("← Back to Welcome"):
            st.session_state.started = False
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== RIGHT COLUMN: OUTPUTS ==========
    with col_output:
        if not predict_btn:
            st.markdown(
                """
                <div class="right-panel" style="display:flex;align-items:center;justify-content:center;min-height:500px;">
                    <div style="text-align:center;font-size:0.95rem;color:#558b2f;">
                        Fill the inputs on the left and click <b>PREDICT YIELD</b><br/>
                        to see your crop yield and smart recommendations.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if n > 300 or p > 200 or k > 250:
                st.error("NPK values must be realistic: N 0–300, P 0–200, K 0–250")
            elif temp < 10 or temp > 45:
                st.error("Temperature should be between 10 and 45°C.")
            else:
                payload = {
                    "N": n,
                    "P": p,
                    "K": k,
                    "temperature": temp,
                    "humidity": hum,
                    "ph": ph,
                    "rainfall": rain,
                    "crop": crop_choice,
                }

                try:
                    resp = requests.post(API_URL, json=payload, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()

                    # Get yields in tons and convert to kilotons
                    res_val_tons = data["usercrop"]["yield"]
                    res_val_kt = res_val_tons / 1000.0
                    
                    recs = data["toprecommendations"]
                    recs_kt = [
                        {
                            "crop": it["crop"],
                            "yield_kt": it["yield"] / 1000.0
                        }
                        for it in recs
                    ]

                    all_yields_kt = [it["yield_kt"] for it in recs_kt] + [res_val_kt]
                    max_yield_kt = max(all_yields_kt)
                    avg_yield_kt = float(np.mean(all_yields_kt))

                    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

                    # Top cards
                    s1, s2 = st.columns(2, gap="medium")
                    with s1:
                        st.markdown(
                            f"""
                            <div class="yield-card">
                                <div class="yield-label">CROP YIELD (IN KILOTONS)</div>
                                <div class="yield-value">{res_val_kt:.4f}</div>
                                <div class="yield-caption">Kilotons per hectare for {crop_choice.capitalize()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with s2:
                        st.markdown(
                            f"""
                            <div class="recom-card">
                                <div class="recom-title">INTELLIGENT CROP RECOM</div>
                                <div class="recom-stat">Top crop yield:<br/><b>{max_yield_kt:.4f} kilotons</b></div>
                                <div class="recom-stat">Avg of top 5:<br/><b>{avg_yield_kt:.4f} kilotons</b></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Crop recommendations list (ALTERNATING BAR COLORS)
                    st.markdown(
                        "<div class='crop-list-title'>Top Recommended Crops</div>",
                        unsafe_allow_html=True,
                    )

                    for idx, item in enumerate(recs_kt):
                        score_kt = item["yield_kt"]
                        pct = (score_kt / max_yield_kt) * 100 if max_yield_kt > 0 else 0
                        bar_class = "crop-bar-fill-a" if idx % 2 == 0 else "crop-bar-fill-b"
                        st.markdown(
                            f"""
                            <div class="crop-item">
                                <div class="crop-item-header">
                                    <span class="crop-name">{item['crop'].capitalize()}</span>
                                    <span class="crop-value">{score_kt:.4f} kilotons</span>
                                </div>
                                <div class="crop-bar-bg">
                                    <div class="{bar_class}" style="width:{pct:.1f}%;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Chart with alternating colors and VALUE LABELS
                    chart_df = (
                        pd.DataFrame(
                            {
                                "Crop": [it["crop"].capitalize() for it in recs_kt],
                                "Yield (kilotons)": [it["yield_kt"] for it in recs_kt],
                            }
                        )
                        .sort_values("Yield (kilotons)", ascending=False)
                        .reset_index(drop=True)
                    )

                    chart_df["color_group"] = np.where(chart_df.index % 2 == 0, "A", "B")

                    st.markdown(
                        "<div class='chart-title'>Yield comparison (top 5)</div>",
                        unsafe_allow_html=True,
                    )

                    # Create the bar chart
                    bars = (
                        alt.Chart(chart_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("Crop:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
                            y=alt.Y("Yield (kilotons):Q", axis=alt.Axis(title="Yield (kilotons)")),
                            color=alt.Color(
                                "color_group:N",
                                scale=alt.Scale(
                                    domain=["A", "B"],
                                    range=["#2e7d32", "#66bb6a"],
                                ),
                                legend=None,
                            ),
                        )
                    )

                    # Create text labels above bars
                    text = (
                        alt.Chart(chart_df)
                        .mark_text(
                            align='center',
                            baseline='bottom',
                            dy=-5,
                            fontSize=11,
                            fontWeight=600,
                            color='#1b5e20'
                        )
                        .encode(
                            x=alt.X("Crop:N", sort=None),
                            y=alt.Y("Yield (kilotons):Q"),
                            text=alt.Text("Yield (kilotons):Q", format=".4f")
                        )
                    )

                    # Combine bars and text
                    chart = (bars + text).properties(height=250)

                    st.altair_chart(chart, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                    st.toast(f"{crop_choice} yield predicted successfully!", icon="✅")

                except requests.exceptions.RequestException:
                    st.error("Backend server offline. Please check your FastAPI connection.")