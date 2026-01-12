import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Crop Yield Classifier",
    page_icon="🌱",
    layout="wide"
)

# Load model & encoder
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

# Title
st.title("🌱 Crop Yield Classifier")
st.markdown("### AI-powered crop recommendation based on soil & climate")

st.divider()

# Layout
left, right = st.columns(2)

# ---------------- LEFT: INPUTS ---------------- #
with left:
    st.subheader("🌾 Soil & Climate Inputs")

    N = st.slider("Nitrogen (N)", 0, 140, 90)
    P = st.slider("Phosphorus (P)", 0, 145, 40)
    K = st.slider("Potassium (K)", 0, 205, 40)

    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 200.0)

    predict_btn = st.button("🤖 Get AI Recommendation", use_container_width=True)

# ---------------- RIGHT: RESULTS ---------------- #
with right:
    st.subheader("📊 Prediction Results")

    if predict_btn:
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        probs = model.predict_proba(data)[0]
        crops = le.inverse_transform(range(len(probs)))

        results = pd.DataFrame({
            "Crop": crops,
            "Confidence (%)": probs * 100
        }).sort_values(by="Confidence (%)", ascending=False)

        top3 = results.head(3)
        bottom3 = results.tail(3)

        # 🌟 Best crop
        st.success("🌟 Best Crop Recommendation")
        st.metric(
            label=top3.iloc[0]["Crop"].title(),
            value=f"{top3.iloc[0]['Confidence (%)']:.1f}% confidence"
        )

        st.divider()

        # 🔝 Top 3
        st.markdown("### 🔝 Top 3 Suitable Crops")
        for i, row in top3.iterrows():
            st.write(f"**{row['Crop'].title()}** — {row['Confidence (%)']:.1f}%")

        st.divider()

        # ❌ Bottom 3
        st.markdown("### ❌ Not Recommended Crops")
        for crop in bottom3["Crop"]:
            st.write(f"🚫 {crop.title()}")

    else:
        st.info("👈 Adjust parameters and click **Get AI Recommendation**")


