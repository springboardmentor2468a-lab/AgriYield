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

# ================== TRANSLATIONS ==================
TRANSLATIONS = {
    'English': {
        'title': 'Smart Yield Predictor',
        'subtitle': 'Maximize agricultural output using AI-driven soil & climate analysis',
        'welcome': 'Welcome to AI AgriYield Predictor! 👋',
        'welcome_desc': 'Our advanced machine learning system helps farmers and agricultural professionals make data-driven decisions by predicting crop yields based on soil parameters and climate conditions.',
        'key_features': 'Key Features',
        'accurate_pred': 'Accurate Predictions',
        'accurate_desc': 'AI-powered yield predictions for 17+ crop varieties',
        'top_rankings': 'Top 5 Rankings',
        'rankings_desc': 'Get ranked recommendations for best crop choices',
        'soil_analysis': 'Soil Analysis',
        'soil_desc': 'Analyze NPK levels, pH, and soil composition',
        'climate_factors': 'Climate Factors',
        'climate_desc': 'Consider temperature, humidity, and rainfall',
        'how_to_use': 'How to Use',
        'step1': 'Input Soil Parameters',
        'step1_desc': 'Enter nitrogen (N), phosphorus (P), and potassium (K) levels from your soil test results.',
        'step2': 'Set Climate Conditions',
        'step2_desc': 'Adjust temperature, humidity, pH level, and expected rainfall for your region.',
        'step3': 'Select Target Crop',
        'step3_desc': 'Choose the crop you want to analyze from our database of 17 crop varieties.',
        'step4': 'Get Predictions',
        'step4_desc': 'Click "Predict Yield" to receive AI-generated yield predictions and recommendations.',
        'step5': 'Review Results',
        'step5_desc': 'View your selected crop\'s yield prediction and explore top 5 alternative crop options.',
        'supported_crops': 'Supported Crops',
        'supported_desc': 'Our AI model supports yield predictions for the following crops:',
        'start_btn': '🚀 Start Predicting Now',
        'back_btn': '← Back to Home',
        'soil_params': 'Soil Parameters',
        'climate_cond': 'Climate Conditions',
        'additional_info': 'Additional Info',
        'nitrogen': 'Nitrogen (N)',
        'phosphorus': 'Phosphorus (P)',
        'potassium': 'Potassium (K)',
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'soil_ph': 'Soil pH',
        'rainfall': 'Rainfall (mm)',
        'year': 'Year',
        'select_crop': '🎯 Select Target Crop',
        'predict_btn': '🌿 Predict Yield',
        'prediction_complete': 'Prediction Complete',
        'estimated_yield': 'ESTIMATED YIELD FOR',
        'tonnes_ha': 'tonnes/ha',
        'ai_prediction': 'AI-based prediction using machine learning',
        'top5_predictions': 'Top 5 Crop Predictions',
        'alt_recommendations': 'Alternative Recommendations',
        'ready_predict': 'Ready to Predict',
        'ready_desc': 'Enter your soil and climate parameters on the left, select your target crop, and click "Predict Yield" to see AI-powered predictions and recommendations.',
        'footer': '© 2026 AI AgriYield Predictor | Built with ❤️ using Streamlit & Machine Learning',
        'language': 'Language',
        'nitrogen_help': 'Nitrogen content in soil',
        'phosphorus_help': 'Phosphorus content in soil',
        'potassium_help': 'Potassium content in soil',
        'temp_help': 'Average temperature',
        'humidity_help': 'Relative humidity',
        'ph_help': 'Soil acidity/alkalinity',
        'rainfall_help': 'Average rainfall',
        'year_help': 'Year of cultivation',
        'crop_help': 'Choose the crop to analyze'
    },
    'हिंदी': {
        'title': 'स्मार्ट उपज भविष्यवक्ता',
        'subtitle': 'एआई-संचालित मिट्टी और जलवायु विश्लेषण का उपयोग करके कृषि उत्पादन को अधिकतम करें',
        'welcome': 'एआई एग्रीयील्ड प्रेडिक्टर में आपका स्वागत है! 👋',
        'welcome_desc': 'हमारी उन्नत मशीन लर्निंग प्रणाली किसानों और कृषि पेशेवरों को मिट्टी के मापदंडों और जलवायु स्थितियों के आधार पर फसल की उपज की भविष्यवाणी करके डेटा-संचालित निर्णय लेने में मदद करती है।',
        'key_features': 'मुख्य विशेषताएं',
        'accurate_pred': 'सटीक भविष्यवाणियां',
        'accurate_desc': '17+ फसल किस्मों के लिए एआई-संचालित उपज भविष्यवाणियां',
        'top_rankings': 'शीर्ष 5 रैंकिंग',
        'rankings_desc': 'सर्वोत्तम फसल विकल्पों के लिए रैंक की गई सिफारिशें प्राप्त करें',
        'soil_analysis': 'मिट्टी विश्लेषण',
        'soil_desc': 'एनपीके स्तर, पीएच और मिट्टी की संरचना का विश्लेषण करें',
        'climate_factors': 'जलवायु कारक',
        'climate_desc': 'तापमान, आर्द्रता और वर्षा पर विचार करें',
        'how_to_use': 'उपयोग कैसे करें',
        'step1': 'मिट्टी मापदंड दर्ज करें',
        'step1_desc': 'अपने मिट्टी परीक्षण परिणामों से नाइट्रोजन (N), फास्फोरस (P), और पोटेशियम (K) स्तर दर्ज करें।',
        'step2': 'जलवायु स्थितियां सेट करें',
        'step2_desc': 'अपने क्षेत्र के लिए तापमान, आर्द्रता, पीएच स्तर और अपेक्षित वर्षा को समायोजित करें।',
        'step3': 'लक्ष्य फसल चुनें',
        'step3_desc': '17 फसल किस्मों के हमारे डेटाबेस से उस फसल को चुनें जिसका आप विश्लेषण करना चाहते हैं।',
        'step4': 'भविष्यवाणियां प्राप्त करें',
        'step4_desc': 'एआई-जनित उपज भविष्यवाणियां और सिफारिशें प्राप्त करने के लिए "उपज की भविष्यवाणी करें" पर क्लिक करें।',
        'step5': 'परिणाम समीक्षा करें',
        'step5_desc': 'अपनी चयनित फसल की उपज भविष्यवाणी देखें और शीर्ष 5 वैकल्पिक फसल विकल्पों का अन्वेषण करें।',
        'supported_crops': 'समर्थित फसलें',
        'supported_desc': 'हमारा एआई मॉडल निम्नलिखित फसलों के लिए उपज भविष्यवाणियों का समर्थन करता है:',
        'start_btn': '🚀 अभी भविष्यवाणी शुरू करें',
        'back_btn': '← होम पर वापस जाएं',
        'soil_params': 'मिट्टी मापदंड',
        'climate_cond': 'जलवायु स्थितियां',
        'additional_info': 'अतिरिक्त जानकारी',
        'nitrogen': 'नाइट्रोजन (N)',
        'phosphorus': 'फास्फोरस (P)',
        'potassium': 'पोटेशियम (K)',
        'temperature': 'तापमान (°C)',
        'humidity': 'आर्द्रता (%)',
        'soil_ph': 'मिट्टी pH',
        'rainfall': 'वर्षा (mm)',
        'year': 'वर्ष',
        'select_crop': '🎯 लक्ष्य फसल चुनें',
        'predict_btn': '🌿 उपज की भविष्यवाणी करें',
        'prediction_complete': 'भविष्यवाणी पूर्ण',
        'estimated_yield': 'अनुमानित उपज',
        'tonnes_ha': 'टन/हेक्टेयर',
        'ai_prediction': 'मशीन लर्निंग का उपयोग करके एआई-आधारित भविष्यवाणी',
        'top5_predictions': 'शीर्ष 5 फसल भविष्यवाणियां',
        'alt_recommendations': 'वैकल्पिक सिफारिशें',
        'ready_predict': 'भविष्यवाणी के लिए तैयार',
        'ready_desc': 'बाईं ओर अपने मिट्टी और जलवायु मापदंड दर्ज करें, अपनी लक्ष्य फसल चुनें, और एआई-संचालित भविष्यवाणियां और सिफारिशें देखने के लिए "उपज की भविष्यवाणी करें" पर क्लिक करें।',
        'footer': '© 2026 एआई एग्रीयील्ड प्रेडिक्टर | Streamlit और मशीन लर्निंग का उपयोग करके ❤️ के साथ निर्मित',
        'language': 'भाषा',
        'nitrogen_help': 'मिट्टी में नाइट्रोजन सामग्री',
        'phosphorus_help': 'मिट्टी में फास्फोरस सामग्री',
        'potassium_help': 'मिट्टी में पोटेशियम सामग्री',
        'temp_help': 'औसत तापमान',
        'humidity_help': 'सापेक्ष आर्द्रता',
        'ph_help': 'मिट्टी की अम्लता/क्षारता',
        'rainfall_help': 'औसत वर्षा',
        'year_help': 'खेती का वर्ष',
        'crop_help': 'विश्लेषण के लिए फसल चुनें'
    },
    'ಕನ್ನಡ': {
        'title': 'ಸ್ಮಾರ್ಟ್ ಇಳುವರಿ ಮುನ್ಸೂಚಕ',
        'subtitle': 'AI-ಚಾಲಿತ ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಬಳಸಿಕೊಂಡು ಕೃಷಿ ಉತ್ಪಾದನೆಯನ್ನು ಹೆಚ್ಚಿಸಿ',
        'welcome': 'AI ಅಗ್ರಿಯೀಲ್ಡ್ ಪ್ರೆಡಿಕ್ಟರ್‌ಗೆ ಸ್ವಾಗತ! 👋',
        'welcome_desc': 'ನಮ್ಮ ಸುಧಾರಿತ ಯಂತ್ರ ಕಲಿಕೆ ವ್ಯವಸ್ಥೆಯು ರೈತರು ಮತ್ತು ಕೃಷಿ ವೃತ್ತಿಪರರಿಗೆ ಮಣ್ಣಿನ ನಿಯತಾಂಕಗಳು ಮತ್ತು ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳ ಆಧಾರದ ಮೇಲೆ ಬೆಳೆ ಇಳುವರಿಯನ್ನು ಊಹಿಸುವ ಮೂಲಕ ಡೇಟಾ-ಚಾಲಿತ ನಿರ್ಧಾರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.',
        'key_features': 'ಪ್ರಮುಖ ವೈಶಿಷ್ಟ್ಯಗಳು',
        'accurate_pred': 'ನಿಖರ ಮುನ್ಸೂಚನೆಗಳು',
        'accurate_desc': '17+ ಬೆಳೆ ಪ್ರಭೇದಗಳಿಗೆ AI-ಚಾಲಿತ ಇಳುವರಿ ಮುನ್ಸೂಚನೆಗಳು',
        'top_rankings': 'ಉನ್ನತ 5 ಶ್ರೇಯಾಂಕಗಳು',
        'rankings_desc': 'ಉತ್ತಮ ಬೆಳೆ ಆಯ್ಕೆಗಳಿಗಾಗಿ ಶ್ರೇಯಾಂಕಿತ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ',
        'soil_analysis': 'ಮಣ್ಣು ವಿಶ್ಲೇಷಣೆ',
        'soil_desc': 'NPK ಮಟ್ಟಗಳು, pH ಮತ್ತು ಮಣ್ಣಿನ ಸಂಯೋಜನೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ',
        'climate_factors': 'ಹವಾಮಾನ ಅಂಶಗಳು',
        'climate_desc': 'ತಾಪಮಾನ, ಆರ್ದ್ರತೆ ಮತ್ತು ಮಳೆಯನ್ನು ಪರಿಗಣಿಸಿ',
        'how_to_use': 'ಹೇಗೆ ಬಳಸುವುದು',
        'step1': 'ಮಣ್ಣಿನ ನಿಯತಾಂಕಗಳನ್ನು ನಮೂದಿಸಿ',
        'step1_desc': 'ನಿಮ್ಮ ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಫಲಿತಾಂಶಗಳಿಂದ ಸಾರಜನಕ (N), ರಂಜಕ (P), ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಮ್ (K) ಮಟ್ಟಗಳನ್ನು ನಮೂದಿಸಿ.',
        'step2': 'ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳನ್ನು ಹೊಂದಿಸಿ',
        'step2_desc': 'ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ತಾಪಮಾನ, ಆರ್ದ್ರತೆ, pH ಮಟ್ಟ ಮತ್ತು ನಿರೀಕ್ಷಿತ ಮಳೆಯನ್ನು ಸರಿಹೊಂದಿಸಿ.',
        'step3': 'ಗುರಿ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'step3_desc': '17 ಬೆಳೆ ಪ್ರಭೇದಗಳ ನಮ್ಮ ಡೇಟಾಬೇಸ್‌ನಿಂದ ನೀವು ವಿಶ್ಲೇಷಿಸಲು ಬಯಸುವ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ.',
        'step4': 'ಮುನ್ಸೂಚನೆಗಳನ್ನು ಪಡೆಯಿರಿ',
        'step4_desc': 'AI-ರಚಿತ ಇಳುವರಿ ಮುನ್ಸೂಚನೆಗಳು ಮತ್ತು ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಲು "ಇಳುವರಿಯನ್ನು ಊಹಿಸಿ" ಕ್ಲಿಕ್ ಮಾಡಿ.',
        'step5': 'ಫಲಿತಾಂಶಗಳನ್ನು ಪರಿಶೀಲಿಸಿ',
        'step5_desc': 'ನಿಮ್ಮ ಆಯ್ದ ಬೆಳೆಯ ಇಳುವರಿ ಮುನ್ಸೂಚನೆಯನ್ನು ವೀಕ್ಷಿಸಿ ಮತ್ತು ಉನ್ನತ 5 ಪರ್ಯಾಯ ಬೆಳೆ ಆಯ್ಕೆಗಳನ್ನು ಅನ್ವೇಷಿಸಿ.',
        'supported_crops': 'ಬೆಂಬಲಿತ ಬೆಳೆಗಳು',
        'supported_desc': 'ನಮ್ಮ AI ಮಾದರಿಯು ಈ ಕೆಳಗಿನ ಬೆಳೆಗಳಿಗೆ ಇಳುವರಿ ಮುನ್ಸೂಚನೆಗಳನ್ನು ಬೆಂಬಲಿಸುತ್ತದೆ:',
        'start_btn': '🚀 ಈಗ ಮುನ್ಸೂಚನೆ ಪ್ರಾರಂಭಿಸಿ',
        'back_btn': '← ಮುಖಪುಟಕ್ಕೆ ಹಿಂತಿರುಗಿ',
        'soil_params': 'ಮಣ್ಣಿನ ನಿಯತಾಂಕಗಳು',
        'climate_cond': 'ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳು',
        'additional_info': 'ಹೆಚ್ಚುವರಿ ಮಾಹಿತಿ',
        'nitrogen': 'ಸಾರಜನಕ (N)',
        'phosphorus': 'ರಂಜಕ (P)',
        'potassium': 'ಪೊಟ್ಯಾಸಿಯಮ್ (K)',
        'temperature': 'ತಾಪಮಾನ (°C)',
        'humidity': 'ಆರ್ದ್ರತೆ (%)',
        'soil_ph': 'ಮಣ್ಣು pH',
        'rainfall': 'ಮಳೆ (mm)',
        'year': 'ವರ್ಷ',
        'select_crop': '🎯 ಗುರಿ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
        'predict_btn': '🌿 ಇಳುವರಿಯನ್ನು ಊಹಿಸಿ',
        'prediction_complete': 'ಮುನ್ಸೂಚನೆ ಪೂರ್ಣಗೊಂಡಿದೆ',
        'estimated_yield': 'ಅಂದಾಜು ಇಳುವರಿ',
        'tonnes_ha': 'ಟನ್/ಹೆಕ್ಟೇರ್',
        'ai_prediction': 'ಯಂತ್ರ ಕಲಿಕೆಯನ್ನು ಬಳಸಿಕೊಂಡು AI-ಆಧಾರಿತ ಮುನ್ಸೂಚನೆ',
        'top5_predictions': 'ಉನ್ನತ 5 ಬೆಳೆ ಮುನ್ಸೂಚನೆಗಳು',
        'alt_recommendations': 'ಪರ್ಯಾಯ ಶಿಫಾರಸುಗಳು',
        'ready_predict': 'ಮುನ್ಸೂಚನೆಗೆ ಸಿದ್ಧ',
        'ready_desc': 'ಎಡಭಾಗದಲ್ಲಿ ನಿಮ್ಮ ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ನಿಯತಾಂಕಗಳನ್ನು ನಮೂದಿಸಿ, ನಿಮ್ಮ ಗುರಿ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ, ಮತ್ತು AI-ಚಾಲಿತ ಮುನ್ಸೂಚನೆಗಳು ಮತ್ತು ಶಿಫಾರಸುಗಳನ್ನು ನೋಡಲು "ಇಳುವರಿಯನ್ನು ಊಹಿಸಿ" ಕ್ಲಿಕ್ ಮಾಡಿ.',
        'footer': '© 2026 AI ಅಗ್ರಿಯೀಲ್ಡ್ ಪ್ರೆಡಿಕ್ಟರ್ | Streamlit ಮತ್ತು ಯಂತ್ರ ಕಲಿಕೆಯನ್ನು ಬಳಸಿ ❤️ ಜೊತೆಗೆ ನಿರ್ಮಿಸಲಾಗಿದೆ',
        'language': 'ಭಾಷೆ',
        'nitrogen_help': 'ಮಣ್ಣಿನಲ್ಲಿ ಸಾರಜನಕ ಅಂಶ',
        'phosphorus_help': 'ಮಣ್ಣಿನಲ್ಲಿ ರಂಜಕ ಅಂಶ',
        'potassium_help': 'ಮಣ್ಟಿಗೆ ಪೊಟ್ಯಾಸಿಯಮ್ ಅಂಶ',
        'temp_help': 'ಸರಾಸರಿ ತಾಪಮಾನ',
        'humidity_help': 'ಸಾಪೇಕ್ಷ ಆರ್ದ್ರತೆ',
        'ph_help': 'ಮಣ್ಣಿನ ಆಮ್ಲತೆ/ಕ್ಷಾರತೆ',
        'rainfall_help': 'ಸರಾಸರಿ ಮಳೆ',
        'year_help': 'ಕೃಷಿಯ ವರ್ಷ',
        'crop_help': 'ವಿಶ್ಲೇಷಿಸಲು ಬೆಳೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ'
    },
    'తెలుగు': {
        'title': 'స్మార్ట్ దిగుబడి అంచనా',
        'subtitle': 'AI-ఆధారిత మట్టి మరియు వాతావరణ విశ్లేషణను ఉపయోగించి వ్యవసాయ ఉత్పత్తిని పెంచండి',
        'welcome': 'AI అగ్రిఈల్డ్ ప్రిడిక్టర్‌కు స్వాగతం! 👋',
        'welcome_desc': 'మా అధునాతన మెషిన్ లెర్నింగ్ సిస్టమ్ రైతులు మరియు వ్యవసాయ నిపుణులకు మట్టి పారామితులు మరియు వాతావరణ పరిస్థితుల ఆధారంగా పంట దిగుబడిని అంచనా వేయడం ద్వారా డేటా-ఆధారిత నిర్ణయాలు తీసుకోవడంలో సహాయపడుతుంది.',
        'key_features': 'ప్రధాన లక్షణాలు',
        'accurate_pred': 'ఖచ్చితమైన అంచనాలు',
        'accurate_desc': '17+ పంట రకాలకు AI-ఆధారిత దిగుబడి అంచనాలు',
        'top_rankings': 'టాప్ 5 ర్యాంకింగ్లు',
        'rankings_desc': 'ఉత్తమ పంట ఎంపికలకు ర్యాంక్ చేయబడిన సిఫార్సులను పొందండి',
        'soil_analysis': 'మట్టి విశ్లేషణ',
        'soil_desc': 'NPK స్థాయిలు, pH మరియు మట్టి కూర్పును విశ్లేషించండి',
        'climate_factors': 'వాతావరణ కారకాలు',
        'climate_desc': 'ఉష్ణోగ్రత, ఆర్ద్రత మరియు వర్షపాతాన్ని పరిగణించండి',
        'how_to_use': 'ఎలా ఉపయోగించాలి',
        'step1': 'మట్టి పారామితులను నమోదు చేయండి',
        'step1_desc': 'మీ మట్టి పరీక్ష ఫలితాల నుండి నైట్రోజన్ (N), భాస్వరం (P) మరియు పొటాషియం (K) స్థాయిలను నమోదు చేయండి.',
        'step2': 'వాతావరణ పరిస్థితులను సెట్ చేయండి',
        'step2_desc': 'మీ ప్రాంతానికి ఉష్ణోగ్రత, ఆర్ద్రత, pH స్థాయి మరియు నిరీక్షిత వర్షపాతాన్ని సర్దుబాటు చేయండి.',
        'step3': 'లక్ష్య పంటను ఎంచుకోండి',
        'step3_desc': '17 పంట రకాల మా డేటాబేస్ నుండి మీరు విశ్లేషించాలనుకుంటున్న పంటను ఎంచుకోండి.',
        'step4': 'అంచనాలను పొందండి',
        'step4_desc': 'AI-జనరేటెడ్ దిగుబడి అంచనాలు మరియు సిఫార్సులను పొందడానికి "దిగుబడిని అంచనా వేయండి" క్లిక్ చేయండి.',
        'step5': 'ఫలితాలను సమీక్షించండి',
        'step5_desc': 'మీ ఎంచుకున్న పంట దిగుబడి అంచనాను వీక్షించండి మరియు టాప్ 5 ప్రత్యామ్నాయ పంట ఎంపికలను అన్వేషించండి.',
        'supported_crops': 'సపోర్ట్ చేయబడిన పంటలు',
        'supported_desc': 'మా AI మోడల్ క్రింది పంటలకు దిగుబడి అంచనాలను సపోర్ట్ చేస్తుంది:',
        'start_btn': '🚀 ఇప్పుడే అంచనా ప్రారంభించండి',
        'back_btn': '← హోమ్‌కు తిరిగి వెళ్ళండి',
        'soil_params': 'మట్టి పారామితులు',
        'climate_cond': 'వాతావరణ పరిస్థితులు',
        'additional_info': 'అదనపు సమాచారం',
        'nitrogen': 'నైట్రోజన్ (N)',
        'phosphorus': 'భాస్వరం (P)',
        'potassium': 'పొటాషియం (K)',
        'temperature': 'ఉష్ణోగ్రత (°C)',
        'humidity': 'ఆర్ద్రత (%)',
        'soil_ph': 'మట్టి pH',
        'rainfall': 'వర్షపాతం (mm)',
        'year': 'సంవత్సరం',
        'select_crop': '🎯 లక్ష్య పంటను ఎంచుకోండి',
        'predict_btn': '🌿 దిగుబడిని అంచనా వేయండి',
        'prediction_complete': 'అంచనా పూర్తయింది',
        'estimated_yield': 'అంచనా దిగుబడి',
        'tonnes_ha': 'టన్నులు/హెక్టారు',
        'ai_prediction': 'మెషిన్ లెర్నింగ్ ఉపయోగించి AI-ఆధారిత అంచనా',
        'top5_predictions': 'టాప్ 5 పంట అంచనాలు',
        'alt_recommendations': 'ప్రత్యామ్నాయ సిఫార్సులు',
        'ready_predict': 'అంచనా వేయడానికి సిద్ధంగా ఉన్నారు',
        'ready_desc': 'ఎడమవైపు మీ మట్టి మరియు వాతావరణ పారామితులను నమోదు చేయండి, మీ లక్ష్య పంటను ఎంచుకోండి మరియు AI-ఆధారిత అంచనాలు మరియు సిఫార్సులను చూడటానికి "దిగుబడిని అంచనా వేయండి" క్లిక్ చేయండి.',
        'footer': '© 2026 AI అగ్రిఈల్డ్ ప్రిడిక్టర్ | Streamlit మరియు మెషిన్ లెర్నింగ్ ఉపయోగించి ❤️ తో నిర్మించబడింది',
        'language': 'భాష',
        'nitrogen_help': 'మట్టిలో నైట్రోజన్ కంటెంట్',
        'phosphorus_help': 'మట్టిలో భాస్వరం కంటెంట్',
        'potassium_help': 'మట్టిలో పొటాషియం కంటెంట్',
        'temp_help': 'సగటు ఉష్ణోగ్రత',
        'humidity_help': 'సాపేక్ష ఆర్ద్రత',
        'ph_help': 'మట్టి యాసిడిటీ/ఆల్కలినిటీ',
        'rainfall_help': 'సగటు వర్షపాతం',
        'year_help': 'సాగు సంవత్సరం',
        'crop_help': 'విశ్లేషించడానికి పంటను ఎంచుకోండి'
    },
    'اردو': {
        'title': 'سمارٹ پیداوار پیش گو',
        'subtitle': 'AI پر مبنی مٹی اور آب و ہوا کے تجزیے کا استعمال کرتے ہوئے زرعی پیداوار کو زیادہ سے زیادہ کریں',
        'welcome': 'AI ایگری یلڈ پریڈکٹر میں خوش آمدید! 👋',
        'welcome_desc': 'ہمارا جدید مشین لرننگ نظام کسانوں اور زرعی پیشہ ور افراد کو مٹی کے پیرامیٹرز اور آب و ہوا کی صورتحال کی بنیاد پر فصل کی پیداوار کی پیشین گوئی کرکے ڈیٹا پر مبنی فیصلے کرنے میں مدد کرتا ہے۔',
        'key_features': 'اہم خصوصیات',
        'accurate_pred': 'درست پیش گوئیاں',
        'accurate_desc': '17+ فصل کی اقسام کے لیے AI پر مبنی پیداوار کی پیش گوئیاں',
        'top_rankings': 'ٹاپ 5 درجہ بندی',
        'rankings_desc': 'بہترین فصل کے انتخاب کے لیے درجہ بندی شدہ سفارشات حاصل کریں',
        'soil_analysis': 'مٹی کا تجزیہ',
        'soil_desc': 'NPK کی سطح، pH اور مٹی کی ساخت کا تجزیہ کریں',
        'climate_factors': 'آب و ہوا کے عوامل',
        'climate_desc': 'درجہ حرارت، نمی اور بارش پر غور کریں',
        'how_to_use': 'استعمال کیسے کریں',
        'step1': 'مٹی کے پیرامیٹرز درج کریں',
        'step1_desc': 'اپنے مٹی کے ٹیسٹ کے نتائج سے نائٹروجن (N)، فاسفورس (P)، اور پوٹاشیم (K) کی سطح درج کریں۔',
        'step2': 'آب و ہوا کی صورتحال مقرر کریں',
        'step2_desc': 'اپنے علاقے کے لیے درجہ حرارت، نمی، pH کی سطح اور متوقع بارش کو ایڈجسٹ کریں۔',
        'step3': 'ہدف فصل منتخب کریں',
        'step3_desc': '17 فصل کی اقسام کے ہمارے ڈیٹا بیس سے وہ فصل منتخب کریں جس کا آپ تجزیہ کرنا چاہتے ہیں۔',
        'step4': 'پیش گوئیاں حاصل کریں',
        'step4_desc': 'AI سے تیار کردہ پیداوار کی پیش گوئیاں اور سفارشات حاصل کرنے کے لیے "پیداوار کی پیش گوئی کریں" پر کلک کریں۔',
        'step5': 'نتائج کا جائزہ لیں',
        'step5_desc': 'اپنی منتخب کردہ فصل کی پیداوار کی پیش گوئی دیکھیں اور ٹاپ 5 متبادل فصل کے اختیارات کو دریافت کریں۔',
        'supported_crops': 'معاون فصلیں',
        'supported_desc': 'ہمارا AI ماڈل ان فصلوں کے لیے پیداوار کی پیش گوئیوں کی حمایت کرتا ہے:',
        'start_btn': '🚀 ابھی پیش گوئی شروع کریں',
        'back_btn': '← ہوم پر واپس جائیں',
        'soil_params': 'مٹی کے پیرامیٹرز',
        'climate_cond': 'آب و ہوا کی صورتحال',
        'additional_info': 'اضافی معلومات',
        'nitrogen': 'نائٹروجن (N)',
        'phosphorus': 'فاسفورس (P)',
        'potassium': 'پوٹاشیم (K)',
        'temperature': 'درجہ حرارت (°C)',
        'humidity': 'نمی (%)',
        'soil_ph': 'مٹی کا pH',
        'rainfall': 'بارش (mm)',
        'year': 'سال',
        'select_crop': '🎯 ہدف فصل منتخب کریں',
        'predict_btn': '🌿 پیداوار کی پیش گوئی کریں',
        'prediction_complete': 'پیش گوئی مکمل',
        'estimated_yield': 'تخمینی پیداوار',
        'tonnes_ha': 'ٹن/ہیکٹر',
        'ai_prediction': 'مشین لرننگ کا استعمال کرتے ہوئے AI پر مبنی پیش گوئی',
        'top5_predictions': 'ٹاپ 5 فصل کی پیش گوئیاں',
        'alt_recommendations': 'متبادل سفارشات',
        'ready_predict': 'پیش گوئی کے لیے تیار',
        'ready_desc': 'بائیں جانب اپنے مٹی اور آب و ہوا کے پیرامیٹرز درج کریں، اپنی ہدف فصل منتخب کریں، اور AI پر مبنی پیش گوئیاں اور سفارشات دیکھنے کے لیے "پیداوار کی پیش گوئی کریں" پر کلک کریں۔',
        'footer': '© 2026 AI ایگری یلڈ پریڈکٹر | سٹریم لٹ اور مشین لرننگ کا استعمال کرتے ہوئے ❤️ کے ساتھ بنایا گیا',
        'language': 'زبان',
        'nitrogen_help': 'مٹی میں نائٹروجن کا مواد',
        'phosphorus_help': 'مٹی میں فاسفورس کا مواد',
        'potassium_help': 'مٹی میں پوٹاشیم کا مواد',
        'temp_help': 'اوسط درجہ حرارت',
        'humidity_help': 'نسبتی نمی',
        'ph_help': 'مٹی کی تیزابیت/الکلائنٹی',
        'rainfall_help': 'اوسط بارش',
        'year_help': 'کاشت کا سال',
        'crop_help': 'تجزیہ کے لیے فصل منتخب کریں'
    }
}

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

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

/* Language selector */
.language-selector {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: rgba(30, 41, 59, 0.9);
    padding: 8px 15px;
    border-radius: 10px;
    border: 1px solid rgba(74, 222, 128, 0.2);
}

.language-selector select {
    background: transparent !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 5px 10px !important;
    font-size: 14px !important;
    outline: none !important;
}

/* RTL support for Urdu */
[dir="rtl"] {
    text-align: right !important;
    direction: rtl !important;
}

[dir="rtl"] .section-header {
    flex-direction: row-reverse;
}

[dir="rtl"] .instruction-item {
    flex-direction: row-reverse;
    border-left: none;
    border-right: 3px solid #4ade80;
}

[dir="rtl"] .feature-card {
    border-left: none;
    border-right: 4px solid #4ade80;
}

[dir="rtl"] .rec-item {
    border-left: none;
    border-right: 4px solid #4ade80;
}

[dir="rtl"] .step-number {
    margin-right: 0;
    margin-left: 15px;
}
</style>
""", unsafe_allow_html=True)

# ================== HELPER FUNCTIONS ==================
def t(key):
    """Get translation for current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

def show_language_selector():
    """Display language selector in top right"""
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.markdown('<div class="language-selector">', unsafe_allow_html=True)
            
            languages = ['English', 'हिंदी', 'ಕನ್ನಡ', 'తెలుగు', 'اردو']
            selected_lang = st.selectbox(
                t('language'),
                languages,
                index=languages.index(st.session_state.language),
                key="lang_select",
                label_visibility="collapsed"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if selected_lang != st.session_state.language:
                st.session_state.language = selected_lang
                st.rerun()

def apply_rtl_for_urdu():
    """Apply RTL styling for Urdu language"""
    if st.session_state.language == 'اردو':
        st.markdown("""
        <style>
        .stApp * {
            text-align: right !important;
            direction: rtl !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        
        .main-title {
            background: linear-gradient(135deg, #22c55e 0%, #4ade80 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        
        .section-header {
            flex-direction: row-reverse !important;
        }
        
        .instruction-item {
            flex-direction: row-reverse !important;
            border-left: none !important;
            border-right: 3px solid #4ade80 !important;
        }
        
        .feature-card {
            border-left: none !important;
            border-right: 4px solid #4ade80 !important;
        }
        
        .rec-item {
            border-left: none !important;
            border-right: 4px solid #4ade80 !important;
        }
        
        .step-number {
            margin-right: 0 !important;
            margin-left: 15px !important;
        }
        
        .rec-item {
            flex-direction: row-reverse !important;
        }
        </style>
        """, unsafe_allow_html=True)

# ================== WELCOME PAGE ==================
def show_welcome_page():
    # Language selector
    show_language_selector()
    apply_rtl_for_urdu()
    
    # Header
    st.markdown(f"""
    <div style="text-align:center; margin-bottom: 50px;{' direction: rtl;' if st.session_state.language == 'اردو' else ''}">
        <div style="font-size: 70px; margin-bottom: 20px;">🌾</div>
        <h1 class="main-title">{t('title')}</h1>
        <p class="subtitle">{t('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 30px;{' direction: rtl;' if st.session_state.language == 'اردو' else ''}">
            <h2 style="font-size: 32px; color: #4ade80;">{t('welcome')}</h2>
            <p style="color: #94a3b8; font-size: 16px; line-height: 1.8;">
                {t('welcome_desc')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Features
        st.markdown(f"""
        <div class="section-header" style="border: none; margin-top: 30px;{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
            <span style="font-size: 28px;">✨</span>
            <h3>{t('key_features')}</h3>
        </div>
        """, unsafe_allow_html=True)

        features = [
            ("🎯", t('accurate_pred'), t('accurate_desc')),
            ("📊", t('top_rankings'), t('rankings_desc')),
            ("🌱", t('soil_analysis'), t('soil_desc')),
            ("🌤️", t('climate_factors'), t('climate_desc')),
        ]

        for icon, title, desc in features:
            st.markdown(f"""
            <div class="feature-card" style="{'border-left: none; border-right: 4px solid #4ade80;' if st.session_state.language == 'اردو' else ''}">
                <div style="display: flex; align-items: center; gap: 15px;{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
                    <span style="font-size: 32px;">{icon}</span>
                    <div style="{'text-align: right;' if st.session_state.language == 'اردو' else ''}">
                        <h4 style="margin: 0; color: #f8fafc; font-size: 18px;">{title}</h4>
                        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 14px;">{desc}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-header" style="border: none;{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
            <span style="font-size: 28px;">📖</span>
            <h3>{t('how_to_use')}</h3>
        </div>
        """, unsafe_allow_html=True)

        instructions = [
            ("1", t('step1'), t('step1_desc')),
            ("2", t('step2'), t('step2_desc')),
            ("3", t('step3'), t('step3_desc')),
            ("4", t('step4'), t('step4_desc')),
            ("5", t('step5'), t('step5_desc')),
        ]

        for num, title, desc in instructions:
            st.markdown(f"""
            <div class="instruction-item" style="{'flex-direction: row-reverse; border-left: none; border-right: 3px solid #4ade80;' if st.session_state.language == 'اردو' else ''}">
                <div class="step-number" style="{'margin-right: 0; margin-left: 15px;' if st.session_state.language == 'اردو' else ''}">{num}</div>
                <div class="instruction-content" style="{'text-align: right;' if st.session_state.language == 'اردو' else ''}">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # CTA Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t('start_btn'), use_container_width=True, key="start_btn"):
            st.session_state.page = 'predictor'
            st.rerun()

    # Supported Crops
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-header" style="border: none;{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
        <span style="font-size: 28px;">🌾</span>
        <h3>{t('supported_crops')}</h3>
    </div>
    <p style="color: #94a3b8; margin-bottom: 20px;{' text-align: right;' if st.session_state.language == 'اردو' else ''}">{t('supported_desc')}</p>
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
    st.markdown(f"""
    <div class="footer" style="{'direction: rtl;' if st.session_state.language == 'اردو' else ''}">
        <p>{t('footer')}</p>
    </div>
    """, unsafe_allow_html=True)


# ================== PREDICTOR PAGE ==================
def show_predictor_page():
    # Language selector
    show_language_selector()
    apply_rtl_for_urdu()
    
    # Back button
    if st.button(t('back_btn'), key="back_btn"):
        st.session_state.page = 'welcome'
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div style="text-align:center; margin-bottom: 40px;{' direction: rtl;' if st.session_state.language == 'اردو' else ''}">
        <div style="font-size: 50px; margin-bottom: 15px;">🌾</div>
        <h1 class="main-title" style="font-size: 42px;">{t('title')}</h1>
        <p class="subtitle">{t('subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Layout
    if st.session_state.language == 'اردو':
        # For Urdu, we need to reverse the columns for RTL
        left, right = st.columns([1.5, 1], gap="large")
    else:
        left, right = st.columns([1, 1.5], gap="large")

    # Input Panel
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="section-header" style="{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
            <span style="font-size: 28px;">🌱</span>
            <h3>{t('soil_params')}</h3>
        </div>
        """, unsafe_allow_html=True)

        N = st.slider(t('nitrogen'), 0, 150, 90, help=t('nitrogen_help'))
        P = st.slider(t('phosphorus'), 0, 150, 42, help=t('phosphorus_help'))
        K = st.slider(t('potassium'), 0, 200, 43, help=t('potassium_help'))

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="section-header" style="{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
            <span style="font-size: 28px;">🌤️</span>
            <h3>{t('climate_cond')}</h3>
        </div>
        """, unsafe_allow_html=True)

        temp = st.slider(t('temperature'), 0.0, 50.0, 26.5, help=t('temp_help'))
        humidity = st.slider(t('humidity'), 0.0, 100.0, 80.0, help=t('humidity_help'))
        ph = st.slider(t('soil_ph'), 3.0, 10.0, 6.5, help=t('ph_help'))
        rainfall = st.slider(t('rainfall'), 0.0, 300.0, 160.0, help=t('rainfall_help'))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="section-header" style="{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
            <span style="font-size: 28px;">📅</span>
            <h3>{t('additional_info')}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        year = st.number_input(t('year'), value=2023, help=t('year_help'))
        target_crop = st.selectbox(t('select_crop'), CROPS, help=t('crop_help'))

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(t('predict_btn'), use_container_width=True)
        
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
            st.markdown(f"""
            <div class="success-badge" style="{'flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
                <span>✅</span>
                <span>{t('prediction_complete')}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Main Result Card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="label-text" style="{"text-align: right;" if st.session_state.language == "اردو" else ""}">{t("estimated_yield")}</p>', unsafe_allow_html=True)
            st.markdown(f'<h2 class="crop-name" style="{"text-align: right;" if st.session_state.language == "اردو" else ""}">{target_crop.capitalize()}</h2>', unsafe_allow_html=True)
            st.markdown(f'<div class="yield-value" style="{"text-align: right;" if st.session_state.language == "اردو" else ""}">{selected_yield:.2f} {t("tonnes_ha")}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="label-text" style="{"text-align: right;" if st.session_state.language == "اردو" else ""}">{t("ai_prediction")}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Chart Card
            st.markdown('<div class="card" style="margin-top: 25px;">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="section-header" style="{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
                <span style="font-size: 28px;">📊</span>
                <h3>{t('top5_predictions')}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            df = pd.DataFrame(top5.items(), columns=["Crop", "Yield"])

            fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a2332")
            ax.set_facecolor("#1a2332")

            bars = ax.barh(df["Crop"], df["Yield"], color="#4ade80", height=0.6)

            ax.set_xlabel(f"{t('tonnes_ha')}", color="#94a3b8", fontsize=12, fontweight="600")
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
            st.markdown(f"""
            <div class="section-header" style="{' flex-direction: row-reverse;' if st.session_state.language == 'اردو' else ''}">
                <span style="font-size: 28px;">🌟</span>
                <h3>{t('alt_recommendations')}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i, (crop, val) in enumerate(top5.items(), 1):
                st.markdown(f"""
                <div class="rec-item" style="{'flex-direction: row-reverse; border-left: none; border-right: 4px solid #4ade80;' if st.session_state.language == 'اردو' else ''}">
                    <span class="rec-crop">#{i} {crop.capitalize()}</span>
                    <span class="rec-yield">{val:.2f} {t('tonnes_ha')}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show placeholder when no prediction
            st.markdown(f"""
            <div class="card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; text-align: center;">
                <div style="font-size: 80px; margin-bottom: 20px; opacity: 0.3;">🌾</div>
                <h3 style="color: #94a3b8; font-weight: 600;">{t('ready_predict')}</h3>
                <p style="color: #64748b; font-size: 16px; max-width: 400px; margin-top: 10px;">
                    {t('ready_desc')}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="footer" style="{'direction: rtl;' if st.session_state.language == 'اردو' else ''}">
        <p>{t('footer')}</p>
    </div>
    """, unsafe_allow_html=True)


# ================== PAGE ROUTING ==================
if st.session_state.page == 'welcome':
    show_welcome_page()
else:
    show_predictor_page()