
# 🌾 AgriYield Predictor  
### AI-Powered Crop Production Prediction & Top-5 Crop Recommendation System

AgriYield Predictor is an end-to-end **machine learning–based agriculture analytics project** that predicts crop production using soil nutrients and weather parameters.  
The system also recommends the **Top-5 most suitable crops** to maximize yield under given environmental conditions.

This project is designed as an **internship-ready, research-oriented, and deployment-ready ML solution** for smart farming.

---

## 🚀 Project Objective

To assist farmers, agricultural researchers, and planners in making **data-driven crop decisions** by leveraging historical crop production data, soil nutrients, and weather conditions.

---

## 🧠 Key Features

- 📊 Crop production prediction using **Random Forest Regression**
- 🌱 Soil parameters: **Nitrogen (N), Phosphorus (P), Potassium (K), pH**
- 🌦️ Weather parameters: **Temperature, Humidity, Rainfall**
- 🏆 **Top-5 crop recommendations** based on predicted yield
- 📈 Bar-graph visualization of predicted production
- 🔁 Complete ML pipeline:
  
  **EDA → Feature Engineering → Model Training → Prediction → Visualization**

---

## 📊 Dataset Sources

### 1. FAO Crop Production Dataset  
Global crop production statistics from FAO.

🔗 https://www.fao.org/faostat/en/#data/QCL

---

### 2. Crop Recommendation Dataset (Kaggle)  
Soil and weather parameters mapped to crop suitability.

🔗 https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

---

### 3. Government & Weather Data Sources  
- Indian Open Agriculture Data  
  🔗 https://www.data.gov.in/sector/agriculture  

- US Weather Events Dataset  
  🔗 https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events  

- NOAA Climate Data  
  🔗 https://www.ncei.noaa.gov/

---

## 🏗️ System Architecture

<p align="center">
  <img width="975" height="692" alt="image" src="https://github.com/user-attachments/assets/2758c8c9-2bd8-45f0-a603-5861a94e73fb" />
</p>

---

## ⚙️ Tech Stack

### Programming & Tools
- **Language:** Python  
- **IDE:** Jupyter Notebook, VS Code  
- **Version Control:** Git & GitHub  

### Data & Visualization
- pandas, numpy  
- matplotlib, seaborn, plotly  

### Machine Learning
- scikit-learn  
- Linear Regression  
- Random Forest Regression  
- XGBoost (optional experimentation)

### Model Explainability
- SHAP  
- eli5  

### Deployment (Future Scope)
- Flask / Streamlit  
- AWS / Heroku / GCP  

---

## 🧪 Machine Learning Workflow

1. Data collection from multiple sources  
2. Data cleaning & merging  
3. Exploratory Data Analysis (EDA)  
4. Outlier handling using **IQR-based capping**  
5. Feature engineering (season index, water stress index)  
6. Model training & evaluation  
7. Crop-wise yield prediction  
8. Top-5 crop ranking  
9. Visualization & result analysis  

---

## 📈 Model Performance

| Model | R² Score |
|------|---------|
| Linear Regression | ~0.23 |
| Random Forest Regression | **~0.91** |

✅ **Random Forest Regression** was selected due to:
- High predictive accuracy  
- Strong generalization (no overfitting)  
- Ability to capture non-linear relationships  

---

## 🎯 Results & Insights

- Accurate prediction of crop production values  
- Intelligent ranking of crops for given conditions  
- Clear visual comparison using bar charts  
- Rainfall, NPK values, and engineered features were found to be the most influential  

---

## 📁 Project Structure

AgriYield_Predictor/

├── AgriYield_Predictor.ipynb # Complete ML pipeline

├── README.md # Project documentation

└── data/ # Raw & processed datasets

---

## 🧑‍💻 Author

**Shudhanshu Yadav**  
🎓 Data Science Student  
📧 Email: skyadav7683@gmail.com  
🏢 Infosys Springboard AI Internship Project  

---

## 📌 Future Improvements

- 🌐 Web application using Flask or Streamlit  
- ☁️ Real-time weather API integration  
- 📍 Region-specific crop recommendation  
- 🧠 Advanced explainability using SHAP dashboards  
- 📊 Production forecasting across seasons  

---

## 🖥️ Demo Preview

The following visual demonstrates the Top-5 Crop Prediction output:

<p align="center">
  <img width="985" height="615" alt="image" src="https://github.com/user-attachments/assets/cc63dc32-0422-459d-a235-f60320f850f4" />

</p>


⭐ If you found this project useful, feel free to **star the repository** and share feedback!
