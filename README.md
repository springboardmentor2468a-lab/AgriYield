# 🌾 AgroSky AI  
### AI-Powered Crop Yield Prediction & Intelligent Crop Recommendation System

AgroSky AI is an end-to-end **Machine Learning–based smart agriculture system** that uses **Regression and Classification models together** to help farmers and planners make **accurate, data-driven crop decisions**.

The system:
- 📈 Predicts **expected crop yield (kg/ha)**
- 🌱 Identifies **ONE best suitable crop**
- 🧠 Explains **why that crop is suitable**
- 📊 Visualizes predicted yield using **bar charts**
- 🚀 Is **deployment-ready** using trained ML models

---

## 🚀 Project Objective

To assist farmers, agricultural researchers, and policymakers in selecting the **most suitable crop** for given soil and climate conditions while estimating its **expected production**.

---

## 🧠 System Intelligence (Two-Stage ML Pipeline)

AgroSky AI works in **two clear stages**:

---

### 🔹 Stage 1: Crop Yield Prediction (Regression)

A **Random Forest Regression model** predicts the **expected crop yield (kg/ha)** based on soil nutrients and weather conditions.

📌 Output:
- Numerical yield value (e.g., `6844.39 kg/ha`)
- Used later for visualization and ranking

---

### 🔹 Stage 2: Best Crop Recommendation (Classification)

A **Random Forest Classification model** analyzes the same inputs and predicts **ONE most suitable crop**.

📌 Output:
- ✅ **Best crop name**
- 🧠 **Reason for suitability**
- 🎯 Based on learned soil-climate-crop relationships

👉 **Important:**  
This classification model **does NOT return multiple crops**.  
It predicts **only one optimal crop**, which is the **final recommendation**.

---

## 🌾 Input Parameters

### Soil Parameters
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Soil pH

### Climate Parameters
- Temperature (°C)
- Humidity (%)
- Rainfall (mm)

---

## ✨ Key Features

- 📈 Accurate crop yield prediction (Regression)
- 🌱 Single best crop recommendation (Classification)
- 🧠 Explanation of crop suitability
- 📊 Yield visualization using bar charts
- 🔍 Feature importance analysis
- 💾 Trained models saved as `.pkl`
- 🌐 Backend ready for web deployment
- 🎓 Internship & research ready

---

## 📊 Dataset Sources

1. **FAO Crop Production Dataset**  
   https://www.fao.org/faostat/en/#data/QCL  

2. **Crop Recommendation Dataset (Kaggle)**  
   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset  

3. **Government & Climate Data**
- https://www.data.gov.in/sector/agriculture  
- https://www.ncei.noaa.gov/

---

## 🏗️ System Architecture
<img width="725" height="495" alt="image" src="https://github.com/user-attachments/assets/992e7525-1f4b-408f-af89-6528367103b0" />


---

## ⚙️ Tech Stack

### Programming & Tools
- Python
- Jupyter Notebook
- VS Code
- Git & GitHub

### Libraries
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib

---

## 🧪 Machine Learning Workflow

1. Data collection from multiple sources  
2. Data cleaning & merging  
3. Exploratory Data Analysis (EDA)  
4. Outlier handling (IQR capping)  
5. Feature engineering  
6. Feature scaling (StandardScaler)  
7. Label encoding (classification target)  
8. Train-test split (80-20, stratified)  
9. Model training  
10. Model evaluation  
11. Model saving for deployment  

---

## 📈 Model Performance

### 🔹 Regression (Yield Prediction)

| Model | R² Score |
|------|---------|
| Linear Regression | ~0.23 |
| **Random Forest Regression** | **~0.91 ✅** |

✔ Selected due to:
- High accuracy
- Non-linear learning
- Robust performance

---

### 🔹 Classification (Best Crop Selection)

**Model:** Random Forest Classifier  

- Accuracy ≈ **99.5%**
- Precision ≈ **1.00**
- Recall ≈ **1.00**
- F1-Score ≈ **1.00**

📌 Classification predicts **only ONE best crop**, not multiple.

---

## 🔍 Feature Importance (Classification)

Most influential features identified by the model:

1. Rainfall  
2. Humidity  
3. Potassium (K)  
4. Phosphorus (P)  
5. Temperature  
6. Nitrogen (N)  
7. Soil pH  

This confirms that **climate and soil nutrients strongly influence crop suitability**.

---

## 📁 Project Structure


---

## 🧑‍💻 Author

**Shudhanshu Yadav (Sky)**  
🎓 BS Data Science & Applications – IIT Madras  
📧 Email: skyadav7683@gmail.com  
🏢 Infosys Springboard AI Internship Project  

---

## 🌟 Future Enhancements

- 🌐 Full-stack web dashboard
- ☁️ Real-time weather API integration
- 📍 Region-specific crop recommendation
- 🧠 SHAP-based explainability
- 📊 Seasonal yield forecasting

---

## ⭐ Final Note

AgroSky AI demonstrates how **Regression and Classification together** can solve real-world agricultural problems by:

- Predicting **how much crop can be produced**
- Recommending **which crop should be grown**
- Explaining **why that crop is suitable**

If you found this project useful, please ⭐ star the repository and share feedback.

