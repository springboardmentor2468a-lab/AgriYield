# 🌱 AgriYield & AgriCrop Advisor  
## AI-Based Agricultural Prediction & Recommendation Systems

An AI-powered web-based solution consisting of **two independent Machine Learning projects** that help farmers and agricultural planners make **data-driven decisions** using soil and environmental parameters.

The system integrates **Machine Learning (Regression & Classification)** with **Flask-based Web Development** to solve real-world agricultural problems.

---

## 📌 Project Structure Overview

### 🔹 Project 1: Crop Yield Prediction (Regression)
Predicts the **expected yield** of a selected crop based on soil nutrients and climatic conditions using regression-based machine learning models.

### 🔹 Project 2: Crop Recommendation System (Classification)
Recommends the **most suitable crop to cultivate** based on soil and weather parameters using classification algorithms.

> ⚠️ Both projects are implemented **independently** using different datasets and machine learning approaches.

---

## 🎯 Objectives

- Predict crop yield using **regression-based ML models**
- Recommend the most suitable crop using **classification algorithms**
- Provide a user-friendly web interface for farmers
- Demonstrate real-world application of **AI/ML in agriculture**
- Build complete end-to-end ML systems (Data → Model → Web App)

---

## 🚀 Key Features

### ✅ Crop Yield Prediction (Regression Project)

- 🌾 Predicts crop yield based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - Rainfall
  - pH value
- 📈 Yield estimation using regression models
- 🧠 AI-driven numeric prediction output

---

### ✅ Crop Recommendation System (Classification Project)

- 🌱 Recommends the **best crop to cultivate**
- Uses the same soil and climate inputs
- 🤖 ML classification-based decision making
- ⚡ Instant crop recommendation
- 🌐 Web-based interface with **three pages**:
  - Introduction page
  - Input form
  - Result page

---

## 🛠️ Tech Stack Used

### Programming Language
- Python

### Frontend
- HTML
- CSS
- Bootstrap

### Backend
- Flask (Python Web Framework)

### Machine Learning
- pandas
- numpy
- scikit-learn
- pickle / joblib

### Tools
- VS Code
- Jupyter Notebook
- Git & GitHub

---

## 🧠 Machine Learning Workflow

### 🔹 Dataset Collection
- Crop Yield datasets (FAO / Kaggle)
- Crop Recommendation dataset

---

### 🔹 Data Preprocessing
- Handling missing values
- Feature selection
- Feature scaling using `StandardScaler`
- Label encoding (for classification)

---

### 🔹 Model Training

#### 📌 Regression Model (Project 1)
- Used for **predicting crop yield**
- Outputs **continuous numeric values**

#### 📌 Classification Model (Project 2)
- Used for **crop recommendation**
- Outputs **crop name (class label)**

Models are trained and saved using `pickle` or `joblib`.

---

### 🔹 Prediction & Deployment
- User inputs collected via web forms
- Inputs passed to trained ML models
- Predictions displayed on result web pages

---

## 📈 Output

### Crop Yield Prediction
- Displays predicted yield value for the selected crop

### Crop Recommendation
- Displays the most suitable crop for given conditions

---

## 💡 Real-World Applications

- Smart farming and precision agriculture
- AI-based agricultural decision support systems
- Crop planning and yield optimization
- Sustainable and technology-driven farming solutions

---

## 👩‍💻 Author

**Navya Sree Naidu**  
Passionate about **Web Development & Artificial Intelligence / Machine Learning**  
Focused on building **real-world, impact-driven technology solutions**

---

## 🎓 One-Line Viva Explanation

> “This project consists of two machine learning systems: a regression-based crop yield prediction model and a classification-based crop recommendation system, both deployed using Flask.”

---

## 📜 License
This project is for **academic and educational purposes**.
