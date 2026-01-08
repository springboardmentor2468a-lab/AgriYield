# 🌱 AgroSky AI - Run Instructions

## 📋 Prerequisites

Make sure you have installed:
- Python 3.7 or higher
- All required packages (see requirements.txt)

## 🚀 How to Run the Project

### Step 1: Install Dependencies

Open terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

Required packages:
- flask
- flask-cors
- numpy
- pandas
- scikit-learn
- joblib

### Step 2: Check Model Files

Make sure these files are in your project folder:
- ✅ `crop_classifier.pkl`
- ✅ `classifier_scaler.pkl`
- ✅ `final_random_forest_model.pkl`
- ✅ `scaler.pkl`
- ✅ `label_encoder.pkl`

### Step 3: Start the Backend Server

Run this command in terminal:

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**⚠️ IMPORTANT:** Keep this terminal window open! Don't close it.

### Step 4: Open the Website

1. Open your web browser (Chrome, Firefox, Edge, etc.)
2. Go to: `http://127.0.0.1:5000`
3. You should see the AgroSky AI home page

### Step 5: Use the Application

#### For Yield Prediction:
1. Click on **"Yield Prediction"** card on home page
2. Fill in all 7 input fields:
   - Nitrogen (N) - mg/kg
   - Phosphorus (P) - mg/kg
   - Potassium (K) - mg/kg
   - Temperature - °C
   - Humidity - %
   - Rainfall - mm
   - Soil pH - (0-14)
3. Click **"Generate Prediction"** button
4. See results:
   - Top 5 crop recommendations
   - Bar chart showing yields in kg/ha
   - AI insights and tips

#### For Crop Recommendation:
1. Click on **"Crop Recommendation"** card on home page
2. Fill in all 7 input fields (same as above)
3. Click **"Generate Prediction"** button
4. See the recommended crop with suitability score

## 🔧 Troubleshooting

### Problem: "Error making prediction" or "Backend server is not running"

**Solution:**
1. Make sure you ran `python app.py` in terminal
2. Check if you see "Running on http://127.0.0.1:5000" message
3. Don't close the terminal window
4. Try refreshing the browser page

### Problem: "ModuleNotFoundError" or "No module named 'flask'"

**Solution:**
```bash
pip install flask flask-cors numpy pandas scikit-learn joblib
```

### Problem: "FileNotFoundError" for .pkl files

**Solution:**
- Make sure all .pkl model files are in the same folder as `app.py`
- Check file names match exactly (case-sensitive)

### Problem: Browser shows "This site can't be reached"

**Solution:**
1. Make sure backend is running (check terminal)
2. Use exact URL: `http://127.0.0.1:5000` (not `localhost`)
3. Check if port 5000 is already in use

### Problem: Predictions not showing

**Solution:**
1. Check browser console (F12 → Console tab) for errors
2. Make sure all input fields are filled with valid numbers
3. Check backend terminal for error messages
4. Verify model files are loaded correctly

## 📝 Example Input Values

Try these sample values to test:

```
Nitrogen: 90
Phosphorus: 42
Potassium: 43
Temperature: 20.9
Humidity: 82.0
Rainfall: 202.9
Soil pH: 6.5
```

## 🎯 Project Structure

```
AgroSky AI/
├── app.py                    # Flask backend server
├── templates/
│   ├── index.html           # Home page
│   ├── dashboard.html       # Prediction dashboard
│   ├── services.html        # Services page
│   ├── about.html           # About page
│   └── contact.html         # Contact page
├── static/
│   ├── style.css            # All CSS styles
│   └── dashboard.js         # JavaScript for predictions
├── *.pkl                    # Trained ML models
└── requirements.txt         # Python dependencies
```

## ✅ Success Checklist

Before running, make sure:
- [ ] Python is installed
- [ ] All packages are installed (`pip install -r requirements.txt`)
- [ ] All .pkl model files are present
- [ ] Backend server is running (`python app.py`)
- [ ] Browser is open at `http://127.0.0.1:5000`
- [ ] No errors in terminal
- [ ] No errors in browser console (F12)

## 🆘 Still Having Issues?

1. Check terminal for error messages
2. Open browser console (F12) and check for JavaScript errors
3. Verify all files are in correct locations
4. Make sure port 5000 is not blocked by firewall

---

**Built by Team Skype ❤️**  
Team Leader: Shudhanshu Yadav  
IIT Madras BS Data Science and Applications

