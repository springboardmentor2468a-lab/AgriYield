# 🚀 AgroSky AI - Quick Start Guide

## ⚡ Fast Setup (3 Steps)

### Step 1: Install Packages
```bash
pip install flask flask-cors numpy pandas scikit-learn joblib
```

### Step 2: Start Server
```bash
python app.py
```

**Wait for this message:**
```
 * Running on http://127.0.0.1:5000
```

### Step 3: Open Browser
Go to: **http://127.0.0.1:5000**

---

## ✅ Verify It's Working

### Option 1: Check Health (Browser)
Open: **http://127.0.0.1:5000/health**

Should show:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "message": "AgroSky AI Backend is running"
}
```

### Option 2: Test Script
In a NEW terminal window:
```bash
pip install requests
python test_backend.py
```

---

## 🎯 How to Use

### Yield Prediction:
1. Home page → Click **"Yield Prediction"**
2. Fill all 7 fields
3. Click **"Generate Prediction"**
4. See Top 5 crops + Bar chart (kg/ha) + Tips

### Crop Recommendation:
1. Home page → Click **"Crop Recommendation"**
2. Fill all 7 fields
3. Click **"Generate Prediction"**
4. See recommended crop + Growing tips

---

## 🔧 Common Issues

### ❌ "Backend server is not running"
**Fix:** Run `python app.py` in terminal first!

### ❌ "ModuleNotFoundError"
**Fix:** Run `pip install -r requirements.txt`

### ❌ "FileNotFoundError" for .pkl files
**Fix:** Make sure all .pkl files are in same folder as app.py

### ❌ Browser can't connect
**Fix:** 
- Check terminal shows "Running on http://127.0.0.1:5000"
- Use exact URL: `http://127.0.0.1:5000`
- Don't close the terminal!

---

## 📝 Sample Input Values

Try these to test:
```
Nitrogen: 90
Phosphorus: 42
Potassium: 43
Temperature: 20.9
Humidity: 82.0
Rainfall: 202.9
Soil pH: 6.5
```

---

## 📞 Need Help?

1. Check terminal for error messages
2. Open browser console (F12) for JavaScript errors
3. Verify all files are in correct location
4. Make sure port 5000 is not blocked

---

**Built by Team Skype ❤️**

