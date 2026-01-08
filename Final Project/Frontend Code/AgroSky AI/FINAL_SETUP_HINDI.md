# 🌱 AgroSky AI - Final Setup Guide (हिंदी में)

## ✅ Prediction Fix Ho Gaya Hai!

Main ne sab kuch fix kar diya hai. Ab aap easily run kar sakte ho.

---

## 🚀 Kaise Run Kare (Step by Step)

### Step 1: Terminal/Command Prompt Kholo
- Windows: `Win + R` → type `cmd` → Enter
- Ya PowerShell kholo

### Step 2: Project Folder Mein Jao
```bash
cd "C:\Users\HP\Downloads\AgroSky AI"
```

### Step 3: Packages Install Karo
```bash
pip install flask flask-cors numpy pandas scikit-learn joblib
```

Agar already installed hai, to skip karo.

### Step 4: Backend Server Start Karo
```bash
python app.py
```

**Important:** Terminal mein ye message dikhna chahiye:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**⚠️ WARNING:** Terminal window ko **MAT BAND KARO**! Server chal raha hai.

### Step 5: Browser Kholo
1. Chrome/Firefox/Edge kholo
2. Address bar mein type karo: `http://127.0.0.1:5000`
3. Enter press karo

### Step 6: Website Use Karo

#### Yield Prediction Ke Liye:
1. Home page par **"Yield Prediction"** card par click karo
2. Saare 7 fields fill karo:
   - Nitrogen (N)
   - Phosphorus (P)
   - Potassium (K)
   - Temperature (°C)
   - Humidity (%)
   - Rainfall (mm)
   - Soil pH (0-14)
3. **"Generate Prediction"** button click karo
4. Results dekho:
   - Top 5 crops
   - Bar graph (kg/ha mein)
   - Tips aur insights

#### Crop Recommendation Ke Liye:
1. Home page par **"Crop Recommendation"** card par click karo
2. Saare 7 fields fill karo
3. **"Generate Prediction"** button click karo
4. Recommended crop dekho

---

## 🔧 Agar Problem Aaye To

### Problem 1: "Backend server is not running"
**Solution:**
- Terminal check karo - `python app.py` chala hai ya nahi?
- Terminal window band to nahi ho gaya?
- Terminal mein "Running on http://127.0.0.1:5000" dikh raha hai?

### Problem 2: "ModuleNotFoundError"
**Solution:**
```bash
pip install flask flask-cors numpy pandas scikit-learn joblib
```

### Problem 3: Browser connect nahi ho raha
**Solution:**
- Exact URL use karo: `http://127.0.0.1:5000` (localhost mat use karo)
- Terminal check karo - server running hai?
- Firewall check karo - port 5000 block to nahi?

### Problem 4: Prediction nahi aa raha
**Solution:**
1. Browser console kholo: `F12` → Console tab
2. Errors dekho
3. Terminal mein bhi errors check karo
4. Saare input fields fill kiye hain?

---

## ✅ Test Karne Ke Liye

### Health Check:
Browser mein jao: `http://127.0.0.1:5000/health`

Should show:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Sample Values Try Karo:
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

## 📋 Checklist (Run Karne Se Pehle)

- [ ] Python installed hai (3.7+)
- [ ] Saare packages install ho gaye
- [ ] Saare .pkl files project folder mein hain
- [ ] `python app.py` run kiya
- [ ] Terminal mein "Running on..." message dikh raha hai
- [ ] Browser mein `http://127.0.0.1:5000` open kiya
- [ ] Home page dikh raha hai

---

## 🎯 Kya Fix Kiya Gaya Hai

1. ✅ **Error Handling:** Ab better error messages dikhenge
2. ✅ **Mode Detection:** Yield aur Crop recommendation properly detect hoga
3. ✅ **API Validation:** Input validation improve ki
4. ✅ **Health Check:** `/health` endpoint add kiya
5. ✅ **Background Images:** Saare pages par agricultural theme
6. ✅ **Chart Labels:** Bar graph mein kg/ha clearly dikh raha hai
7. ✅ **Tips Section:** Yield prediction aur crop recommendation dono ke liye tips

---

## 📁 Important Files

```
AgroSky AI/
├── app.py              ← Backend server (ye run karna hai)
├── templates/          ← HTML pages
├── static/             ← CSS aur JavaScript
├── *.pkl              ← ML models (must be present)
├── test_backend.py    ← Testing script
└── QUICK_START.md     ← Quick guide
```

---

## 🆘 Still Issues?

1. Terminal check karo - koi error hai?
2. Browser console (F12) check karo
3. Saare files same folder mein hain?
4. Port 5000 free hai?

---

**Built by Team Skype ❤️**  
Team Leader: Shudhanshu Yadav  
IIT Madras BS Data Science and Applications

**Ab sab ready hai! Bas `python app.py` run karo aur browser kholo! 🚀**

