from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_predictor import CropYieldPredictor

app = FastAPI(title="AgriSense AI Backend", version="1.0")
predictor = CropYieldPredictor()

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    crop: str   # <-- using crop NAME

@app.get("/")
async def home():
    return {"message": "AgriSense AI Backend Running!", "crops": predictor.crops}

@app.post("/predict_yield")
async def predict_yield(req: CropInput):
    if req.crop not in predictor.crops:
        raise HTTPException(status_code=400, detail=f"Invalid Crop: {req.crop}")

    user_yield = predictor.predict_yield(
        req.crop, req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall
    )

    top5 = predictor.predict_all_crops(
        req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall
    )

    return {
        "user_crop_yield": user_yield,
        "top_recommendations": top5
    }
