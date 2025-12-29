from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import uvicorn
from model_predictor import CropYieldPredictor

app = FastAPI(title="Crop Yield API", version="1.0.0")
predictor = CropYieldPredictor()

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    crop: str

@app.get("/")
async def root():
    return {"message": "API running"}

@app.post("/predict")
async def predictyield(req: CropInput) -> Dict:
    if req.crop not in predictor.crops:
        raise HTTPException(status_code=400, detail=f"Unknown crop {req.crop}")
    userpred = predictor.predict_single(
        req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall, req.crop
    )
    top5 = predictor.predict_all_crops(
        req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall
    )
    return {
        "usercrop": userpred,
        "toprecommendations": top5,
        "totalcrops": len(predictor.crops),
    }
