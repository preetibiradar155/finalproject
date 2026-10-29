import os
import json
from typing import Optional, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')
ART_DIR = os.path.join(ROOT, 'artifacts')

app = FastAPI(title="Student Marks Prediction API", version="1.0.0")

_model = None


class StudentInput(BaseModel):
    school: str = Field(..., example="GP")
    sex: str = Field(..., example="F")
    age: int = Field(..., ge=15, le=22, example=18)
    address: str = Field(..., example="U")
    famsize: str = Field(..., example="GT3")
    Pstatus: str = Field(..., example="T")
    Medu: int = Field(..., ge=0, le=4)
    Fedu: int = Field(..., ge=0, le=4)
    Mjob: str = Field(..., example="at_home")
    Fjob: str = Field(..., example="teacher")
    reason: str = Field(..., example="course")
    guardian: str = Field(..., example="mother")
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=4)
    schoolsup: str = Field(..., example="no")
    famsup: str = Field(..., example="yes")
    paid: str = Field(..., example="no")
    activities: str = Field(..., example="no")
    nursery: str = Field(..., example="yes")
    higher: str = Field(..., example="yes")
    internet: str = Field(..., example="yes")
    romantic: str = Field(..., example="no")
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    Dalc: int = Field(..., ge=1, le=5)
    Walc: int = Field(..., ge=1, le=5)
    health: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)


class PredictionResponse(BaseModel):
    prediction: float
    model_path: str


def load_model():
    global _model
    if _model is not None:
        return _model
    cand1 = os.path.join(MODELS_DIR, 'best_model.joblib')
    cand2 = os.path.join(ART_DIR, 'best_model.joblib')
    path = cand1 if os.path.exists(cand1) else cand2
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model not found at {path}')
    _model = joblib.load(path)
    return _model


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"Warning: {e}")


@app.get("/")
def root():
    return {"app": "student-marks-prediction", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    try:
        m = load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/model-info")
def model_info():
    try:
        m = load_model()
        path = getattr(m, '__module__', 'unknown')
        cand1 = os.path.join(MODELS_DIR, 'best_model.joblib')
        cand2 = os.path.join(ART_DIR, 'best_model.joblib')
        mp = cand1 if os.path.exists(cand1) else cand2
        info = {
            "model_type": str(type(m)),
            "model_path": mp,
            "steps": [s[0] for s in getattr(m, 'steps', [])] if hasattr(m, 'steps') else None
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: StudentInput):
    try:
        model = load_model()
        df = pd.DataFrame([input_data.model_dump()])
        pred = model.predict(df)
        # return float
        val = float(pred[0]) if hasattr(pred, '__iter__') else float(pred)
        cand1 = os.path.join(MODELS_DIR, 'best_model.joblib')
        cand2 = os.path.join(ART_DIR, 'best_model.joblib')
        mp = cand1 if os.path.exists(cand1) else cand2
        return PredictionResponse(prediction=val, model_path=mp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
