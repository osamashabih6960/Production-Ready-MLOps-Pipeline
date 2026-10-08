from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path("models/best_model.pkl")


# --------------------------------------------------
# Load Model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="AI4I Predictive Maintenance API",
    description="Machine failure prediction API",
    version="1.0.0"
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    Type: int
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AI4I Predictive Maintenance API is running"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = pd.DataFrame([
        {
            "Type": request.Type,
            "Air temperature [K]": request.air_temperature,
            "Process temperature [K]": request.process_temperature,
            "Rotational speed [rpm]": request.rotational_speed,
            "Torque [Nm]": request.torque,
            "Tool wear [min]": request.tool_wear
        }
    ])

    prediction = model.predict(input_data)[0]

    return {
        "machine_failure": int(prediction),
        "prediction": (
            "Machine Failure"
            if prediction == 1
            else "No Machine Failure"
        )
    }