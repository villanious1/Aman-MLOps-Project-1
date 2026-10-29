"""FastAPI interface for the trained house-price model."""
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.house_price.data import FEATURES
from src.house_price.modeling import MODEL_PATH, train_and_evaluate

app = FastAPI(
    title="Aman House Price Estimator",
    description="Educational demo API using a model trained on synthetic housing data.",
    version="1.0.0",
)


class HouseFeatures(BaseModel):
    median_income: float = Field(gt=0, le=20, description="Synthetic income index")
    house_age: float = Field(ge=0, le=100)
    rooms: float = Field(gt=0, le=20000)
    bedrooms: float = Field(gt=0, le=5000)
    population: float = Field(gt=0, le=100000)
    households: float = Field(gt=0, le=30000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@app.get("/")
def root():
    return {"project": "Aman MLOps Project 1", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(features: HouseFeatures):
    if not MODEL_PATH.exists():
        try:
            train_and_evaluate()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model is not available; run training first.") from exc
    try:
        bundle = joblib.load(MODEL_PATH)
        row = pd.DataFrame([features.model_dump()], columns=FEATURES)
        prediction = float(bundle["model"].predict(row)[0])
        return {
            "predicted_price_100k_usd": round(max(0.0, prediction), 3),
            "units": "hundreds of thousands of USD",
            "disclaimer": "Synthetic-data demo only; not a real property valuation.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
