"""Wine Cultivar Classifier """

import json
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import CLASS_NAMES, FEATURE_NAMES
from src.predict import predict_data
from src.train import METRICS_PATH

app = FastAPI(
    title="Wine Cultivar Classifier",
    description="Predicts the cultivar of a wine from 13 chemical measurements using a Random Forest model.",
    version="1.0.0",
)

EXAMPLE = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315": 3.92,
    "proline": 1065.0,
}


class WineData(BaseModel):
    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"examples": [EXAMPLE]}}

    def to_features(self):
        return [getattr(self, name) for name in FEATURE_NAMES]


class WineResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


def build_response(pred, probs):
    return WineResponse(
        class_id=int(pred),
        class_name=CLASS_NAMES[int(pred)],
        confidence=round(float(max(probs)), 4),
        probabilities={CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
    )


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Wine Cultivar Classifier"}


@app.get("/model-info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metrics not found. Run: python -m src.train")
    metrics = json.loads(METRICS_PATH.read_text())
    return {"metrics": metrics, "features": FEATURE_NAMES, "classes": CLASS_NAMES}


@app.post("/predict", response_model=WineResponse)
def predict(data: WineData):
    try:
        preds, probs = predict_data([data.to_features()])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return build_response(preds[0], probs[0])


@app.post("/predict-batch", response_model=List[WineResponse])
def predict_batch(samples: List[WineData]):
    if not samples:
        raise HTTPException(status_code=400, detail="Provide at least one sample")
    if len(samples) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 samples per request")
    try:
        preds, probs = predict_data([s.to_features() for s in samples])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return [build_response(p, pr) for p, pr in zip(preds, probs)]
