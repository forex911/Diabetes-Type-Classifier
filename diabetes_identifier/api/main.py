"""FastAPI application for diabetes classification."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Diabetes Classifier API", version="1.0.0")

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "Type 1", 1: "Type 2", 2: "Gestational", 3: "Other"}

# ---------------------------------------------------------------------------
# Model loading (lazy, on first request)
# ---------------------------------------------------------------------------
_model = None
_project_root = Path(__file__).resolve().parents[2]
_models_dir = _project_root / "diabetes_identifier" / "models"


def _load_model():
    global _model
    if _model is not None:
        return _model
    pkl_path = _models_dir / "structured_classifier.pkl"
    if not pkl_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python -m diabetes_identifier.models.train",
        )
    with open(pkl_path, "rb") as fh:
        _model = pickle.load(fh)
    return _model


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    bmi: float = Field(..., ge=10, le=70, description="Body mass index")
    glucose: float = Field(..., ge=50, le=500, description="Blood glucose mg/dL")
    insulin: float = Field(..., ge=0, le=200, description="Insulin level μU/mL")
    notes: Optional[str] = Field(default="no clinical notes provided", description="Clinical notes")


class PredictResponse(BaseModel):
    label: int
    diagnosis: str
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model = _load_model()
    X = np.array([[req.age, req.bmi, req.glucose, req.insulin]])
    proba = model.predict_proba(X)[0]
    label = int(np.argmax(proba))
    return PredictResponse(
        label=label,
        diagnosis=LABEL_NAMES[label],
        probabilities={LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
