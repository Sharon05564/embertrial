"""
FastAPI backend serving the Model Dashboard and Live Classifier Demo
frontend, plus the real (non-simulated) prediction API.

Run with:
    uvicorn backend.main:app --reload
"""

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.feature_extraction import extract_features, features_to_dataframe, is_pe_file

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "detection_model.pkl"
RESULTS_PATH = ROOT / "model_results.json"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

app = FastAPI(title="EMBER2024 Malware Classification API")

_model = None


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/api/status")
def status():
    return {
        "model_loaded": get_model() is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "results_available": RESULTS_PATH.exists(),
    }


@app.get("/api/dashboard")
def dashboard():
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="model_results.json not found. Run backend/train_and_save_model.py first.",
        )
    return json.loads(RESULTS_PATH.read_text())


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model loaded. Run backend/train_and_save_model.py first.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes). Max is {MAX_UPLOAD_BYTES} bytes.",
        )

    features = extract_features(data)
    X = features_to_dataframe(features)

    malware_prob = float(model.predict_proba(X)[0, 1])
    prediction = "Malware" if malware_prob >= 0.5 else "Benign"

    return {
        "filename": file.filename,
        "file_size": len(data),
        "is_pe": bool(is_pe_file(data)),
        "prediction": prediction,
        "malware_probability": malware_prob,
        "benign_probability": 1.0 - malware_prob,
        "features": features,
    }


# Static frontend
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
