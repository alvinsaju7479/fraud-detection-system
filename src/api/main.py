import json
from typing import Dict, List, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import MODELS_DIR, PROCESSED_DIR, TARGET_COL
from src.api.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Fraud Detection API", version="1.1")

MODEL = None
THRESHOLD: float = 0.5
REQUIRED_FEATURES: List[str] = []


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running. Go to /docs"}


@app.on_event("startup")
def load_artifacts():
    """
    Load model + threshold + expected feature schema once at startup.
    """
    global MODEL, THRESHOLD, REQUIRED_FEATURES

    model_path = MODELS_DIR / "xgb_calibrated.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found at: {model_path}")

    MODEL = joblib.load(model_path)

    # Load threshold if present
    tpath = MODELS_DIR / "thresholds.json"
    if tpath.exists():
        with open(tpath, "r", encoding="utf-8") as f:
            THRESHOLD = float(json.load(f).get("threshold_cost_based", 0.5))

    # Load required features from processed train file (best source of truth)
    train_path = PROCESSED_DIR / "train.parquet"
    if train_path.exists():
        df = pd.read_parquet(train_path)
        REQUIRED_FEATURES = [c for c in df.columns if c != TARGET_COL]
    else:
        # fallback: if train parquet missing, infer from model pipeline if possible
        REQUIRED_FEATURES = []

    if not REQUIRED_FEATURES:
        # For creditcard dataset, fallback schema if needed
        REQUIRED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@app.get("/sample")
def sample():
    """
    Returns a valid request body pulled from test.parquet so you can call /predict easily.
    """
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    cols = [c for c in df.columns if c != TARGET_COL]

    row = df[cols].iloc[0].to_dict()  # first row
    return {"features": row}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "threshold": THRESHOLD
    }


@app.get("/schema")
def schema():
    """
    Returns required feature columns (for client integration).
    """
    return {"required_features": REQUIRED_FEATURES, "count": len(REQUIRED_FEATURES)}


@app.get("/metadata")
def metadata():
    """
    Extra info to help in README / debugging.
    """
    return {
        "model_file": "models/xgb_calibrated.joblib",
        "threshold_file": "models/thresholds.json",
        "threshold_strategy": "cost_based",
        "threshold": THRESHOLD
    }


def _validate_and_build_df(features: Dict[str, Any]) -> pd.DataFrame:
    """
    Ensures:
      - all required features present
      - no extra unknown features (optional: allow extras)
      - values are numeric
      - dataframe columns are ordered exactly as REQUIRED_FEATURES
    """
    if not isinstance(features, dict):
        raise HTTPException(status_code=422, detail="features must be a dictionary")

    missing = [c for c in REQUIRED_FEATURES if c not in features]
    extra = [c for c in features.keys() if c not in REQUIRED_FEATURES]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_features",
                "missing": missing[:20],
                "missing_count": len(missing)
            }
        )

    if extra:
        # strict mode: error on extra keys (keeps API clean)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unexpected_features",
                "extra": extra[:20],
                "extra_count": len(extra)
            }
        )

    # coerce to numeric and fail if non-numeric
    row = {}
    non_numeric = []
    for k in REQUIRED_FEATURES:
        v = features.get(k)
        try:
            row[k] = float(v)
        except Exception:
            non_numeric.append(k)

    if non_numeric:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "non_numeric_values",
                "fields": non_numeric
            }
        )

    df = pd.DataFrame([row], columns=REQUIRED_FEATURES)
    return df


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict for a single transaction.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = _validate_and_build_df(req.features)

    proba = float(MODEL.predict_proba(df)[:, 1][0])
    is_fraud = bool(proba >= THRESHOLD)

    return PredictResponse(
        fraud_probability=proba,
        threshold=THRESHOLD,
        is_fraud=is_fraud,
        reasons=None
    )


@app.post("/batch_predict")
def batch_predict(payload: Dict[str, Any]):
    """
    Batch input format:
    {
      "rows": [
        {"Time":..., "V1":..., ..., "Amount":...},
        {...}
      ]
    }
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        raise HTTPException(status_code=422, detail="payload must contain a non-empty 'rows' list")

    # Validate each row
    data = []
    for i, r in enumerate(rows):
        try:
            df_one = _validate_and_build_df(r)
            data.append(df_one.iloc[0].to_dict())
        except HTTPException as e:
            raise HTTPException(status_code=422, detail={"row_index": i, "detail": e.detail})

    df = pd.DataFrame(data, columns=REQUIRED_FEATURES)

    proba = MODEL.predict_proba(df)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    return {
        "threshold": THRESHOLD,
        "count": len(rows),
        "results": [
            {"fraud_probability": float(p), "is_fraud": bool(y)}
            for p, y in zip(proba, preds)
        ]
    }
