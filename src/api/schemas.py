from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictRequest(BaseModel):
    features: Dict[str, float]  # {"Time":..., "V1":..., "Amount":...}

class Reason(BaseModel):
    feature: str
    impact: float

class PredictResponse(BaseModel):
    fraud_probability: float
    threshold: float
    is_fraud: bool
    reasons: Optional[List[Reason]] = None
