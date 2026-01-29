# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TARGET_COL = "Class"

# For cost-based threshold tuning (edit as you want)
COST_FALSE_NEGATIVE = 500  # missed fraud
COST_FALSE_POSITIVE = 5    # manual review cost
