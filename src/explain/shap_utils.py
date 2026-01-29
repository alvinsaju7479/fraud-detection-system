import json
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COL
from src.data.features import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger("shap")

def main():
    shap_dir = REPORTS_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    feature_cols = get_feature_columns(test_df)

    model = joblib.load(MODELS_DIR / "xgb.joblib")  # use raw xgb pipeline for SHAP
    # Pipeline -> need transformed features
    X = test_df[feature_cols].sample(2000, random_state=42)

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    X_trans = prep.transform(X)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)

    # Global summary
    plt.figure()
    shap.summary_plot(shap_values, X_trans, show=False)
    out1 = shap_dir / "shap_summary.png"
    plt.savefig(out1, bbox_inches="tight", dpi=200)
    plt.close()
    logger.info(f"Saved {out1}")

    # Local example: pick the most risky row
    proba = model.predict_proba(X)[:, 1]
    idx = proba.argmax()
    x_one = X_trans[idx:idx+1]

    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[idx], x_one, matplotlib=True, show=False)
    out2 = shap_dir / "shap_force.png"
    plt.savefig(out2, bbox_inches="tight", dpi=200)
    plt.close()
    logger.info(f"Saved {out2}")

if __name__ == "__main__":
    main()
