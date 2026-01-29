import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger("make_dataset")

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / "creditcard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    # Basic checks
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found.")
    if df.isnull().any().any():
        # This dataset usually has no nulls, but handle anyway
        df = df.dropna()

    # Save as parquet for faster reads later
    out_path = PROCESSED_DIR / "full.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"Saved processed dataset -> {out_path}")
    logger.info(f"Shape: {df.shape}, Fraud rate: {df[TARGET_COL].mean():.6f}")

if __name__ == "__main__":
    main()
