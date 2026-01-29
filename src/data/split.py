import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DIR, TARGET_COL, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger("split")

def main(test_size: float = 0.2):
    full_path = PROCESSED_DIR / "full.parquet"
    if not full_path.exists():
        raise FileNotFoundError("Run make_dataset first to create full.parquet")

    df = pd.read_parquet(full_path)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL]
    )

    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Train saved -> {train_path} | shape={train_df.shape} | fraud={train_df[TARGET_COL].mean():.6f}")
    logger.info(f"Test  saved -> {test_path} | shape={test_df.shape} | fraud={test_df[TARGET_COL].mean():.6f}")

if __name__ == "__main__":
    main()
