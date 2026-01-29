import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from src.config import TARGET_COL

def get_feature_columns(df: pd.DataFrame):
    # For creditcard.csv: Time, V1..V28, Amount, Class
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return feature_cols

def build_preprocessor(feature_cols):
    # All features are numeric here. We'll robust-scale everything.
    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler(with_centering=True, with_scaling=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols)
        ],
        remainder="drop"
    )
    return preprocessor
