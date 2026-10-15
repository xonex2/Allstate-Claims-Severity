import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

TARGET_COL = "loss"
ID_COLS = ["id"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place train.csv in the data folder."
        )

    return pd.read_csv(path)


def get_data_summary(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_counts": missing.to_dict(),
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_summary": df[TARGET_COL].describe().round(2).to_dict()
        if TARGET_COL in df.columns
        else {},
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.drop_duplicates(inplace=True)

    obj_cols = clean.select_dtypes(include="object").columns
    for column in obj_cols:
        clean[column] = clean[column].astype("string").str.strip()

    numeric_cols = clean.select_dtypes(include="number").columns
    for column in numeric_cols:
        if clean[column].isnull().any():
            clean[column] = clean[column].fillna(clean[column].median())

    categorical_cols = clean.select_dtypes(include=["object", "string"]).columns
    for column in categorical_cols:
        if clean[column].isnull().any():
            mode = clean[column].mode(dropna=True)
            clean[column] = clean[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    if TARGET_COL in clean.columns:
        clean["log_loss"] = np.log1p(clean[TARGET_COL].clip(lower=0))

    return clean


def get_feature_target_split(df: pd.DataFrame):
    drop_cols = [column for column in ID_COLS + [TARGET_COL, "log_loss"] if column in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y
