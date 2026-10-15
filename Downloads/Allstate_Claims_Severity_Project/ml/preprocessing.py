import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data_loader import ID_COLS, TARGET_COL


def fill_null_values(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()

    for column in filled.select_dtypes(include="number").columns:
        if filled[column].isnull().any():
            filled[column] = filled[column].fillna(filled[column].median())

    for column in filled.select_dtypes(include=["object", "string"]).columns:
        if filled[column].isnull().any():
            mode = filled[column].mode(dropna=True)
            filled[column] = filled[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return filled


def remove_null_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()


def build_preprocessing_report(df: pd.DataFrame) -> dict:
    filled_df = fill_null_values(df)
    removed_df = remove_null_values(df)
    regression = prepare_linear_regression_data(filled_df)

    return {
        "raw": _summary(df),
        "filled": _summary(filled_df),
        "removed": _summary(removed_df),
        "regression": regression,
        "null_columns": _null_columns(df),
    }


def prepare_linear_regression_data(df: pd.DataFrame) -> dict:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Regression target column {TARGET_COL} was not found.")

    model_df = df.copy()
    model_df["log_loss"] = np.log1p(model_df[TARGET_COL])

    numeric_cols = model_df.select_dtypes(include="number").columns.tolist()
    drop_cols = ID_COLS + [TARGET_COL, "log_loss"]
    feature_cols = [column for column in numeric_cols if column not in drop_cols]

    if not feature_cols:
        raise ValueError("No numeric feature columns are available for Linear Regression.")

    X = model_df[feature_cols]
    y = model_df["log_loss"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return {
        "target": "log_loss",
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "sample_rows": X_train.head(10).round(3).to_dict(orient="records"),
        "sample_columns": feature_cols,
    }


def _summary(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _null_columns(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    return missing[missing > 0].to_dict()
