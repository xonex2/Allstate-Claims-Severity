import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from ml.data_loader import TARGET_COL


PREFERRED_NUMERIC_FEATURES = [
    "cont1",
    "cont2",
    "cont3",
    "cont4",
    "cont5",
    "cont6",
    "cont7",
    "cont8",
    "cont9",
    "cont10",
    "cont11",
    "cont12",
    "cont13",
    "cont14",
]

PREFERRED_CATEGORICAL_FEATURES = [
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
    "cat10",
]


def normalization(df):
    return _scale_features(df, MinMaxScaler())


def standard_scaling(df):
    return _scale_features(df, StandardScaler())


def one_hot_encoding(df):
    train_df, test_df, available_cols = _split_feature_data(
        df,
        _categorical_features(df),
        "No categorical feature columns are available for one-hot encoding.",
    )

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    train_encoded = encoder.fit_transform(train_df[available_cols])
    test_encoded = encoder.transform(test_df[available_cols])
    encoded_cols = encoder.get_feature_names_out(available_cols)

    return (
        pd.DataFrame(train_encoded, columns=encoded_cols, index=train_df.index),
        pd.DataFrame(test_encoded, columns=encoded_cols, index=test_df.index),
    )


def _scale_features(df, scaler):
    train_df, test_df, available_cols = _split_feature_data(
        df,
        _numeric_features(df),
        "No numeric continuous feature columns are available for scaling.",
    )
    train_scaled = scaler.fit_transform(train_df[available_cols])
    test_scaled = scaler.transform(test_df[available_cols])

    return (
        pd.DataFrame(train_scaled, columns=available_cols, index=train_df.index),
        pd.DataFrame(test_scaled, columns=available_cols, index=test_df.index),
    )


def _split_feature_data(df, feature_cols, error_message):
    available_cols = [column for column in feature_cols if column in df.columns]
    if not available_cols:
        raise ValueError(error_message)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )
    return train_df, test_df, available_cols


def _numeric_features(df):
    preferred = [column for column in PREFERRED_NUMERIC_FEATURES if column in df.columns]
    if preferred:
        return preferred
    return [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in {"id", TARGET_COL, "log_loss"}
    ][:10]


def _categorical_features(df):
    preferred = [column for column in PREFERRED_CATEGORICAL_FEATURES if column in df.columns]
    if preferred:
        return preferred
    return [
        column
        for column in df.select_dtypes(include=["object", "string"]).columns
    ][:10]
