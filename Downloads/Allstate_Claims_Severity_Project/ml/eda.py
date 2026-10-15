import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data_loader import TARGET_COL


sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")
NUMERIC_CHART_LIMIT = 10


def get_summary_stats(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    describe_df = df[numeric_cols].describe().T.round(2) if numeric_cols else pd.DataFrame()

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_summary = {
        column: df[column].value_counts().head(10).to_dict()
        for column in categorical_cols[:10]
    }

    return {
        "numeric_describe": describe_df.to_dict(orient="index"),
        "categorical_summary": cat_summary,
        "average_loss": round(float(df[TARGET_COL].mean()), 2) if TARGET_COL in df.columns else None,
        "max_loss": round(float(df[TARGET_COL].max()), 2) if TARGET_COL in df.columns else None,
        "correlation_with_target": _correlation_with_target(df, numeric_cols),
    }


def generate_charts(df: pd.DataFrame) -> list:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    filenames = []

    if TARGET_COL in df.columns:
        fname = "loss_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x=TARGET_COL, kde=True, bins=30, ax=ax, color="#176b87")
        ax.set_title("Distribution of Insurance Claim Loss")
        ax.set_xlabel("Loss")
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    if "log_loss" in df.columns:
        fname = "log_loss_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x="log_loss", kde=True, bins=30, ax=ax, color="#d9822b")
        ax.set_title("Distribution of Log Loss")
        ax.set_xlabel("log(1 + loss)")
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    cont_cols = [column for column in df.columns if column.startswith("cont")]
    if cont_cols and TARGET_COL in df.columns:
        fname = "continuous_correlation_heatmap.png"
        fig, ax = plt.subplots(figsize=(9, 7))
        corr_cols = cont_cols[:NUMERIC_CHART_LIMIT] + [TARGET_COL]
        sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, ax=ax, annot=False)
        ax.set_title("Continuous Feature Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    cat_cols = [column for column in df.columns if column.startswith("cat")]
    if cat_cols:
        fname = "categorical_cardinality.png"
        cardinality = df[cat_cols].nunique().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=cardinality.values, y=cardinality.index, ax=ax, color="#176b87")
        ax.set_title("Categorical Feature Cardinality")
        ax.set_xlabel("Number of Unique Levels")
        fig.tight_layout()
        fig.savefig(os.path.join(CHARTS_DIR, fname), dpi=110)
        plt.close(fig)
        filenames.append(fname)

    return filenames


def _correlation_with_target(df: pd.DataFrame, numeric_cols):
    if TARGET_COL not in df.columns:
        return {}

    columns = [column for column in numeric_cols if column not in {TARGET_COL, "id"}]
    if not columns:
        return {}

    corr = df[columns + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    return corr.abs().sort_values(ascending=False).head(10).round(3).to_dict()
