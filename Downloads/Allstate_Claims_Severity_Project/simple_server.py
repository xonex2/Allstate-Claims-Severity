import html
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    clean = df.drop_duplicates().copy()
    for column in clean.select_dtypes(include="object").columns:
        clean[column] = clean[column].astype("string").str.strip()
        if clean[column].isnull().any():
            mode = clean[column].mode(dropna=True)
            clean[column] = clean[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    for column in clean.select_dtypes(include="number").columns:
        if clean[column].isnull().any():
            clean[column] = clean[column].fillna(clean[column].median())
    if "loss" in clean.columns:
        clean["log_loss"] = clean["loss"].clip(lower=0).map(lambda value: math.log1p(value))
    return clean


def page(title, body):
    nav = """
    <aside class="sidebar">
        <h2>Menu</h2>
        <a href="/" class="menu-btn">Home</a>
        <a href="/load" class="menu-btn">Load Data</a>
        <a href="/eda" class="menu-btn">EDA</a>
        <a href="/preprocessing" class="menu-btn">Preprocessing</a>
        <a href="/feature_engg" class="menu-btn">Feature Engg</a>
    </aside>
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header class="header">
    <h1>Allstate Claims Severity Dashboard</h1>
    <p>Dataset: <strong>Allstate Claims Severity Prediction</strong></p>
</header>
<div class="layout">
    {nav}
    <main class="content">{body}</main>
</div>
<footer class="footer"><p>Machine Learning Dashboard | First Review Web Output</p></footer>
</body>
</html>"""


def table_from_records(records, columns):
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    rows = []
    for record in records:
        cells = "".join(f"<td>{html.escape(str(record.get(column, '')))}</td>" for column in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='table-scroll'><table class='data-table'><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def mini_table(mapping):
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in mapping.items()
    )
    return f"<table class='mini-table'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def svg_histogram(values, title, color):
    values = [float(value) for value in values if pd.notna(value)]
    if not values:
        return ""
    bins = 12
    min_value, max_value = min(values), max(values)
    width = max((max_value - min_value) / bins, 1)
    counts = [0] * bins
    for value in values:
        idx = min(int((value - min_value) / width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    bars = []
    for idx, count in enumerate(counts):
        x = 45 + idx * 42
        height = 220 * count / max_count
        y = 270 - height
        bars.append(f"<rect x='{x}' y='{y:.1f}' width='28' height='{height:.1f}' fill='{color}'></rect>")
    return f"""
    <div class="chart-box">
        <svg viewBox="0 0 580 330" role="img" aria-label="{html.escape(title)}">
            <text x="24" y="28" font-size="20" font-weight="700" fill="#0f4056">{html.escape(title)}</text>
            <line x1="40" y1="270" x2="555" y2="270" stroke="#64707d"></line>
            <line x1="40" y1="55" x2="40" y2="270" stroke="#64707d"></line>
            {''.join(bars)}
            <text x="40" y="302" font-size="12" fill="#64707d">Min {min_value:.2f}</text>
            <text x="440" y="302" font-size="12" fill="#64707d">Max {max_value:.2f}</text>
        </svg>
    </div>
    """


def home():
    body = """
    <section class="hero">
        <h1>Claims Severity Analysis</h1>
        <p>A first-review dashboard for loading the Allstate claims dataset, cleaning it, exploring claim loss patterns, and preparing ML features.</p>
    </section>
    <section class="card-grid">
        <a class="card" href="/load"><h2>Data Loading</h2><p>Load the raw CSV, clean missing values, and preview rows and schema.</p></a>
        <a class="card" href="/eda"><h2>Exploratory Data Analysis</h2><p>View summary statistics, loss distribution, correlations, and charts.</p></a>
        <a class="card" href="/preprocessing"><h2>Preprocessing</h2><p>Fill null values, remove null rows, and prepare Linear Regression data.</p></a>
        <a class="card" href="/feature_engg"><h2>Feature Engineering</h2><p>Normalize continuous columns and encode categorical claim features.</p></a>
        <div class="card card-disabled"><h2>Predict</h2><p>Coming soon: train the final model and estimate insurance claim loss.</p></div>
    </section>
    <section class="dataset-note">
        <h3>Dataset</h3>
        <p>The app runs with the included sample <code>data/train.csv</code>. Replace it with the full Allstate <code>train.csv</code> later.</p>
    </section>
    """
    return page("Allstate Dashboard Home", body)


def load_page():
    raw = load_data()
    clean = clean_data(raw)
    raw_summary = {
        "Rows": raw.shape[0],
        "Columns": raw.shape[1],
        "Duplicate rows": int(raw.duplicated().sum()),
        "Missing values": int(raw.isnull().sum().sum()),
    }
    clean_summary = {
        "Rows": clean.shape[0],
        "Columns": clean.shape[1],
        "Duplicate rows": int(clean.duplicated().sum()),
        "Missing values": int(clean.isnull().sum().sum()),
    }
    loss_summary = clean["loss"].describe().round(2).to_dict() if "loss" in clean else {}
    body = f"""
    <h1>Data Loading &amp; Cleaning</h1>
    <div class="summary-grid">
        <div class="summary-box"><h3>Raw Dataset</h3>{mini_table(raw_summary)}</div>
        <div class="summary-box"><h3>Cleaned Dataset</h3>{mini_table(clean_summary)}</div>
        <div class="summary-box"><h3>Loss Summary</h3>{mini_table(loss_summary)}</div>
    </div>
    <h2>Preview</h2>
    {table_from_records(clean.head(10).to_dict(orient="records"), clean.columns)}
    """
    return page("Data Loading", body)


def eda_page():
    df = clean_data(load_data())
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = {}
    if "loss" in df.columns:
        corr = df[[column for column in numeric_cols if column != "id"]].corr()["loss"].drop("loss").abs().sort_values(ascending=False).head(10).round(3).to_dict()
    charts = svg_histogram(df["loss"], "Distribution of Loss", "#176b87")
    if "log_loss" in df.columns:
        charts += svg_histogram(df["log_loss"], "Distribution of Log Loss", "#d9822b")
    describe = df[numeric_cols].describe().T.round(2).reset_index().rename(columns={"index": "feature"})
    body = f"""
    <h1>Exploratory Data Analysis</h1>
    <p class="hint">Summary statistics, average loss, correlations, and chart output.</p>
    <div class="summary-grid">
        <div class="summary-box"><h3>Average Claim Loss</h3><p class="big-stat">{df["loss"].mean():.2f}</p></div>
        <div class="summary-box"><h3>Maximum Claim Loss</h3><p class="big-stat">{df["loss"].max():.2f}</p></div>
        <div class="summary-box"><h3>Top Numeric Correlations</h3>{mini_table(corr)}</div>
    </div>
    <h2>Charts</h2><div class="chart-grid">{charts}</div>
    <h2>Numeric Feature Summary</h2>
    {table_from_records(describe.to_dict(orient="records"), describe.columns)}
    """
    return page("EDA", body)


def preprocessing_page():
    raw = load_data()
    clean = clean_data(raw)
    no_null = raw.dropna()
    numeric_features = [column for column in clean.select_dtypes(include="number").columns if column not in {"id", "loss", "log_loss"}]
    train_rows = int(len(clean) * 0.8)
    sample = clean[numeric_features].head(10).round(3)
    report = {
        "Raw rows": raw.shape[0],
        "Raw null values": int(raw.isnull().sum().sum()),
        "Filled rows": clean.shape[0],
        "Null values after fill": int(clean.isnull().sum().sum()),
        "Rows after null removal": no_null.shape[0],
        "Regression target": "log_loss",
        "Feature count": len(numeric_features),
        "Train rows": train_rows,
        "Test rows": clean.shape[0] - train_rows,
    }
    body = f"""
    <h1>Data Preprocessing &amp; Linear Regression Prep</h1>
    <p class="hint">Compare null-value filling and null-value removing before creating numeric features for Linear Regression.</p>
    <div class="summary-grid"><div class="summary-box"><h3>Preprocessing Report</h3>{mini_table(report)}</div></div>
    <h2>Sample Numeric Features</h2>
    {table_from_records(sample.to_dict(orient="records"), sample.columns)}
    """
    return page("Preprocessing", body)


def feature_engg_page():
    df = clean_data(load_data())
    cont_cols = [column for column in df.columns if column.startswith("cont")]
    cat_cols = [column for column in df.columns if column.startswith("cat")][:6]
    scaled = df[cont_cols].copy()
    for column in cont_cols:
        min_value = scaled[column].min()
        max_value = scaled[column].max()
        scaled[column] = (scaled[column] - min_value) / (max_value - min_value)
    encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols).astype(int)
    body = f"""
    <h1>Feature Engineering</h1>
    <p class="hint">Model-ready transformations for Allstate claims severity features.</p>
    <h2>MinMaxScaler Normalization</h2>
    {table_from_records(scaled.head(20).round(4).to_dict(orient="records"), scaled.columns)}
    <h2>OneHotEncoder Categorical Encoding</h2>
    {table_from_records(encoded.head(20).to_dict(orient="records"), encoded.columns)}
    """
    return page("Feature Engineering", body)


class Handler(BaseHTTPRequestHandler):
    routes = {
        "/": home,
        "/load": load_page,
        "/eda": eda_page,
        "/preprocessing": preprocessing_page,
        "/feature_engg": feature_engg_page,
    }

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/static/style.css":
            self.send_response(200)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.end_headers()
            self.wfile.write((BASE_DIR / "static" / "style.css").read_bytes())
            return

        if path not in self.routes:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        try:
            content = self.routes[path]().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            content = page("Error", f"<div class='alert alert-error'><strong>Error:</strong> {html.escape(str(exc))}</div>").encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Serving Allstate dashboard at http://127.0.0.1:8000")
    server.serve_forever()
