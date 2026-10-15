from flask import Flask, render_template

from ml.data_loader import clean_data, get_data_summary, load_data
from ml.eda import generate_charts, get_summary_stats
from ml.feature_engg import normalization, one_hot_encoding, standard_scaling
from ml.preprocessing import build_preprocessing_report


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/load")
def load_page():
    try:
        raw_df = load_data()
        cleaned_df = clean_data(raw_df)
        raw_summary = get_data_summary(raw_df)
        cleaned_summary = get_data_summary(cleaned_df)
        preview = cleaned_df.head(10)

        return render_template(
            "data_loading.html",
            raw_summary=raw_summary,
            cleaned_summary=cleaned_summary,
            preview_cols=preview.columns,
            preview_rows=preview.to_dict(orient="records"),
        )
    except Exception as exc:
        return render_template("data_loading.html", error=str(exc))


@app.route("/eda")
def eda_page():
    try:
        df = clean_data(load_data())
        stats = get_summary_stats(df)
        chart_files = generate_charts(df)

        return render_template(
            "eda.html",
            stats=stats,
            chart_files=chart_files,
        )
    except Exception as exc:
        return render_template("eda.html", error=str(exc))


@app.route("/feature_engg")
def feature_engg_page():
    try:
        df = clean_data(load_data())
        minmax_train_df, minmax_test_df = normalization(df)
        standard_train_df, standard_test_df = standard_scaling(df)
        encoded_train_df, encoded_test_df = one_hot_encoding(df)

        return render_template(
            "feature_engg.html",
            columns=minmax_train_df.columns,
            train_rows=minmax_train_df.head(20).round(4).values,
            test_rows=minmax_test_df.head(20).round(4).values,
            standard_columns=standard_train_df.columns,
            standard_train_rows=standard_train_df.head(20).round(4).values,
            standard_test_rows=standard_test_df.head(20).round(4).values,
            encoded_columns=encoded_train_df.columns,
            encoded_train_rows=encoded_train_df.head(20).astype(int).values,
            encoded_test_rows=encoded_test_df.head(20).astype(int).values,
        )
    except Exception as exc:
        return render_template("feature_engg.html", error=str(exc))


@app.route("/preprocessing")
def preprocessing_page():
    try:
        report = build_preprocessing_report(load_data())
        return render_template("preprocessing.html", report=report)
    except Exception as exc:
        return render_template("preprocessing.html", error=str(exc))


if __name__ == "__main__":
    app.run(debug=True, port=8000)
