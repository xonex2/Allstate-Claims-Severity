# Allstate Claims Severity Flask Project

This project follows the same first-review dashboard flow as the reference project:

- Load and clean the Allstate claims CSV
- Show dataset summaries and preview rows
- Generate exploratory analysis charts
- Prepare model features with scaling and one-hot encoding
- Prepare numeric features for a future regression model

The final prediction page is intentionally left as "Coming soon" because this zip is completed only up to the same review stage as the reference project.

## Setup

```powershell
pip install -r requirements.txt
```

## Dataset

The project includes a small demo `data/train.csv` so it runs immediately.

For the full project later, replace it with the real Kaggle Allstate Claims Severity `train.csv`. The expected target column is:

```text
loss
```

## Run

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## If Flask Is Not Installed

You can still show the first-review web output with the included fallback server:

```powershell
python simple_server.py
```

Then open the same URL:

```text
http://127.0.0.1:8000
```
