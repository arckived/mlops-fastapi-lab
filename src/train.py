"""Train a Random Forest model on the Wine dataset and save it with its metrics."""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def train_model(n_estimators=100, random_state=42):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "model": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    results = train_model()
    print("Model trained and saved to", MODEL_PATH)
    print(json.dumps(results, indent=2))
