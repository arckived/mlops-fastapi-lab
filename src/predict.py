"""Load the trained model and make predictions."""

import joblib

from src.train import MODEL_PATH

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python -m src.train")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_data(features):
    """features: list of samples, each a list of 13 values."""
    model = load_model()
    return model.predict(features), model.predict_proba(features)
