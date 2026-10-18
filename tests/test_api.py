import pytest
from fastapi.testclient import TestClient

from src.data import CLASS_NAMES
from src.main import EXAMPLE, app
from src.train import MODEL_PATH, train_model

SAMPLE_B = {
    "alcohol": 14.13,
    "malic_acid": 4.1,
    "ash": 2.74,
    "alcalinity_of_ash": 24.5,
    "magnesium": 96.0,
    "total_phenols": 2.05,
    "flavanoids": 0.76,
    "nonflavanoid_phenols": 0.56,
    "proanthocyanins": 1.35,
    "color_intensity": 9.2,
    "hue": 0.61,
    "od280_od315": 1.6,
    "proline": 560.0,
}


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists():
        train_model()
    return TestClient(app)


def test_train_model_metrics():
    metrics = train_model()
    assert metrics["accuracy"] >= 0.85
    assert metrics["model"] == "RandomForestClassifier"


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info(client):
    data = client.get("/model-info").json()
    assert len(data["features"]) == 13
    assert data["classes"] == CLASS_NAMES
    assert data["metrics"]["accuracy"] >= 0.85


def test_predict_valid(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    data = response.json()
    assert data["class_id"] == 0
    assert data["class_name"] == "Cultivar A"
    assert 0 <= data["confidence"] <= 1
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=0.01)


def test_predict_missing_field(client):
    bad = dict(EXAMPLE)
    del bad["alcohol"]
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_negative_value(client):
    bad = dict(EXAMPLE, proline=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_wrong_type(client):
    bad = dict(EXAMPLE, hue="high")
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_batch(client):
    response = client.post("/predict-batch", json=[EXAMPLE, SAMPLE_B])
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    for r in results:
        assert r["class_name"] in CLASS_NAMES


def test_predict_batch_empty(client):
    assert client.post("/predict-batch", json=[]).status_code == 400
