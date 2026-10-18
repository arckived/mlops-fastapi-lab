# Wine Cultivar Classifier

A machine learning web service that predicts the cultivar (grape variety group) of a wine from 13 chemical measurements. A Random Forest model is trained on the scikit-learn Wine dataset and served through a REST API built with FastAPI and uvicorn. Training and testing run automatically on GitHub Actions.

## Features
* Random Forest classifier with feature scaling, trained on the Wine dataset (178 samples, 13 features, 3 cultivars)
* Accuracy and macro F1 score saved to `model/metrics.json` after training
* Predictions include the class name, confidence and the probability of every class
* Batch predictions for up to 100 samples in one request
* Input validation that rejects missing fields, wrong types and negative values
* Clear error messages when the model is missing or a request is invalid

## Prerequisites
* Python 3.10 or newer (check with `python3 --version`)
* Git

## Installation

1. Clone the repository and move into it:

```bash
git clone https://github.com/arckived/mlops-fastapi-lab.git
cd mlops-fastapi-lab
```

2. Create and activate a virtual environment:

```bash
python3 -m venv fastapi_env
source fastapi_env/bin/activate
```

On Windows, activate it with `fastapi_env\Scripts\activate` instead.

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

All commands below must be run from the project root folder (`mlops-fastapi-lab`) with the virtual environment activated.

## Running the App

### 1. Train the model

```bash
python -m src.train
```

This trains the model, saves it to `model/wine_model.pkl` and saves its evaluation metrics to `model/metrics.json`. The accuracy and F1 score are printed in the terminal.

### 2. Start the server

```bash
uvicorn src.main:app --reload
```

The server runs at http://127.0.0.1:8000. Press `Control + C` to stop it.

### 3. Use the app in your browser

Open http://127.0.0.1:8000/docs to see the interactive documentation. To make a prediction:

1. Click **POST /predict**
2. Click **Try it out** (example values are already filled in)
3. Click **Execute** and scroll down to see the response

### 4. Or use the command line

With the server running, open a second terminal and run:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6, "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06, "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64, "hue": 1.04, "od280_od315": 3.92, "proline": 1065.0}'
```

Example response:

```json
{
  "class_id": 0,
  "class_name": "Cultivar A",
  "confidence": 0.98,
  "probabilities": {"Cultivar A": 0.98, "Cultivar B": 0.01, "Cultivar C": 0.01}
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model-info` | Model metrics, feature names and class names |
| POST | `/predict` | Predict the cultivar for one wine sample |
| POST | `/predict-batch` | Predict cultivars for a list of up to 100 samples |

### Input fields
Every prediction request needs these 13 numeric fields, all zero or greater: `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280_od315`, `proline`.

### Error responses
| Code | Meaning |
|------|---------|
| 400 | Batch request is empty or has more than 100 samples |
| 422 | A field is missing, has the wrong type or is negative |
| 503 | The model has not been trained yet (run `python -m src.train`) |

## Running the Tests

```bash
pytest -v
```

The output should end with `9 passed`. The tests check:
* Model training and its accuracy (at least 85%)
* The health check and model info endpoints
* Single and batch predictions
* Validation errors for missing fields, wrong types, negative values and empty batches

## Continuous Integration
On every push and pull request to `main`, the GitHub Actions workflow (`.github/workflows/api_tests.yml`) installs the dependencies, trains the model and runs the tests on Python 3.10, 3.11 and 3.12. The test reports and model metrics are uploaded as artifacts, which you can download from the run's page in the **Actions** tab.

## Project Structure

```
mlops-fastapi-lab/
├── .github/workflows/api_tests.yml   CI workflow
├── model/
│   ├── wine_model.pkl                Trained model
│   └── metrics.json                  Model evaluation metrics
├── src/
│   ├── data.py                       Dataset loading and splitting
│   ├── train.py                      Model training and evaluation
│   ├── predict.py                    Model loading and prediction
│   └── main.py                       FastAPI app and endpoints
├── tests/test_api.py                 Tests
├── requirements.txt
└── README.md
```
