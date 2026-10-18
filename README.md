# Wine Cultivar Classifier 

A machine learning built with FastAPI and uvicorn that predicts the cultivar of a wine from 13 chemical measurements. The model is a Random Forest classifier trained on the scikit-learn Wine dataset, and the project includes automated training and testing with GitHub Actions.

## Features
* Random Forest classifier with feature scaling, trained on the Wine dataset (178 samples, 13 features, 3 cultivars)
* Accuracy and macro F1 score saved to `model/metrics.json` after training
* Prediction responses include the class name, confidence and probabilities for every class
* Batch prediction endpoint for up to 100 samples per request
* Input validation that rejects missing fields, wrong types and negative values
* Clear error responses when the model is missing or a request is invalid

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model-info` | Model metrics, feature names and class names |
| POST | `/predict` | Predict the cultivar for one wine sample |
| POST | `/predict-batch` | Predict cultivars for a list of samples |

## Project Structure

    model/wine_model.pkl             Trained model
    model/metrics.json               Model evaluation metrics
    src/data.py                      Dataset loading and splitting
    src/train.py                     Model training and evaluation
    src/predict.py                   Model loading and prediction
    src/main.py                      FastAPI app and endpoints
    tests/test_api.py                API and training tests
    .github/workflows/api_tests.yml  CI workflow

## Setup (macOS)

    python3 -m venv fastapi_env
    source fastapi_env/bin/activate
    pip install -r requirements.txt

## Usage

Train the model:

    python -m src.train

Start the API:

    uvicorn src.main:app --reload

Open http://127.0.0.1:8000/docs to explore and test the endpoints in the browser.

## Example Request

    POST /predict
    {
      "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
      "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06,
      "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
      "hue": 1.04, "od280_od315": 3.92, "proline": 1065.0
    }

Example response:

    {
      "class_id": 0,
      "class_name": "Cultivar A",
      "confidence": 0.98,
      "probabilities": {"Cultivar A": 0.98, "Cultivar B": 0.01, "Cultivar C": 0.01}
    }

## Testing

    pytest -v

The tests cover model training metrics, every endpoint, validation errors and batch prediction.

