import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="California Housing Price Prediction API")

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "california_housing_model.joblib"

# Train the model at startup if the serialized file is not present
if not Path(MODEL_PATH).exists():
    print("Model file not found — training now…")
    subprocess.run([sys.executable, "train.py"], check=True)

model = joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    # Define the shape of our input data using Pydantic's BaseModel.
    # FastAPI will automatically validate the input and return an error
    # response if the data does not match the expected format.
    MedInc: float      # Median income in block group
    HouseAge: float    # Median house age in block group
    AveRooms: float    # Average number of rooms per household
    AveBedrms: float   # Average number of bedrooms per household
    Population: float  # Block group population

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
            }
        }
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the interactive prediction form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_price(features: HouseFeatures):
    """Predicts the median house value based on input features."""
    input_data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_data)
    return {"predicted_median_house_value": float(prediction[0])}