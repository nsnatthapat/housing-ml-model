from fastapi import FastAPI # Framework for building APIs
from pydantic import BaseModel # For data validation and settings management using Python type annotations
import joblib # For saving and loading machine learning models
import pandas as pd # For data manipulation and analysis

app = FastAPI(title="California Housing Price Prediction API") # Create a FastAPI instance with a title for the API documentation

model = joblib.load('california_housing_model.joblib') # We already have a model so we load it here

class HouseFeatures(BaseModel): 
# Defne the shape of out input data using Pydantic's BaseModel. 
# IN other words, we are defining the expected features for our prediction endpoint. 
# If there is deviation, FastAPI will automatically handle validation and return an error response if the input data does not match the expected format.
    MedInc: float      # Median income in block group
    HouseAge: float    # Median house age in block group
    AveRooms: float    # Average number of rooms per household
    AveBedrms: float   # Average number of bedrooms per household
    Population: float  # Block group population
    
    class Config: # Sample
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0
            }
        }

@app.post("/predict") 
# The @ sign is a decorator in Python, which is a special type of function that modifies the behavior of another function. 
# In this case, @app.post("/predict") is a decorator provided by FastAPI that indicates that the following function (predict_price) should be called when an HTTP POST request is made to the "/predict" endpoint of the API.
# /predict is the endpoint where clients can send their data to get a prediction

def predict_price(features: HouseFeatures):
    """
    Predicts the median house value based on input features.
    """
    input_data = pd.DataFrame([features.dict()])
    
    prediction = model.predict(input_data)
    
    predicted_value = prediction[0]
    
    return {"predicted_median_house_value": predicted_value}

@app.get("/") # This is the root endpoint of the API, which can be used to check if the API is running and to provide a welcome message or basic information about the API.
def read_root():
    return {"message": "Welcome to the Housing Price Prediction API!"}