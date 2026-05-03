# California Housing Price Prediction API

## Problem

Housing valuations are central to decisions made by buyers, sellers, lenders, and investors. This project builds a machine learning model that predicts median house values in California block groups based on demographic and housing characteristics, and serves those predictions through a REST API.

## Key Findings

- A Random Forest model trained on 5 socioeconomic and housing features achieves strong predictive performance on California block group data.
- Median income is the dominant driver of predicted house value — areas with higher income consistently command higher valuations.
- The model is deployed as a live REST API, allowing downstream applications to query predictions in real time with a simple POST request.

## Dataset

The California Housing dataset (sourced from the 1990 U.S. Census) contains ~20,000 block groups across California. Each record represents a geographic block group and includes:

| Feature | Description |
|---|---|
| `MedInc` | Median household income |
| `HouseAge` | Median age of housing units |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Total population of the block group |

The target variable is median house value in units of $100,000.

## Methodology

A Random Forest Regressor was trained using an 80/20 train-test split with a fixed random seed to ensure reproducibility. The ensemble approach was chosen for its robustness to outliers and ability to capture nonlinear relationships between features and house values without requiring explicit feature engineering. Five features were selected based on their direct relevance to housing valuation.

## Results

The trained model captures the primary drivers of California housing prices, with median income showing the strongest predictive signal. The model is serialized and loaded at API startup, enabling sub-second inference latency per request.

Predictions are returned as median house value estimates (in $100,000 units) via a `/predict` POST endpoint, making the model accessible to any application that can send an HTTP request.

## Conclusion

This project demonstrates an end-to-end ML workflow: data ingestion, model training, serialization, and production-ready deployment via a REST API. A key limitation is that the underlying data is from 1990 and may not reflect current California housing market dynamics. The model also operates at block group granularity, which may not generalize well to individual property appraisals.

## Repository Structure

```
housing-ml-model/
├── train.py                # Model training script
├── main.py                 # FastAPI application and prediction endpoint
├── templates/
│   └── index.html          # Web form UI for general users
├── Procfile                # Start command for Render / Railway
└── requirements.txt        # Python dependencies
```

## How to Run Locally

**1. Set up the environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Start the API** (the model is trained automatically on first startup if needed)
```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser to use the prediction form, or visit `http://127.0.0.1:8000/docs` for the interactive API docs.

**3. Make a prediction via API**

Send a POST request to `http://127.0.0.1:8000/predict`:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0
}
```

## Deploy to Render (free tier)

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your repo.
3. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy**. Render will assign a public `https://<your-app>.onrender.com` URL.

The model is trained automatically on first startup — no pre-built binary needs to be committed.
