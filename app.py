
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
BASE_DIR = Path(__file__).resolve().parent

model_data = joblib.load(BASE_DIR / "model.pkl")

model = model_data["model"]
feature_columns = model_data["feature_columns"]

# Create FastAPI application
app = FastAPI(
    title="EV Battery Failure Prediction API",
    description="Predict EV Battery Failure using Random Forest",
    version="1.0"
)

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Input class
class BatteryInput(BaseModel):
    battery_details: dict


# Prediction endpoint
@app.post("/predict")
def predict(data: BatteryInput):

    input_df = pd.DataFrame([data.battery_details])

    input_df = input_df.reindex(columns=feature_columns)

    probabilities = model.predict_proba(input_df)[0]

    failure_index = list(model.classes_).index(1)

    failure_probability = probabilities[failure_index]

    prediction = int(
        failure_probability >= model_data["threshold"]
    )

    return {
        "prediction": prediction,
        "result": "Battery Failure" if prediction == 1 else "No Battery Failure",
        "failure_probability": round(float(failure_probability * 100), 2)
    }


@app.get("/")
def home():
    return {
        "message": "EV Battery Failure Prediction API is running"
    }
