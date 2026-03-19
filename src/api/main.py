from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model once at startup
model = joblib.load("models/best_model.pkl")


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(data: CustomerData):

    # Create input dictionary
    input_dict = {
        "SeniorCitizen": data.SeniorCitizen,
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges
    }

    df = pd.DataFrame([input_dict])

    # Add missing columns (for encoded features)
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[model.feature_names_in_]

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }