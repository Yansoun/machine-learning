import joblib
import pandas as pd
import json

# Load once
model = joblib.load("models/best_model.pkl")

with open("models/train_stats.json") as f:
    train_stats = json.load(f)


def predict_logic(data):
    input_dict = {
        "SeniorCitizen": data["SeniorCitizen"],
        "tenure": data["tenure"],
        "MonthlyCharges": data["MonthlyCharges"],
        "TotalCharges": data["TotalCharges"]
    }

    drift_report = {}

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        train_mean = train_stats[col]["mean"]
        train_std = train_stats[col]["std"]
        new_value = input_dict[col]

        diff = abs(new_value - train_mean)

        if diff > 2 * train_std:
            status = "DRIFT DETECTED 🚨"
        else:
            status = "OK"

        drift_report[col] = {
            "value": new_value,
            "train_mean": train_mean,
            "difference": diff,
            "status": status
        }

    df = pd.DataFrame([input_dict])

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "drift": drift_report,
        "status": status
    }