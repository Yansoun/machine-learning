import pandas as pd
import numpy as np
from src.api.services.predict import predict_logic


def high_risk_customers_logic(top_n=10):

    df = pd.read_csv("data/processed/X_test.csv")

    results = []

    # 1. Collect predictions FIRST
    for idx, row in df.iterrows():
        data = row.to_dict()
        pred = predict_logic(data)

        prob = pred["churn_probability"]

        results.append({
            "customer_id": idx,
            "churn_probability": prob
        })

    # 🛑 SAFETY CHECK
    if len(results) == 0:
        raise ValueError("No predictions generated")

    # 2. Compute thresholds AFTER loop
    probs = [r["churn_probability"] for r in results]

    threshold_high = np.percentile(probs, 90)
    threshold_medium = np.percentile(probs, 70)

    # 3. Assign risk levels
    for r in results:
        prob = r["churn_probability"]

        if prob >= threshold_high:
            r["risk_level"] = "high"
        elif prob >= threshold_medium:
            r["risk_level"] = "medium"
        else:
            r["risk_level"] = "low"

    # 4. Sort
    results = sorted(results, key=lambda x: x["churn_probability"], reverse=True)

    # 5. Return top N
    return results[:top_n]