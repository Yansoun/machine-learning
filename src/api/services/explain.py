from src.api.services.predict import predict_logic

def explain_customer_logic(data):

    # 1. Get prediction result
    result = predict_logic(data)

    probability = result["churn_probability"]
    drift = result["drift"]

    # 2. Compute risk level
    if probability > 0.7:
        risk = "high"
    elif probability > 0.4:
        risk = "medium"
    else:
        risk = "low"

    # 3. Detect top factors from drift
    factors = []

    for feature, info in drift.items():
        diff = info["difference"]

        if diff > 0:
            if feature == "tenure" and info["value"] < info["train_mean"]:
                factors.append(("low tenure", diff))

            elif feature == "MonthlyCharges" and info["value"] > info["train_mean"]:
                factors.append(("high monthly charges", diff))

            elif feature == "TotalCharges" and info["value"] > info["train_mean"]:
                factors.append(("high total charges", diff))

    # Sort by biggest difference
    # 3. Detect top factors
    factors = []

    for feature, info in drift.items():
        diff = info["difference"]
        value = info["value"]
        mean = info["train_mean"]

        if feature == "tenure" and value < mean:
            factors.append(("tenure", diff))

        elif feature == "MonthlyCharges" and value > mean:
            factors.append(("MonthlyCharges", diff))

        elif feature == "TotalCharges" and value > mean:
            factors.append(("TotalCharges", diff))

    # Sort by importance
    factors = sorted(factors, key=lambda x: x[1], reverse=True)

    # Extract top features FIRST
    top_factors = [f[0] for f in factors[:3]]
    mapping = {
            "tenure": "low tenure",
            "MonthlyCharges": "high monthly charges",
            "TotalCharges": "high total charges"
        }

    top_factors = [mapping.get(f, f) for f in top_factors]

    # 4. Recommendation logic
    if risk == "high":
        recommendation = "Offer discount or long-term plan"
    elif risk == "medium":
        recommendation = "Engage with retention campaign"
    else:
        recommendation = "No action needed"

    return {
        "churn_probability": probability,
        "risk_level": risk,
        "top_factors": top_factors,
        "recommendation": recommendation
    }