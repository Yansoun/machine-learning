from fastapi import APIRouter
from src.api.services.predict import predict_logic
from src.api.services.explain import explain_customer_logic
from src.api.services.high_risk import high_risk_customers_logic
from src.api.models.schema import CustomerData

router = APIRouter()

@router.post("/predict")
def predict(data: CustomerData):
    return predict_logic(data)
@router.post("/explain_customer")
def explain_customer(data: CustomerData):
    return explain_customer_logic(data)
@router.get("/high_risk_customers")
def get_high_risk_customers(top_n: int = 10):
    return high_risk_customers_logic(top_n)