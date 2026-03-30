from fastapi import FastAPI
from src.api.routes import churn

app = FastAPI()

app.include_router(churn.router)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}