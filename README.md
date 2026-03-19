# Customer Churn ML System

This project predicts customer churn using machine learning and exposes predictions through an API and dashboard.

## Features
- Data preprocessing pipeline
- Model training (Logistic Regression, Random Forest, XGBoost)
- MLflow experiment tracking
- FastAPI prediction API
- Streamlit dashboard

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- Streamlit
- MLflow

## Run the API
uvicorn src.api.main:app --reload

## Run the dashboard
streamlit run src/dashboard/app.py
## Architecture

User (Streamlit Dashboard)
        ↓
FastAPI API
        ↓
ML Model (Scikit-learn)
        ↓
Prediction Response