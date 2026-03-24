# 🚀 Customer Churn Prediction – Production-Ready ML System

An end-to-end machine learning system that predicts customer churn and simulates real-world deployment with monitoring, drift detection, and business decision support.

---

## 📌 Overview

This project goes beyond building a model — it demonstrates how ML systems operate in production.

It includes:
- Model training & experimentation
- API deployment
- Interactive dashboard
- Data drift detection
- Batch predictions
- Business-oriented risk classification

---

## ⚙️ Features

### 🔹 Machine Learning Pipeline
- Data preprocessing & feature engineering
- Multiple models trained:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Best model selection based on ROC-AUC
- Experiment tracking with MLflow

### 🔹 API (FastAPI)
- REST API for real-time predictions
- Input validation using Pydantic
- Returns:
  - Prediction
  - Churn probability
  - Drift analysis

### 🔹 Monitoring & Drift Detection
- Compares incoming data vs training distribution
- Detects feature drift using statistical differences
- Alerts when data deviates significantly

### 🔹 Streamlit Dashboard
- Real-time prediction interface
- Probability gauge visualization
- Drift status display
- Batch prediction via CSV upload
- Risk segmentation (Low / Medium / High)

### 🔹 Business Logic Layer
- Converts probabilities into actionable risk levels:
  - High Risk → Immediate action
  - Medium Risk → Marketing intervention
  - Low Risk → No action

---

## 🧠 Tech Stack

- Python
- Scikit-learn
- XGBoost
- MLflow
- FastAPI
- Streamlit
- Pandas / NumPy

---

## 🏗️ System Architecture
User (Streamlit Dashboard)
↓
FastAPI API (Prediction Service)
↓
ML Model (Trained Pipeline)
↓
Drift Detection + Risk Classification
↓
Response (Prediction + Insights)
---

## ▶️ How to Run

### 1️⃣ Start the API
```bash
uvicorn src.api.main:app --reload
### 2️⃣ Launch the Dashboard
streamlit run src/dashboard/app.py
📊 Example Output
Churn Probability: 0.78
Risk Level: High
Drift Status:
tenure → Drift detected
MonthlyCharges → OK
🎯 Key Takeaways
Built a production-style ML system, not just a model
Integrated monitoring and drift detection
Bridged ML outputs with business decisions
Simulated real-world ML deployment workflow
🚀 Future Improvements
Automated retraining pipeline
Advanced drift detection (KS test / Evidently)
Model performance monitoring over time
Cloud deployment (Docker + CI/CD)
👨‍💻 Author

Yessine Zouari