# 💳 Credit Risk Prediction

An end-to-end machine learning project that predicts **creditworthiness of loan applicants** using structured financial data and advanced classification algorithms.

---

## 🚀 Overview
The goal of this project is to assess **whether a client is likely to default** or maintain good credit standing based on demographic and financial indicators.  
This helps financial institutions make smarter, data-driven loan approval decisions.

---

## 📊 Dataset
**Source:** German Credit Data (built-in dataset from Scikit-learn)  
**Description:**
- 1,000 customer records  
- Features include: checking account status, loan duration, credit history, purpose, credit amount, employment, property, housing, etc.  
- Target variable: `class` (Good / Bad credit risk)

Key preprocessing steps:
- One-hot encoding for categorical features  
- Feature scaling for numerical columns  
- Handling imbalance with class weights and SMOTE  

---

## 🧩 Model & Methodology
Models tested:  
- **Logistic Regression (Balanced)**  
- **Random Forest**  
- **Gradient Boosting**  
- **XGBoost**

After thorough evaluation:  
✅ **Balanced Logistic Regression** achieved the best trade-off between accuracy and recall for bad credit detection.

| Metric | Score |
|:--|:--|
| Accuracy | 0.76 |
| Recall (Bad Credit) | 0.70 |
| ROC-AUC | 0.77 |

This model was chosen for deployment due to its **interpretability, stability, and high recall** — critical in financial risk contexts.

---

## 💻 Tech Stack
- **Python**
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit, imbalanced-learn  

---

## ⚙️ How to Run Locally
```bash
git clone https://github.com/Yansoun/machine-learning/tree/project6.git
cd credit-risk-prediction
pip install -r requirements.txt
streamlit run app.py
