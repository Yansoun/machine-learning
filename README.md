## 📞  Customer Churn Prediction**

```markdown
# 📞 Customer Churn Prediction

An end-to-end machine learning project predicting customer churn for telecom companies using structured data and advanced classification algorithms.

---

## 🚀 Overview
The objective is to identify **customers likely to leave (churn)** based on usage, contracts, and payment behavior.  
This enables companies to proactively retain customers through targeted offers or services.

---

## 📊 Dataset
**Source:** Telco Customer Churn Dataset (Kaggle)  
**Description:**
- 7,043 customer records  
- Features: demographic info, account details, service usage  
- Target variable: `Churn` (Yes/No)

Key preprocessing:
- One-hot encoding for categorical features  
- Missing value handling  
- Feature scaling using StandardScaler  
- Class imbalance addressed using weighted models  

---

## 🧩 Model & Methodology
- Models tested: **Logistic Regression**, **Random Forest**, **SVM**, **XGBoost**
- Best performer: **XGBoost (Balanced)**  
- Evaluation metrics: Precision, Recall, F1-score (focus on recall for churn class)
- Streamlit app for live churn prediction

---

## 💻 Tech Stack
- **Python**
- **Libraries:** Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Streamlit  

---

## ⚙️ How to Run Locally
```bash
git clone https://github.com/Yansoun/machine-learning/customer-churn.git
cd customer-churn
pip install -r requirements.txt
streamlit run app.py
🌐 Deployment

Live demo: https://machine-learning-churn.streamlit.app

📈 Results

Logistic Regression Accuracy: 80.7%

XGBoost Balanced Model: 76% accuracy, improved recall for churners

Streamlit UI for real-time churn prediction

🔮 Future Improvements

Integrate customer retention simulation dashboard

Add automated report generation for insights

Explore deep learning models for improved recall

👤 Author

Yessine Zouari
Machine Learning Engineer
📧 https://www.linkedin.com/in/yessine-zouari-a84b9a349
#MachineLearning #DataScience #XGBoost #Streamlit #Python #AI #PredictiveAnalytics #CustomerRetention