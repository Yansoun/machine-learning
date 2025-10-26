# 📈 Sales Prediction using Machine Learning

A complete end-to-end machine learning solution to predict future product sales based on historical data, seasonality, and key business factors.

---

## 🚀 Overview
This project aims to **forecast product sales** to help businesses optimize inventory, marketing, and logistics.  
By analyzing trends in historical sales data, the model provides accurate predictions that can guide decision-making.

---

## 📊 Dataset
**Source:** Kaggle - Sales Forecasting Dataset  
**Description:**
- Thousands of sales records across multiple stores and product categories  
- Features: store type, location, promotions, date, etc.  
- Target variable: `Sales`  

Key preprocessing:
- Handled missing values and outliers  
- Converted date columns and extracted time-based features (month, year, etc.)  
- Applied feature scaling and one-hot encoding for categorical data

---

## 🧩 Model & Methodology
- Models tested: **Linear Regression**, **Random Forest**, **XGBoost**
- Evaluation metric: Mean Absolute Error (MAE)
- Best model: **XGBoost** (lowest MAE and highest R² score)
- Final model tuned with GridSearchCV for maximum precision

---

## 💻 Tech Stack
- **Python**  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit  

---

## ⚙️ How to Run Locally
```bash
git clone https://github.com/Yansoun/machine_learning/sales-prediction.git
cd sales-prediction
pip install -r requirements.txt
streamlit run app.py
🌐 Deployment

Live demo: https://machine-learning-sales-prediction.streamlit.app

📈 Results

Model Accuracy (R²): 0.91

MAE: 12.3% average deviation

Streamlit app for real-time prediction of sales based on input features

🔮 Future Improvements

Add deep learning models for time series forecasting (LSTM)

Include economic and marketing trend data

Enhance visualization dashboard

👤 Author

Yessine Zouari
Machine Learning Engineer
📧 https://www.linkedin.com/in/yessine-zouari-a84b9a349