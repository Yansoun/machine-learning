import streamlit as st
import joblib
import numpy as np

# ------------------------- #
# 🎯 App Configuration
# ------------------------- #
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- #
# 🧠 Load Model and Scaler
# ------------------------- #
@st.cache_resource
def load_assets():
    model = joblib.load("xgboost_churn_model.pk1")
    scaler = joblib.load("scaler.pk1")
    return model, scaler

model, scaler = load_assets()

# ------------------------- #
# 🎨 Custom CSS Styling
# ------------------------- #
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #145a8c;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🏷️ Page Header
# ------------------------- #
st.markdown("<div class='main-title'>📊 Customer Churn Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict whether a customer is likely to churn using AI-powered insights</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- #
# 📋 Sidebar Information
# ------------------------- #
with st.sidebar:
    st.header("ℹ️ About the App")
    st.write("""
    This app predicts whether a telecom customer is likely to **churn (leave)** or **stay** based on service and demographic data.
    
    **How it works:**
    - You enter customer details below
    - The input data is scaled and processed
    - The XGBoost model predicts churn probability
    
    **Model Details:**
    - Algorithm: XGBoost Classifier
    - Accuracy: ~79%
    - Scaler: StandardScaler
    
    This project was built by **Yessine Zouari** as part of his Machine Learning roadmap 🚀
    """)
    st.markdown("---")
    st.write("📧 Contact: yessinezouari@example.com")

# ------------------------- #
# 🧾 Feature Inputs
# ------------------------- #
st.subheader("🧩 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

with col2:
    gender_Male = st.selectbox("Gender", ["Female", "Male"])
    Partner_Yes = st.selectbox("Has Partner?", ["No", "Yes"])
    Dependents_Yes = st.selectbox("Has Dependents?", ["No", "Yes"])
    PhoneService_Yes = st.selectbox("Phone Service?", ["No", "Yes"])

with col3:
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    PaperlessBilling_Yes = st.selectbox("Paperless Billing?", ["No", "Yes"])

# ------------------------- #
# 🧮 Convert Inputs to Model Format
# ------------------------- #
if st.button("🔍 Predict Churn"):
    # Convert categorical values to numerical
    gender_Male = 1 if gender_Male == "Male" else 0
    Partner_Yes = 1 if Partner_Yes == "Yes" else 0
    Dependents_Yes = 1 if Dependents_Yes == "Yes" else 0
    PhoneService_Yes = 1 if PhoneService_Yes == "Yes" else 0
    PaperlessBilling_Yes = 1 if PaperlessBilling_Yes == "Yes" else 0

    # Handle multi-category variables
    MultipleLines_No_phone_service = 1 if MultipleLines == "No phone service" else 0
    MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0

    InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0
    InternetService_No = 1 if InternetService == "No" else 0

    Contract_One_year = 1 if Contract == "One year" else 0
    Contract_Two_year = 1 if Contract == "Two year" else 0

    PaymentMethod_Credit_card = 1 if PaymentMethod == "Credit card (automatic)" else 0
    PaymentMethod_Electronic_check = 1 if PaymentMethod == "Electronic check" else 0
    PaymentMethod_Mailed_check = 1 if PaymentMethod == "Mailed check" else 0

    # Prepare input array
    input_data = np.array([[
        SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
        gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes,
        MultipleLines_No_phone_service, MultipleLines_Yes,
        InternetService_Fiber_optic, InternetService_No,
        Contract_One_year, Contract_Two_year,
        PaperlessBilling_Yes,
        PaymentMethod_Credit_card, PaymentMethod_Electronic_check, PaymentMethod_Mailed_check
    ]])

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    if prediction == 1:
        st.error(f"❌ The model predicts this customer is **likely to churn** (Probability: {prob*100:.2f}%)")
        st.info("💡 Recommendation: Offer retention incentives or personalized offers.")
    else:
        st.success(f"✅ The model predicts this customer will **stay** (Probability: {(1-prob)*100:.2f}%)")
        st.info("💡 Recommendation: Keep providing value and track engagement periodically.")

# ------------------------- #
# 🧠 Footer
# ------------------------- #
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem 0;'>
Built with ❤️ using <strong>Streamlit</strong> & <strong>XGBoost</strong> | Project by <strong>Yessine Zouari</strong>
</div>
""", unsafe_allow_html=True)
