import streamlit as st
import joblib
import numpy as np

# ------------------------- #
# 🎯 App Configuration
# ------------------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
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
# 🎨 Professional CSS Styling
# ------------------------- #
st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean white background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
    }
    
    /* Professional header */
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 3rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 400;
    }
    
    /* Section styling */
    .section-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Input styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(59, 130, 246, 0.3);
    }
    
    .stButton button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #111827;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li {
        color: #4b5563;
        font-size: 0.9rem;
    }
    
    /* Result cards */
    .result-high-risk {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.2);
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .result-probability {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Recommendations */
    .recommendations {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    
    .recommendations h4 {
        color: #1e40af;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .recommendations ul {
        margin: 0;
        padding-left: 1.25rem;
    }
    
    .recommendations li {
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.875rem;
        padding: 2rem 0;
        margin-top: 4rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #e5e7eb;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🏷️ Professional Header
# ------------------------- #
st.markdown("""
    <div class='main-header'>
        <div class='main-title'>Customer Churn Prediction</div>
        <div class='main-subtitle'>Machine Learning-Powered Customer Retention Analytics</div>
    </div>
""", unsafe_allow_html=True)

# ------------------------- #
# 📋 Professional Sidebar
# ------------------------- #
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This application uses machine learning to predict customer churn probability 
    based on service usage patterns and demographic information.
    """)
    
    st.markdown("---")
    
    st.markdown("### Model Information")
    st.markdown("""
    - **Algorithm:** XGBoost Classifier
    - **Accuracy:** 79%
    - **Features:** 30 input variables
    - **Preprocessing:** StandardScaler
    """)
    
    st.markdown("---")
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Enter customer details in each section
    2. Click the "Analyze Churn Risk" button
    3. Review the prediction and recommendations
    """)
    
    st.markdown("---")
    
    st.markdown("### Developer")
    st.markdown("""
    **Yessine Zouari**  
    Machine Learning Engineer
    
    yessinezouari@example.com
    """)

# ------------------------- #
# 📊 Input Sections
# ------------------------- #

# Demographics Section
st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    gender_Male = st.selectbox("Gender", ["Female", "Male"])
with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col3:
    Partner_Yes = st.selectbox("Partner", ["No", "Yes"])
with col4:
    Dependents_Yes = st.selectbox("Dependents", ["No", "Yes"])

st.markdown("<br>", unsafe_allow_html=True)

# Account Information
st.markdown("<div class='section-header'>Account Information</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with col2:
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
with col3:
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)

st.markdown("<br>", unsafe_allow_html=True)

# Services
st.markdown("<div class='section-header'>Phone & Internet Services</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    PhoneService_Yes = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

with col2:
    InternetService = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

with col3:
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

st.markdown("<br>", unsafe_allow_html=True)

# Additional Services
st.markdown("<div class='section-header'>Additional Services</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
with col2:
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
with col3:
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("<br>", unsafe_allow_html=True)

# Billing
st.markdown("<div class='section-header'>Billing & Contract</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col2:
    PaperlessBilling_Yes = st.selectbox("Paperless Billing", ["No", "Yes"])
with col3:
    PaymentMethod = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

st.markdown("<br><br>", unsafe_allow_html=True)

# ------------------------- #
# 🔍 Prediction Button
# ------------------------- #
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Analyze Churn Risk")

if predict_button:
    try:
        # Convert inputs
        gender_Male = 1 if gender_Male == "Male" else 0
        Partner_Yes = 1 if Partner_Yes == "Yes" else 0
        Dependents_Yes = 1 if Dependents_Yes == "Yes" else 0
        PhoneService_Yes = 1 if PhoneService_Yes == "Yes" else 0
        PaperlessBilling_Yes = 1 if PaperlessBilling_Yes == "Yes" else 0

        MultipleLines_No_phone_service = 1 if MultipleLines == "No phone service" else 0
        MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0

        InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0
        InternetService_No = 1 if InternetService == "No" else 0

        OnlineSecurity_No_internet = 1 if OnlineSecurity == "No internet service" else 0
        OnlineSecurity_Yes = 1 if OnlineSecurity == "Yes" else 0

        OnlineBackup_No_internet = 1 if OnlineBackup == "No internet service" else 0
        OnlineBackup_Yes = 1 if OnlineBackup == "Yes" else 0

        DeviceProtection_No_internet = 1 if DeviceProtection == "No internet service" else 0
        DeviceProtection_Yes = 1 if DeviceProtection == "Yes" else 0

        TechSupport_No_internet = 1 if TechSupport == "No internet service" else 0
        TechSupport_Yes = 1 if TechSupport == "Yes" else 0

        StreamingTV_No_internet = 1 if StreamingTV == "No internet service" else 0
        StreamingTV_Yes = 1 if StreamingTV == "Yes" else 0

        StreamingMovies_No_internet = 1 if StreamingMovies == "No internet service" else 0
        StreamingMovies_Yes = 1 if StreamingMovies == "Yes" else 0

        Contract_One_year = 1 if Contract == "One year" else 0
        Contract_Two_year = 1 if Contract == "Two year" else 0

        PaymentMethod_Credit_card = 1 if PaymentMethod == "Credit card (automatic)" else 0
        PaymentMethod_Electronic_check = 1 if PaymentMethod == "Electronic check" else 0
        PaymentMethod_Mailed_check = 1 if PaymentMethod == "Mailed check" else 0

        # Prepare input
        input_data = np.array([[
            SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
            gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes,
            MultipleLines_No_phone_service, MultipleLines_Yes,
            InternetService_Fiber_optic, InternetService_No,
            OnlineSecurity_No_internet, OnlineSecurity_Yes,
            OnlineBackup_No_internet, OnlineBackup_Yes,
            DeviceProtection_No_internet, DeviceProtection_Yes,
            TechSupport_No_internet, TechSupport_Yes,
            StreamingTV_No_internet, StreamingTV_Yes,
            StreamingMovies_No_internet, StreamingMovies_Yes,
            Contract_One_year, Contract_Two_year,
            PaperlessBilling_Yes,
            PaymentMethod_Credit_card, PaymentMethod_Electronic_check, PaymentMethod_Mailed_check
        ]])

        # Scale and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display results
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div class='result-high-risk'>
                    <div class='result-title'>High Churn Risk</div>
                    <div class='result-probability'>{prob*100:.1f}%</div>
                    <div class='result-label'>Probability of Churn</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='recommendations'>
                    <h4>Recommended Actions</h4>
                    <ul>
                        <li>Initiate immediate customer retention outreach</li>
                        <li>Offer personalized incentives or service upgrades</li>
                        <li>Schedule customer satisfaction review call</li>
                        <li>Review contract terms and pricing options</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low-risk'>
                    <div class='result-title'>Low Churn Risk</div>
                    <div class='result-probability'>{(1-prob)*100:.1f}%</div>
                    <div class='result-label'>Probability of Retention</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='recommendations'>
                    <h4>Recommended Actions</h4>
                    <ul>
                        <li>Continue providing excellent customer service</li>
                        <li>Monitor engagement metrics quarterly</li>
                        <li>Introduce loyalty rewards program</li>
                        <li>Encourage product adoption and feature usage</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.info("Please ensure all inputs are correctly filled and model files are available.")

# ------------------------- #
# Footer
# ------------------------- #
st.markdown("""
<div class='footer'>
    Built with Streamlit and XGBoost | © 2024 Yessine Zouari
</div>
""", unsafe_allow_html=True)
