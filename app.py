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
# 🎨 Enhanced Custom CSS Styling
# ------------------------- #
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Input sections */
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 2px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.8rem 2rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Input fields */
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🏷️ Page Header
# ------------------------- #
st.markdown("<div class='main-title'>📊 Customer Churn Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>✨ AI-Powered Insights to Predict Customer Retention ✨</div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- #
# 📋 Sidebar Information
# ------------------------- #
with st.sidebar:
    st.markdown("### ℹ️ About This App")
    st.markdown("---")
    st.markdown("""
    This intelligent system predicts whether a telecom customer is likely to **churn (leave)** or **stay** based on their service usage and demographic profile.
    
    #### 🔍 How It Works:
    1. **Enter** customer details
    2. **Process** data through StandardScaler
    3. **Predict** using XGBoost ML model
    4. **Get** actionable insights
    
    #### 📈 Model Performance:
    - **Algorithm:** XGBoost Classifier
    - **Accuracy:** ~79%
    - **Preprocessing:** StandardScaler
    
    #### 👨‍💻 Developer:
    **Yessine Zouari**  
    Machine Learning Engineer
    
    📧 yessinezouari@example.com
    """)
    st.markdown("---")
    st.markdown("🚀 *Part of ML Journey 2024*")

# ------------------------- #
# 🧾 Feature Inputs
# ------------------------- #
st.markdown("<div class='section-title'>🧩 Customer Information</div>", unsafe_allow_html=True)

# Basic Information
st.markdown("#### 👤 Demographics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    gender_Male = st.selectbox("👤 Gender", ["Female", "Male"])
with col2:
    SeniorCitizen = st.selectbox("👴 Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col3:
    Partner_Yes = st.selectbox("💑 Partner", ["No", "Yes"])
with col4:
    Dependents_Yes = st.selectbox("👨‍👩‍👧 Dependents", ["No", "Yes"])

st.markdown("---")

# Account Information
st.markdown("#### 📋 Account Details")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("📅 Tenure (months)", 0, 72, 12)
with col2:
    MonthlyCharges = st.number_input("💵 Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
with col3:
    TotalCharges = st.number_input("💰 Total Charges ($)", 0.0, 10000.0, 500.0, step=50.0)

st.markdown("---")

# Services
st.markdown("#### 📞 Phone & Internet Services")
col1, col2, col3 = st.columns(3)

with col1:
    PhoneService_Yes = st.selectbox("📱 Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("📞 Multiple Lines", ["No phone service", "No", "Yes"])

with col2:
    InternetService = st.selectbox("🌐 Internet Service", ["No", "DSL", "Fiber optic"])
    OnlineSecurity = st.selectbox("🔒 Online Security", ["No", "Yes", "No internet service"])

with col3:
    OnlineBackup = st.selectbox("💾 Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("🛡️ Device Protection", ["No", "Yes", "No internet service"])

st.markdown("---")

# Additional Services
st.markdown("#### 🎬 Additional Services")
col1, col2, col3 = st.columns(3)

with col1:
    TechSupport = st.selectbox("🔧 Tech Support", ["No", "Yes", "No internet service"])
with col2:
    StreamingTV = st.selectbox("📺 Streaming TV", ["No", "Yes", "No internet service"])
with col3:
    StreamingMovies = st.selectbox("🎬 Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")

# Billing Information
st.markdown("#### 💳 Billing & Contract")
col1, col2, col3 = st.columns(3)

with col1:
    Contract = st.selectbox("📝 Contract Type", ["Month-to-month", "One year", "Two year"])
with col2:
    PaperlessBilling_Yes = st.selectbox("📄 Paperless Billing", ["No", "Yes"])
with col3:
    PaymentMethod = st.selectbox("💳 Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

st.markdown("---")

# ------------------------- #
# 🧮 Prediction Button & Logic
# ------------------------- #
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 PREDICT CHURN PROBABILITY")

if predict_button:
    try:
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

        # Prepare input array
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

        # Scale data
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        # Create beautiful result display
        if prediction == 1:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                            padding: 2rem; border-radius: 15px; text-align: center; 
                            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);'>
                    <h1 style='color: white; font-size: 3rem; margin: 0;'>⚠️</h1>
                    <h2 style='color: white; margin: 0.5rem 0;'>High Churn Risk</h2>
                    <h1 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>{prob*100:.1f}%</h1>
                    <p style='color: rgba(255,255,255,0.9); margin: 0;'>Probability of Churning</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                st.warning("💡 **Recommended Actions:**")
                st.markdown("""
                - 🎁 Offer personalized retention incentives
                - 💬 Schedule customer satisfaction call
                - 📧 Send targeted promotional offers
                - 🔄 Review contract terms and upgrade options
                """)
        else:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); 
                            padding: 2rem; border-radius: 15px; text-align: center; 
                            box-shadow: 0 8px 25px rgba(86, 171, 47, 0.4);'>
                    <h1 style='color: white; font-size: 3rem; margin: 0;'>✅</h1>
                    <h2 style='color: white; margin: 0.5rem 0;'>Low Churn Risk</h2>
                    <h1 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>{(1-prob)*100:.1f}%</h1>
                    <p style='color: rgba(255,255,255,0.9); margin: 0;'>Probability of Staying</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                st.success("💡 **Recommended Actions:**")
                st.markdown("""
                - 🌟 Continue providing excellent service
                - 📊 Monitor engagement metrics regularly
                - 🎉 Reward loyalty with exclusive benefits
                - 📱 Encourage product adoption and usage
                """)

    except Exception as e:
        st.error(f"❌ **Prediction Error:** {str(e)}")
        st.info(f"🔍 Expected features: {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'Unknown'}")
        st.warning("⚠️ Please ensure all input features match the trained model configuration.")

# ------------------------- #
# 🧠 Footer
# ------------------------- #
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem 0; font-size: 0.9rem;'>
    <p style='margin: 0.5rem 0;'>⚡ Powered by <strong>XGBoost</strong> & <strong>Streamlit</strong></p>
    <p style='margin: 0.5rem 0;'>🎨 Designed & Developed by <strong>Yessine Zouari</strong></p>
    <p style='margin: 0.5rem 0;'>© 2024 | Machine Learning Portfolio Project</p>
</div>
""", unsafe_allow_html=True)
