import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime

# =========================
# 🎯 Page Configuration
# =========================
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 🎨 Custom CSS Styling
# =========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .risk-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    .status-badge-good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(56,239,125,0.4);
    }
    
    .status-badge-bad {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(235,51,73,0.4);
    }
    
    .info-card {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.95);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.95) !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #1e3c72 !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #333 !important;
        font-weight: 500;
    }
    
    .feature-box {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s;
    }
    
    .feature-box:hover {
        background: rgba(255,255,255,0.15);
        transform: translateY(-4px);
    }
    
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        padding: 2rem;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🧠 Load Model
# =========================
@st.cache_resource
def load_model():
    return pickle.load(open('credit_risk_model.pk1', 'rb'))

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found. Please ensure 'credit_risk_model.pk1' is in the correct directory.")
    st.stop()

# =========================
# 🖼️ Hero Section
# =========================
st.markdown("<h1 class='hero-title'>💳 Credit Risk Assessment System</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>AI-Powered Credit Scoring • Real-time Risk Analysis • Data-Driven Decisions</p>", unsafe_allow_html=True)

# =========================
# 📊 Main Content Area
# =========================
col_main1, col_main2 = st.columns([2, 1], gap="large")

with col_main2:
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Model Performance")
    
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.metric("Accuracy", "76%", "High")
        st.metric("ROC-AUC", "0.77", "Strong")
    with perf_col2:
        st.metric("Recall", "70%", "Good")
        st.metric("Model", "LogReg", "Balanced")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Input Data**: Enter applicant information
    2. **AI Analysis**: Model evaluates 20+ factors
    3. **Risk Score**: Instant credit risk prediction
    4. **Decision**: Low or High risk classification
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 🧩 Sidebar Inputs
# =========================
with st.sidebar:
    st.markdown("### 👤 Applicant Information")
    
    with st.expander("💰 Financial Details", expanded=True):
        duration = st.number_input("Loan Duration (months)", min_value=4, max_value=72, value=24, help="Duration of the loan in months")
        credit_amount = st.number_input("Credit Amount ($)", min_value=100, max_value=20000, value=2500, step=100)
        installment_commitment = st.slider("Installment Commitment", min_value=1, max_value=4, value=2)
        existing_credits = st.number_input("Existing Credits", min_value=1, max_value=5, value=1)
    
    with st.expander("🏦 Account Status", expanded=True):
        checking_status = st.selectbox("Checking Account", ['<0', '0<=X<200', '>=200', 'no checking'])
        savings_status = st.selectbox("Savings Account", ['<100', '100<=X<500', '500<=X<1000', '>=1000', 'no savings'])
        credit_history = st.selectbox("Credit History", ['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'])
    
    with st.expander("📋 Personal Information", expanded=True):
        age = st.number_input("Age", min_value=18, max_value=75, value=35)
        personal_status = st.selectbox("Marital Status", ['male single', 'female div/dep/mar', 'male mar/wid', 'male div/sep'])
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=1)
        residence_since = st.number_input("Residence Since (years)", min_value=1, max_value=10, value=2)
        own_telephone = st.selectbox("Own Telephone", ['yes', 'no'])
        foreign_worker = st.selectbox("Foreign Worker", ['yes', 'no'])
    
    with st.expander("💼 Employment & Assets", expanded=True):
        employment = st.selectbox("Employment Duration", ['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'])
        job = st.selectbox("Job Type", ['unskilled resident', 'unskilled non resident', 'skilled', 'high qualif/self emp/officer'])
        property_magnitude = st.selectbox("Property", ['car', 'life insurance', 'real estate', 'no known property'])
        housing = st.selectbox("Housing", ['own', 'rent', 'for free'])
    
    with st.expander("🎯 Loan Purpose", expanded=True):
        purpose = st.selectbox("Purpose", ['radio/tv', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliance', 'repairs', 'vacation/others'])
        other_parties = st.selectbox("Other Parties", ['none', 'guarantor', 'co applicant'])
        other_payment_plans = st.selectbox("Other Payment Plans", ['none', 'bank', 'stores'])
    
    st.markdown("---")
    predict_button = st.button("🔮 Analyze Credit Risk", use_container_width=True)

# =========================
# 📊 Prepare Input Data
# =========================
input_data = pd.DataFrame({
    'checking_status': [checking_status],
    'duration': [duration],
    'credit_history': [credit_history],
    'purpose': [purpose],
    'credit_amount': [credit_amount],
    'savings_status': [savings_status],
    'employment': [employment],
    'installment_commitment': [installment_commitment],
    'personal_status': [personal_status],
    'other_parties': [other_parties],
    'residence_since': [residence_since],
    'property_magnitude': [property_magnitude],
    'age': [age],
    'other_payment_plans': [other_payment_plans],
    'housing': [housing],
    'existing_credits': [existing_credits],
    'job': [job],
    'num_dependents': [num_dependents],
    'own_telephone': [own_telephone],
    'foreign_worker': [foreign_worker]
})

# =========================
# 🔮 Prediction Section
# =========================
with col_main1:
    if predict_button:
        with st.spinner("🔄 Analyzing credit risk..."):
            prediction_proba = model.predict_proba(input_data)[0]
            risk_bad = prediction_proba[0]
            risk_good = prediction_proba[1]
        
        st.markdown("<div class='risk-card'>", unsafe_allow_html=True)
        
        # Status Badge
        if risk_bad >= 0.5:
            st.markdown(f"<div class='status-badge-bad'>⚠️ HIGH CREDIT RISK</div>", unsafe_allow_html=True)
            risk_level = "High"
            risk_color = "#f45c43"
        else:
            st.markdown(f"<div class='status-badge-good'>✅ LOW CREDIT RISK</div>", unsafe_allow_html=True)
            risk_level = "Low"
            risk_color = "#38ef7d"
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{risk_good*100:.1f}%</div>
                    <div class='metric-label'>Good Credit Probability</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{risk_bad*100:.1f}%</div>
                    <div class='metric-label'>Bad Credit Probability</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            confidence = max(risk_good, risk_bad)
            confidence_level = "Very High" if confidence > 0.8 else "High" if confidence > 0.65 else "Moderate"
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value'>{confidence_level}</div>
                    <div class='metric-label'>Model Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_good * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Credit Score", 'font': {'size': 24, 'color': '#333'}},
            delta = {'reference': 50, 'increasing': {'color': "#38ef7d"}, 'decreasing': {'color': "#f45c43"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ccc",
                'steps': [
                    {'range': [0, 30], 'color': '#ffebee'},
                    {'range': [30, 50], 'color': '#fff3e0'},
                    {'range': [50, 70], 'color': '#e8f5e9'},
                    {'range': [70, 100], 'color': '#c8e6c9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#333", 'family': "Inter"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("### 💡 Recommendation")
        if risk_bad >= 0.5:
            st.error(f"""
            **High Risk Assessment** - This applicant shows a **{risk_bad*100:.1f}%** probability of default.
            
            ⚠️ **Suggested Actions:**
            - Require additional collateral
            - Consider higher interest rate
            - Request co-signer or guarantor
            - Conduct further due diligence
            """)
        else:
            st.success(f"""
            **Low Risk Assessment** - This applicant shows a **{risk_good*100:.1f}%** probability of good credit standing.
            
            ✅ **Suggested Actions:**
            - Approve loan application
            - Offer competitive interest rates
            - Consider premium customer benefits
            - Standard terms and conditions apply
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Breakdown Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Risk Distribution")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=['Good Credit', 'Bad Credit'],
            y=[risk_good * 100, risk_bad * 100],
            marker=dict(
                color=['#38ef7d', '#f45c43'],
                line=dict(color='#fff', width=2)
            ),
            text=[f"{risk_good*100:.1f}%", f"{risk_bad*100:.1f}%"],
            textposition='outside',
            textfont=dict(size=14, color='#fff', family='Inter')
        ))
        
        fig2.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=60),
            showlegend=False,
            font={'color': "#fff", 'family': "Inter"}
        )
        fig2.update_xaxes(
            title_text="Credit Category",
            title_font=dict(color='#fff'),
            tickfont=dict(color='#fff')
        )
        fig2.update_yaxes(
            title_text="Probability (%)",
            title_font=dict(color='#fff'),
            tickfont=dict(color='#fff'),
            range=[0, 100]
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        # Default view when no prediction yet
        st.markdown("<div class='risk-card'>", unsafe_allow_html=True)
        st.markdown("### 👋 Welcome to Credit Risk Assessment")
        st.markdown("""
        Fill in the applicant information in the sidebar and click **"Analyze Credit Risk"** to get started.
        
        Our AI model evaluates 20+ factors including:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - 💰 Financial history
            - 🏦 Account status
            - 💼 Employment details
            - 🏠 Housing situation
            """)
        with col2:
            st.markdown("""
            - 👤 Personal information
            - 🎯 Loan purpose
            - 📊 Credit history
            - 🔍 Payment behavior
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("<br>", unsafe_allow_html=True)
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            st.markdown("""
                <div class='feature-box'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>⚡</div>
                    <div style='color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>Instant Results</div>
                    <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Get predictions in milliseconds</div>
                </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
                <div class='feature-box'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🎯</div>
                    <div style='color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>76% Accuracy</div>
                    <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Reliable risk assessment</div>
                </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            st.markdown("""
                <div class='feature-box'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
                    <div style='color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>AI-Powered</div>
                    <div style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Advanced ML algorithms</div>
                </div>
            """, unsafe_allow_html=True)

# =========================
# 📚 Footer
# =========================
st.markdown("""
    <div class='footer'>
        <div style='margin-bottom: 1rem;'>
            <strong style='color: #ffffff; font-size: 1.1rem;'>Yessine Zouari</strong><br>
            <span style='color: rgba(255,255,255,0.8);'>Data Scientist & ML Engineer</span>
        </div>
        <div style='color: rgba(255,255,255,0.6); font-size: 0.9rem;'>
            Balanced Logistic Regression Model • Built with Scikit-learn & Streamlit • © 2025
        </div>
    </div>
""", unsafe_allow_html=True)