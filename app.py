import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================
# 🎯 Page Configuration
# =========================
st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🎯",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        color: rgba(255,255,255,0.95);
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .segment-card {
        background: rgba(255,255,255,0.98);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 1.5rem 0;
    }
    
    .input-card {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .cluster-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.8rem;
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
        margin: 1rem 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.95;
    }
    
    .persona-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 16px;
        padding: 2rem;
        border-left: 5px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .persona-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .persona-title {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .persona-desc {
        color: #444;
        font-size: 1rem;
        line-height: 1.7;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
        transition: all 0.3s;
    }
    
    .feature-item:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-text {
        color: white;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        transition: all 0.3s;
        box-shadow: 0 6px 20px rgba(245,87,108,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(245,87,108,0.6);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] p {
        color: rgba(255,255,255,0.9) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
    }
    
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Sidebar inputs */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div,
    section[data-testid="stSidebar"] .stNumberInput > div {
        background: rgba(255,255,255,0.95);
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(102,126,234,0.2) !important;
        border: 1px solid rgba(102,126,234,0.4);
        color: #ffffff !important;
    }
    
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.9);
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    .stats-card {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(79,172,254,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🧠 Load Models
# =========================
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pk1')
    kmeans = joblib.load('kmeans_model.pk1')
    return scaler, kmeans

try:
    scaler, kmeans = load_models()
except:
    st.error("⚠️ Model files not found. Please ensure 'scaler.pk1' and 'kmeans_model.pk1' are in the correct directory.")
    st.stop()

# =========================
# 🖼️ Hero Section
# =========================
st.markdown("<h1 class='hero-title'>🎯 Customer Segmentation AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Intelligent K-Means Clustering • Real-time Persona Analysis • Data-Driven Marketing</p>", unsafe_allow_html=True)

# =========================
# 📊 Sidebar - Input Section
# =========================
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff !important; text-align: center; margin-bottom: 1rem;'>👤 Customer Profile</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 1.5rem;'>Enter demographic and behavioral data</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("📋 Demographics", expanded=True):
        gender = st.selectbox("Gender", ["Male", "Female"], help="Biological gender of the customer")
        age = st.slider("Age", 18, 70, 30, help="Customer's age in years")
    
    with st.expander("💰 Financial Profile", expanded=True):
        income = st.number_input(
            "Annual Income (k$)", 
            min_value=10, 
            max_value=150, 
            value=60,
            step=5,
            help="Annual income in thousands of dollars"
        )
    
    with st.expander("🛍️ Shopping Behavior", expanded=True):
        spending = st.slider(
            "Spending Score (1–100)", 
            1, 100, 50,
            help="Score based on customer behavior and purchasing data"
        )
    
    st.markdown("---")
    predict_button = st.button("🔮 Analyze Customer Segment")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #ffffff !important;'>📊 Model Info</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: rgba(102,126,234,0.2); padding: 1rem; border-radius: 12px; border: 1px solid rgba(102,126,234,0.4);'>
        <p style='color: #ffffff; margin-bottom: 0.5rem;'><strong>Algorithm:</strong> K-Means Clustering</p>
        <p style='color: #ffffff; margin-bottom: 0.5rem;'><strong>Features:</strong> 4</p>
        <ul style='color: rgba(255,255,255,0.9); margin-left: 1rem;'>
            <li>Gender</li>
            <li>Age</li>
            <li>Annual Income</li>
            <li>Spending Score</li>
        </ul>
        <p style='color: #ffffff; margin-top: 0.5rem; margin-bottom: 0;'><strong>Clusters:</strong> 5 Segments</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 📊 Main Content
# =========================
col1, col2 = st.columns([1.5, 1], gap="large")

# Encode Gender
gender_encoded = 1 if gender == "Male" else 0

# Create input data
input_data = pd.DataFrame({
    'Genre': [gender_encoded],
    'Age': [age],
    'Annual Income (k$)': [income],
    'Spending Score (1-100)': [spending]
})

# Cluster descriptions with detailed info
cluster_info = {
    0: {
        "name": "Mature Average Spenders",
        "icon": "🧓",
        "color": "#8e44ad",
        "description": "Older, steady customers with moderate spending habits. Loyal but cautious with their purchases.",
        "characteristics": [
            "Age: 40-60 years",
            "Income: Moderate to High",
            "Spending: Conservative",
            "Behavior: Loyal, Value-driven"
        ],
        "strategy": "Focus on loyalty programs, quality over quantity, personalized service, and relationship building."
    },
    1: {
        "name": "High-Income Savers",
        "icon": "💰",
        "color": "#27ae60",
        "description": "Wealthy but cautious spenders with strong saving habits. High potential for premium upselling.",
        "characteristics": [
            "Age: 30-50 years",
            "Income: Very High",
            "Spending: Low to Moderate",
            "Behavior: Quality-focused, Research-oriented"
        ],
        "strategy": "Target with premium products, emphasize value and exclusivity, provide detailed information and guarantees."
    },
    2: {
        "name": "Young Enthusiastic Shoppers",
        "icon": "🛍️",
        "color": "#e74c3c",
        "description": "Energetic, middle-income customers who love to shop. High engagement and spending frequency.",
        "characteristics": [
            "Age: 18-35 years",
            "Income: Moderate",
            "Spending: High",
            "Behavior: Impulsive, Trend-following"
        ],
        "strategy": "Use social media marketing, flash sales, trending products, and gamification to engage."
    },
    3: {
        "name": "Affluent Spenders",
        "icon": "👑",
        "color": "#f39c12",
        "description": "Upper-middle income customers with strong purchasing power. Consistent high-value transactions.",
        "characteristics": [
            "Age: 25-45 years",
            "Income: High",
            "Spending: Very High",
            "Behavior: Brand-conscious, Experience-seeking"
        ],
        "strategy": "Premium products, VIP treatment, exclusive offers, and exceptional customer experience."
    },
    4: {
        "name": "Budget Conscious Youth",
        "icon": "🎯",
        "color": "#3498db",
        "description": "Younger customers with lower income but active spending. Price-sensitive but engaged.",
        "characteristics": [
            "Age: 18-30 years",
            "Income: Low to Moderate",
            "Spending: Moderate",
            "Behavior: Value-seeking, Digital-savvy"
        ],
        "strategy": "Discount offers, payment plans, student discounts, and affordable product lines."
    }
}

with col1:
    if predict_button:
        with st.spinner("🔄 Analyzing customer profile..."):
            # Scale and predict
            scaled_input = scaler.transform(input_data)
            cluster = kmeans.predict(scaled_input)[0]
            
            info = cluster_info[cluster]
        
        st.markdown("<div class='segment-card'>", unsafe_allow_html=True)
        
        # Cluster badge
        st.markdown(f"<div class='cluster-badge'>{info['icon']} CLUSTER {cluster}</div>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: {info['color']}; margin-top: 1rem;'>{info['name']}</h2>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Customer metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.markdown(f"""
                <div class='metric-box' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                    <div class='metric-value'>{gender}</div>
                    <div class='metric-label'>Gender</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class='metric-box' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                    <div class='metric-value'>{age}</div>
                    <div class='metric-label'>Age</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
                <div class='metric-box' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                    <div class='metric-value'>${income}k</div>
                    <div class='metric-label'>Income</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(f"""
                <div class='metric-box' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
                    <div class='metric-value'>{spending}</div>
                    <div class='metric-label'>Spending</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Description
        st.markdown(f"### 📝 Segment Profile")
        st.markdown(f"<p style='font-size: 1.1rem; color: #444; line-height: 1.8;'>{info['description']}</p>", unsafe_allow_html=True)
        
        # Characteristics
        st.markdown("### 🎯 Key Characteristics")
        char_col1, char_col2 = st.columns(2)
        for i, char in enumerate(info['characteristics']):
            with char_col1 if i % 2 == 0 else char_col2:
                st.markdown(f"✓ **{char}**")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Marketing Strategy
        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
        st.markdown("### 💡 Recommended Marketing Strategy")
        st.markdown(f"<p style='font-size: 1.05rem; line-height: 1.7;'>{info['strategy']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Radar Chart - Customer Profile
        st.markdown("### 📊 Customer Profile Visualization")
        
        categories = ['Age Score', 'Income Level', 'Spending Score', 'Engagement']
        values = [
            (age / 70) * 100,  # Normalize age
            (income / 150) * 100,  # Normalize income
            spending,
            ((spending + income/2) / 100) * 100  # Engagement composite
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f'rgba{tuple(int(info["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
            line=dict(color=info["color"], width=3),
            name='Customer Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                bgcolor='rgba(255,255,255,0.9)'
            ),
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#333')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Default state - Show overview
        st.markdown("<div class='segment-card'>", unsafe_allow_html=True)
        st.markdown("### 👋 Welcome to Customer Segmentation AI")
        st.markdown("""
        Use the sidebar to enter customer details and discover which segment they belong to.
        Our AI-powered K-Means clustering model identifies 5 distinct customer personas.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("<div class='feature-grid'>", unsafe_allow_html=True)
        
        features = [
            ("🎯", "5 Segments"),
            ("⚡", "Real-time"),
            ("🧠", "AI-Powered"),
            ("📊", "Data-Driven")
        ]
        
        for icon, text in features:
            st.markdown(f"""
                <div class='feature-item'>
                    <div class='feature-icon'>{icon}</div>
                    <div class='feature-text'>{text}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='stats-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 All Customer Segments")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show all personas
    for cluster_id, info in cluster_info.items():
        with st.expander(f"{info['icon']} {info['name']}", expanded=False):
            st.markdown(f"**Cluster {cluster_id}**")
            st.markdown(info['description'])
            st.markdown("**Target:** " + info['characteristics'][3])

# =========================
# 📚 Footer
# =========================
st.markdown("""
    <div class='footer'>
        <div style='margin-bottom: 1rem;'>
            <strong style='color: #ffffff; font-size: 1.2rem;'>Yessine Zouari</strong><br>
            <span style='color: rgba(255,255,255,0.9);'>Machine Learning Engineer & Data Scientist</span>
        </div>
        <div style='color: rgba(255,255,255,0.7); font-size: 0.95rem;'>
            K-Means Clustering • Built with Scikit-learn & Streamlit • © 2025
        </div>
    </div>
""", unsafe_allow_html=True)