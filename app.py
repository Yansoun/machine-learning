import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import plotly.graph_objects as go

# ---------------------------------------------
# ⚙️ Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="AI Food Calorie Predictor",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# 🎨 Custom CSS Styling - Clean & Modern
# ---------------------------------------------
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background */
    .stApp {
        background: #0a0e27;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0f1729 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e6ed !important;
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f9fafb !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Upload section */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Success/Warning/Error boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        border-left: 4px solid !important;
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Info box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #60a5fa !important;
        margin-top: 0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 🧠 Load Model and Class Indices
# ---------------------------------------------
import gdown
import os

MODEL_PATH = "food_classifier_model.h5"
MODEL_URL = "https://drive.google.com/file/d/1-xW38Bf56Irz9zzdqA78-pcuhBc_skn0/view?usp=sharing"  # Replace with your Google Drive file ID

@st.cache_resource
def load_model():
    try:
        # Download model from Google Drive if not exists
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📦 Downloading model from cloud... This may take a moment."):
                st.info("🌐 Model not found locally. Downloading from Google Drive...")
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded successfully!")
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.warning("💡 Make sure to replace 'YOUR_FILE_ID_HERE' with your actual Google Drive file ID")
        return None

@st.cache_data
def load_class_indices():
    try:
        with open("class_indices.json", "r") as f:
            class_indices = json.load(f)
        return class_indices
    except FileNotFoundError:
        st.error("⚠️ class_indices.json not found!")
        return None
    except json.JSONDecodeError:
        st.error("⚠️ class_indices.json is empty or corrupted. Please check the file.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading class indices: {e}")
        return None

model = load_model()
class_indices = load_class_indices()

if class_indices:
    class_labels = {v: k for k, v in class_indices.items()}
else:
    class_labels = {}
    st.stop()

# ---------------------------------------------
# 🍽️ Enhanced Nutritional Database
# ---------------------------------------------
nutrition_dict = {
    "Bread": {"calories": 265, "protein": 9, "carbs": 49, "fat": 3, "emoji": "🍞"},
    "Dairy product": {"calories": 180, "protein": 7, "carbs": 11, "fat": 10, "emoji": "🧀"},
    "Dessert": {"calories": 340, "protein": 4, "carbs": 55, "fat": 12, "emoji": "🍰"},
    "Egg": {"calories": 155, "protein": 13, "carbs": 1, "fat": 11, "emoji": "🥚"},
    "Fried food": {"calories": 312, "protein": 15, "carbs": 25, "fat": 18, "emoji": "🍟"},
    "Meat": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15, "emoji": "🥩"},
    "Noodles-Pasta": {"calories": 130, "protein": 5, "carbs": 25, "fat": 1, "emoji": "🍝"},
    "Rice": {"calories": 130, "protein": 3, "carbs": 28, "fat": 0, "emoji": "🍚"},
    "Seafood": {"calories": 206, "protein": 20, "carbs": 0, "fat": 12, "emoji": "🦐"},
    "Soup": {"calories": 85, "protein": 4, "carbs": 12, "fat": 2, "emoji": "🍲"},
    "Vegetable-Fruit": {"calories": 60, "protein": 2, "carbs": 14, "fat": 0, "emoji": "🥗"}
}

# ---------------------------------------------
# 📊 Sidebar
# ---------------------------------------------
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🍱 AI Food Calorie Predictor")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### 📱 About")
    st.markdown("""
    This application uses deep learning to:
    - 🔍 Identify food categories
    - 📊 Estimate nutritional content
    - 💡 Provide health insights
    """)
    
    st.markdown("---")
    
    st.markdown("#### ⚙️ Model Info")
    st.markdown(f"""
    - **Categories:** {len(class_labels)} food types
    - **Input:** 128×128 RGB images
    - **Framework:** TensorFlow/Keras
    """)
    
    st.markdown("---")
    
    st.markdown("#### 🎯 Supported Categories")
    for category, info in nutrition_dict.items():
        st.markdown(f"{info['emoji']} {category}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 0.85rem; color: #6b7280;'>
    <strong>v1.0.0</strong><br>
    Built by Yessine Zouari
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------
# 🎨 Main Header
# ---------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='margin-bottom: 0;'>🍱 AI Food Analyzer</h1>
            <p style='font-size: 1.3rem; color: #9ca3af; margin-top: 0.5rem;'>
                Instant Nutritional Intelligence
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------
# 📤 Upload Section
# ---------------------------------------------
uploaded_file = st.file_uploader(
    "📸 Upload Food Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG (Max 200MB)"
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------
# 🔍 Analysis Section
# ---------------------------------------------
if uploaded_file is not None and model is not None:
    
    col_img, col_results = st.columns([1, 1], gap="large")
    
    with col_img:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📷 Uploaded Image")
        image_obj = Image.open(uploaded_file)
        st.image(image_obj, use_container_width=True)
        st.markdown(f"<p style='text-align: center; color: #6b7280; font-size: 0.9rem;'>{uploaded_file.name} • {uploaded_file.size / 1024:.1f} KB</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Preprocess
    img = image_obj.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    with st.spinner("🔍 Analyzing image with AI..."):
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = round(100 * np.max(prediction), 2)
        class_name = class_labels.get(predicted_class, "Unknown")
        
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [(class_labels[i], prediction[0][i] * 100) for i in top_3_indices]

    nutrition = nutrition_dict.get(class_name, {})
    
    with col_results:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### {nutrition.get('emoji', '🍽️')} Detection Result")
        
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #6366f1; margin: 0; font-size: 2rem;'>{class_name}</h2>
            <p style='font-size: 1.2rem; color: #9ca3af; margin: 0.5rem 0;'>
                Confidence: <strong style='color: #a855f7;'>{confidence}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence >= 80:
            st.success("✅ High confidence prediction")
        elif confidence >= 60:
            st.info("ℹ️ Moderate confidence prediction")
        else:
            st.warning("⚠️ Low confidence - try a clearer image")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Nutritional Info
    st.markdown("### 📊 Nutritional Information (per 100g)")
    st.markdown("<br>", unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    
    metrics = [
        ("Calories", f"{nutrition.get('calories', 'N/A')}", "kcal", "🔥"),
        ("Protein", f"{nutrition.get('protein', 'N/A')}", "g", "💪"),
        ("Carbs", f"{nutrition.get('carbs', 'N/A')}", "g", "🌾"),
        ("Fat", f"{nutrition.get('fat', 'N/A')}", "g", "🧈")
    ]
    
    for col, (label, value, unit, emoji) in zip(metric_cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 2rem;'>{emoji}</div>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label} ({unit})</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📈 Detailed Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("#### 🎯 Top Predictions")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=[pred[1] for pred in top_3_predictions],
            y=[pred[0] for pred in top_3_predictions],
            orientation='h',
            marker=dict(
                color=['#6366f1', '#8b5cf6', '#a855f7'],
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            ),
            text=[f"{pred[1]:.1f}%" for pred in top_3_predictions],
            textposition='outside',
            textfont=dict(color='#e5e7eb', size=12)
        ))
        fig1.update_layout(
            height=280,
            xaxis_title="Confidence (%)",
            xaxis_range=[0, 100],
            margin=dict(l=20, r=60, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(size=11, color="#9ca3af"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with viz_col2:
        if nutrition:
            st.markdown("#### 🥗 Macronutrient Distribution")
            fig2 = go.Figure(data=[go.Pie(
                labels=['Protein', 'Carbs', 'Fat'],
                values=[
                    nutrition.get('protein', 0),
                    nutrition.get('carbs', 0),
                    nutrition.get('fat', 0)
                ],
                hole=.5,
                marker=dict(colors=['#6366f1', '#8b5cf6', '#a855f7']),
                textfont=dict(color='#e5e7eb', size=12)
            )])
            fig2.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                font=dict(size=11, color="#9ca3af"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Health Insights
    st.markdown("### 💡 Health Insights")
    st.markdown("<br>", unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        <div class='info-box'>
            <h4>📝 Important Notes</h4>
            <ul style='color: #9ca3af; margin: 0; padding-left: 1.5rem;'>
                <li>Values are estimates per 100g</li>
                <li>Actual values vary by preparation</li>
                <li>Consult a nutritionist for dietary advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        calories = nutrition.get('calories', 0)
        if calories:
            if calories > 250:
                level, color = "High", "#ef4444"
            elif calories > 150:
                level, color = "Moderate", "#f59e0b"
            else:
                level, color = "Low", "#10b981"
            
            st.markdown(f"""
            <div class='info-box'>
                <h4>⚡ Calorie Level</h4>
                <p style='font-size: 1.8rem; color: {color}; font-weight: 700; margin: 0.5rem 0;'>
                    {level}
                </p>
                <p style='color: #9ca3af; margin: 0;'>
                    {calories} kcal per 100g serving
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Action Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Analyze Another Image", use_container_width=True):
            st.rerun()

elif uploaded_file is None:
    # Welcome Screen
    st.markdown("""
    <div class='card' style='text-align: center; padding: 4rem 2rem;'>
        <div style='font-size: 5rem; margin-bottom: 1rem;'>📸</div>
        <h2 style='color: #f9fafb; margin-bottom: 1rem;'>Get Started</h2>
        <p style='color: #9ca3af; font-size: 1.1rem; margin-bottom: 2rem;'>
            Upload a food image to receive instant nutritional analysis
        </p>
        <p style='color: #6b7280; font-size: 0.95rem;'>
            Supported: JPG, JPEG, PNG • Max size: 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("### ✨ Key Features")
    st.markdown("<br>", unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    features = [
        ("🎯", "Accurate Recognition", "AI-powered classification across 11 food categories"),
        ("⚡", "Instant Analysis", "Get results in seconds with detailed nutritional data"),
        ("📊", "Visual Insights", "Interactive charts and confidence metrics")
    ]
    
    for col, (emoji, title, desc) in zip([feat_col1, feat_col2, feat_col3], features):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>{emoji}</div>
                <h4 style='color: #f9fafb; margin-bottom: 0.5rem;'>{title}</h4>
                <p style='color: #9ca3af; font-size: 0.9rem; margin: 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("⚠️ Model could not be loaded. Please check if 'food_classifier_model.h5' exists.")

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.05);'>
        <p style='margin: 0; font-size: 0.9rem;'>
            Powered by <strong style='color: #9ca3af;'>TensorFlow</strong> • 
            <strong style='color: #9ca3af;'>Keras</strong> • 
            <strong style='color: #9ca3af;'>Streamlit</strong>
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
            Developed by <strong style='color: #9ca3af;'>Yessine Zouari</strong>
        </p>
    </div>
""", unsafe_allow_html=True)
