import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------
# ⚙️ Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="AI Fruit Freshness Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------
# 🎨 Custom CSS Styling
# ---------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #0f0f0f;
    }
    
    .block-container {
        padding: 3rem 2rem;
        max-width: 1400px;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .stFileUploader {
        background: #1a1a1a;
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
    }
    
    .stFileUploader label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    
    .metric-container {
        background: #0f0f0f;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #333;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    .status-badge-fresh {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.2rem;
    }
    
    .status-badge-rotten {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        font-size: 1.2rem;
    }
    
    .feature-card {
        background: #1a1a1a;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #333;
        text-align: center;
        transition: all 0.3s;
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-4px);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #a0a0a0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .tech-badge {
        background: #1a1a1a;
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #667eea;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 4rem;
        border-top: 1px solid #333;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .uploadedFile {
        border-radius: 12px;
        overflow: hidden;
    }
    
    div[data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 🧠 Load the trained model
# ---------------------------------------------
@st.cache_resource
def load_model(path="fruit_classifier_model.h5"):
    model = tf.keras.models.load_model(path)
    return model

model = load_model("fruit_classifier_model.h5")

# Try to read model output shape to detect binary vs multiclass
out_shape = model.output_shape  # e.g. (None, 1) or (None, 6)
is_binary_model = (len(out_shape) == 2 and out_shape[-1] == 1)

# Default class label mapping
# For binary (sigmoid) we assume: 0 -> Fresh, 1 -> Rotten
# For multiclass, provide class names in the correct order
if is_binary_model:
    class_labels = ["Fresh", "Rotten"]
else:
    class_labels = ["freshapples", "freshbanana", "freshoranges",
                    "rottenapples", "rottenbanana", "rottenoranges"]

# ---------------------------------------------
# Helper: preprocess image
# ---------------------------------------------
def preprocess_pil_image(pil_img, target_size=(128, 128)):
    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = image.img_to_array(pil_img).astype("float32")
    arr /= 255.0  # CRUCIAL: normalize same as training
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------------------------
# 🖼️ Hero Section
# ---------------------------------------------
st.markdown("<h1 class='hero-title'>AI Fruit Freshness Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Powered by Deep Learning • Real-time Analysis • Professional Grade</p>", unsafe_allow_html=True)

# ---------------------------------------------
# 📤 Upload Section
# ---------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a fruit image (Apple, Banana, Orange, or any fruit)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process image
    image_obj = Image.open(uploaded_file)
    
    # Preprocess with proper normalization
    img_array = preprocess_pil_image(image_obj, target_size=(128, 128))
    
    # Prediction
    with st.spinner("🔄 Analyzing image..."):
        preds = model.predict(img_array, verbose=0)
    
    # Interpret prediction robustly:
    if is_binary_model:
        # sigmoid output: preds shape (1,1) with value ~ probability of class 1
        prob_class1 = float(preds[0][0])   # probability of class index 1
        prob_class0 = 1.0 - prob_class1
        probs = np.array([prob_class0, prob_class1])
        predicted_index = int(prob_class1 >= 0.5)
        confidence = round(100.0 * probs[predicted_index], 2)
        result_label = class_labels[predicted_index]
        is_fresh = (predicted_index == 0)
    else:
        # softmax/multiclass output: preds shape (1, n_classes)
        probs = preds[0]
        predicted_index = int(np.argmax(probs))
        confidence = round(100.0 * float(probs[predicted_index]), 2)
        result_label = class_labels[predicted_index]
        is_fresh = "fresh" in result_label.lower()
    
    # Layout
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("### 📸 Uploaded Image")
        st.image(image_obj, use_column_width=True)
    
    with col2:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        
        # Status Badge
        badge_class = "status-badge-fresh" if is_fresh else "status-badge-rotten"
        status_emoji = "✓" if is_fresh else "✗"
        st.markdown(f"<div class='{badge_class}'>{status_emoji} {result_label.capitalize()}</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{confidence}%</div>
                    <div class='metric-label'>Confidence</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{"Fresh" if is_fresh else "Rotten"}</div>
                    <div class='metric-label'>Status</div>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            confidence_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
            st.markdown(f"""
                <div class='metric-container'>
                    <div class='metric-value'>{confidence_level}</div>
                    <div class='metric-label'>Certainty</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recommendation
        if is_fresh:
            st.success("🍃 This fruit appears fresh and safe to consume!")
        else:
            st.error("⚠️ This fruit shows signs of spoilage. Consider discarding it.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Confidence Chart
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Confidence Distribution")
    
    # Prepare chart data
    if is_binary_model:
        x_labels = class_labels
        y_vals = (probs * 100).tolist()
    else:
        x_labels = class_labels
        y_vals = (probs * 100).tolist()
    
    colors = ["#38ef7d" if i == predicted_index else "#667eea" for i in range(len(x_labels))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_vals,
        marker=dict(
            color=colors,
            line=dict(color='#0f0f0f', width=2)
        ),
        text=[f"{v:.1f}%" for v in y_vals],
        textposition='outside',
        textfont=dict(color='#ffffff', size=12)
    ))
    
    fig.update_layout(
        plot_bgcolor='#0f0f0f',
        paper_bgcolor='#0f0f0f',
        xaxis=dict(
            title="Classification",
            titlefont=dict(color='#ffffff'),
            tickfont=dict(color='#a0a0a0'),
            gridcolor='#333'
        ),
        yaxis=dict(
            title="Confidence (%)",
            titlefont=dict(color='#ffffff'),
            tickfont=dict(color='#a0a0a0'),
            gridcolor='#333'
        ),
        height=450,
        showlegend=False,
        hovermode='x unified',
        margin=dict(t=20, b=60, l=60, r=20)
    )
    fig.update_yaxes(range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)

else:
    # Features Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### ⚡ Key Features")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🚀</div>
                <div class='feature-title'>Lightning Fast</div>
                <div class='feature-desc'>Get results in milliseconds with optimized CNN architecture</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🎯</div>
                <div class='feature-title'>High Accuracy</div>
                <div class='feature-desc'>Trained on thousands of images for reliable predictions</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🔒</div>
                <div class='feature-title'>Privacy First</div>
                <div class='feature-desc'>All processing happens locally, your data stays secure</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>📱</div>
                <div class='feature-title'>Universal</div>
                <div class='feature-desc'>Works seamlessly on any device or platform</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Tech Stack
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center;'>
            <span class='tech-badge'>TensorFlow</span>
            <span class='tech-badge'>Keras</span>
            <span class='tech-badge'>Convolutional Neural Networks</span>
            <span class='tech-badge'>Computer Vision</span>
            <span class='tech-badge'>Streamlit</span>
            <span class='tech-badge'>Python</span>
            <span class='tech-badge'>Plotly</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Model Info
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='feature-card' style='text-align: left;'>
                <div style='color: #667eea; font-size: 1.5rem; margin-bottom: 1rem;'>📊 Model Architecture</div>
                <div class='feature-desc'>
                    • <strong>Type:</strong> Convolutional Neural Network<br>
                    • <strong>Input Shape:</strong> 128x128x3 RGB<br>
                    • <strong>Classes:</strong> Binary (Fresh/Rotten)<br>
                    • <strong>Framework:</strong> TensorFlow/Keras<br>
                    • <strong>Preprocessing:</strong> Normalized to [0, 1]
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-card' style='text-align: left;'>
                <div style='color: #667eea; font-size: 1.5rem; margin-bottom: 1rem;'>🍎 Supported Fruits</div>
                <div class='feature-desc'>
                    • <strong>Apples:</strong> Fresh & Rotten classification<br>
                    • <strong>Bananas:</strong> Ripeness detection<br>
                    • <strong>Oranges:</strong> Quality assessment<br>
                    • <strong>Universal:</strong> Works with various fruits<br><br>
                    <em>More fruits coming soon!</em>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <div style='margin-bottom: 1rem;'>
            <strong style='color: #ffffff; font-size: 1.1rem;'>Yessine Zouari</strong><br>
            <span style='color: #667eea;'>Machine Learning Engineer</span>
        </div>
        <div style='color: #666; font-size: 0.9rem;'>
            Built with TensorFlow & Streamlit • © 2025
        </div>
    </div>
""", unsafe_allow_html=True)
