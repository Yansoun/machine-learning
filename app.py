import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
        border-radius: 10px;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #145a8c;
        transform: scale(1.02);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header { 
            font-size: 2rem; 
        }
        .sub-header { 
            font-size: 1rem; 
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("sentiment_model.pk1")
    vectorizer = joblib.load("vectorizer.pk1")
    return model, vectorizer

model, vectorizer = load_models()

# Model accuracy (replace with your actual model accuracy)
MODEL_ACCURACY = 0.892  # 89.2% - Update this with your actual model performance

# Initialize session state for review counter
if 'review_count' not in st.session_state:
    st.session_state.review_count = 0

# Sidebar - About Section
with st.sidebar:
    st.markdown("### 📖 About This App")
    st.write("""
    This **Sentiment Analysis App** uses machine learning to classify text reviews 
    as either **Positive** or **Negative**.
    
    **How It Works:**
    - Enter any product review or text
    - The model analyzes the sentiment
    - Get instant classification results
    
    **Dataset:**
    The model was trained on the **IMDB Movie Reviews Dataset**, which contains 
    50,000 labeled reviews for sentiment classification tasks.
    
    **Model Details:**
    - Algorithm: Logistic Regression
    - Vectorizer: TF-IDF
    - Features: Text-based sentiment patterns
    """)
    
    st.markdown("---")
    st.markdown("### 🧪 Try These Examples")
    
    example_reviews = {
        "Positive": "This product exceeded my expectations! Absolutely love it and would recommend to everyone.",
        "Negative": "Terrible quality. Waste of money. Would not recommend to anyone."
    }
    
    for sentiment, review in example_reviews.items():
        if st.button(f"📝 {sentiment} Example", key=sentiment):
            st.session_state.example_text = review
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.write("**Yessine Zouari**")
    st.write("Machine Learning Engineer")

# Main content
st.markdown('<div class="main-header">🎭 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze the sentiment of any text instantly using AI</div>', unsafe_allow_html=True)

st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Enter Your Review")
    
    # Check if example text exists in session state
    default_text = st.session_state.get('example_text', '')
    user_input = st.text_area(
        label="Type or paste your review below:",
        height=200,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout...",
        value=default_text,
        label_visibility="collapsed"
    )
    
    # Clear example text after use
    if 'example_text' in st.session_state and default_text:
        del st.session_state.example_text
    
    # Predict button
    predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Quick Stats")
    st.metric("Model Accuracy", f"{MODEL_ACCURACY*100:.1f}%")
    st.metric("Reviews Analyzed", f"{st.session_state.review_count:,}")
    st.metric("Avg Response Time", "< 1s")

# Prediction logic
if predict_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing!")
    else:
        with st.spinner("🤔 Analyzing sentiment..."):
            # Transform and predict
            X_input = vectorizer.transform([user_input])
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]
            confidence = round(max(proba) * 100, 2)
            
            # Increment review counter
            st.session_state.review_count += 1
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Analysis Results")
            
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                if prediction == 1:
                    st.success("### ✅ Positive Sentiment Detected!")
                    st.markdown("""
                    <div style='text-align: center; font-size: 4rem;'>
                        😊
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("💡 This review expresses positive emotions and satisfaction.")
                else:
                    st.error("### ❌ Negative Sentiment Detected!")
                    st.markdown("""
                    <div style='text-align: center; font-size: 4rem;'>
                        😞
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("💡 This review expresses negative emotions or dissatisfaction.")
                
                # Confidence score
                st.markdown("#### 🎯 Confidence Score")
                st.progress(max(proba))
                st.caption(f"Model confidence: **{confidence}%**")
            
            # Word count and character count
            st.markdown("---")
            st.markdown("### 📝 Text Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.markdown("#### 🧾 Words")
                st.markdown(f"<h2 style='text-align: center;'>{len(user_input.split())}</h2>", unsafe_allow_html=True)
            with stat_col2:
                st.markdown("#### 🔤 Characters")
                st.markdown(f"<h2 style='text-align: center;'>{len(user_input)}</h2>", unsafe_allow_html=True)
            with stat_col3:
                sentiment_label = "Positive" if prediction == 1 else "Negative"
                sentiment_emoji = "😊" if prediction == 1 else "😞"
                st.markdown("#### 💬 Sentiment")
                st.markdown(f"<h2 style='text-align: center;'>{sentiment_emoji}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'><strong>{sentiment_label}</strong></p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='margin: 0;'>🚀 Built with Streamlit & Scikit-learn</p>
    <p style='margin: 0;'>Created by <strong>Yessine Zouari</strong> | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)