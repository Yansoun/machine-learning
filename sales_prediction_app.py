import streamlit as st
import pandas as pd
import joblib

# 🎨 Page setup
st.set_page_config(page_title="Sales Prediction App", page_icon="📈", layout="centered")

# Load the saved model and scaler
model = joblib.load("Sales_model.pk1")
scaler = joblib.load("scaler.pk1")

# 🌟 App title and intro
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>📊 Sales Prediction App</h1>
    <p style='text-align: center;'>Predict sales based on TV, Radio, and Newspaper advertising spend!</p>
    <hr style='border:1px solid #ccc;'>
""", unsafe_allow_html=True)

# 📋 Sidebar info
st.sidebar.header("👤 About the App")
st.sidebar.write("""
This app uses a trained **Machine Learning model** to predict sales based on advertising budgets.  
Created by Yessine Zouari.
""")

# 🧮 Input section
st.subheader("Enter Advertising Budgets ($):")
tv = st.number_input("TV Advertising ($)", min_value=0.0, value=100.0, step=10.0)
radio = st.number_input("Radio Advertising ($)", min_value=0.0, value=50.0, step=10.0)
newspaper = st.number_input("Newspaper Advertising ($)", min_value=0.0, value=20.0, step=10.0)

# 🧠 Prediction button
if st.button("🚀 Predict Sales"):
    # Prepare input
    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "radio", "newspaper"])
    try:
        # Scale the data and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"💰 Predicted Sales: **{prediction[0]:.2f} units**")
    except ValueError as e:
        st.error(f"Error: {e}")

# ✨ Footer
st.markdown("""
    <hr>
    <p style='text-align:center; font-size:13px; color:gray;'>
    Built with ❤️ using Streamlit & Scikit-learn
    </p>
""", unsafe_allow_html=True)
