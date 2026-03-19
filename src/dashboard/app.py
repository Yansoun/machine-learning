import streamlit as st
import requests
import plotly.graph_objects as go

st.title("Customer Churn Prediction")
st.write("Enter customer information to predict churn risk.")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
senior = st.selectbox("Senior Citizen", [0, 1])

if st.button("Predict Churn"):

    data = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data
    )

    result = response.json()

    prob = result["churn_probability"]
    if prob < 0.4:
        st.success("Low churn risk ✅")

    elif prob < 0.7:
        st.warning("Medium churn risk ⚠️")

    else:
        st.error("High churn risk 🚨")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
        }
    ))

    st.plotly_chart(fig)