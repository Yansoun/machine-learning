🎯 Customer Segmentation with K-Means Clustering

An end-to-end machine learning project that segments customers into meaningful groups using unsupervised learning, enabling data-driven marketing strategies and personalization.

🚀 Overview

The goal of this project is to identify distinct customer personas based on demographic and behavioral patterns.
Using K-Means clustering and PCA visualization, we extract 5 customer segments that businesses can use for:

targeted marketing

customer profiling

retention strategies

product recommendations

📊 Dataset

Source: Kaggle — Mall Customers Dataset
Description:

200 customer records

Features include:

Genre (Gender)

Age

Annual Income (k$)

Spending Score (1–100)

Preprocessing Steps:

Removed CustomerID

Encoded gender into numerical values

Scaled numerical features using StandardScaler

Applied PCA for dimensionality reduction (2D visualization)

Used inertia and elbow method to determine optimal clusters

🧩 Methodology
1. K-Means Clustering

Models tested with cluster sizes from 1 to 10.
Using the Elbow Method, the optimal number of clusters found was:

👉 5 segments

Each cluster represents a different customer persona with unique behavior.

2. PCA Visualization

PCA was used to project the 4-dimensional customer features into 2 dimensions for visualization.
This helps clearly show how clusters are separated.

3. Cluster Insights

Each segment reveals a different customer type, for example:

High-income savers

Enthusiastic shoppers

Budget-conscious youth

Mature steady spenders

Affluent premium spenders

These insights are integrated into the Streamlit app.

💻 Tech Stack

Python

Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Streamlit, Joblib

🌐 Streamlit Web App

A full interactive web app was built to:

Input a customer profile

Predict the customer’s cluster

Display segment insights

Show personalized marketing recommendations

Visualize customer features using a radar chart

Live App: (Add link here)

⚙️ How to Run Locally
git clone https://github.com/Yansoun/machine-learning/tree/project8_customer_segmentation.git
cd customer-segmentation
pip install -r requirements.txt
streamlit run app.py

📁 Project Structure
📦 customer-segmentation
│
├── app.py
├── kmeans_model.pk1
├── scaler.pk1
├── README.md
├── requirements.txt
└── data/