# 🍎 AI Fruit Freshness Classifier

A deep learning–based web application that classifies fruits as **Fresh** or **Rotten** using **computer vision and transfer learning**. Built with **TensorFlow** and deployed using **Streamlit**, this project demonstrates how AI can be applied to food quality control and smart retail systems.

---

## 🧠 Project Overview  
This model uses a **Convolutional Neural Network (CNN)** enhanced with **transfer learning (MobileNetV2)** to detect the freshness of fruits (apples, bananas, oranges).  
It classifies each image into two categories:
- **Fresh**
- **Rotten**

The goal is to help reduce food waste and improve automatic quality inspection through deep learning.

---

## ⚙️ Technical Breakdown  
**Model:** MobileNetV2 (pretrained on ImageNet, fine-tuned for fruit dataset)  
**Accuracy:** 98.7% (on validation data)  
**Frameworks:** TensorFlow, Keras  
**Frontend:** Streamlit (interactive web app)  
**Visualization:** Plotly (confidence bar chart)  
**Dataset:** Custom dataset (train/test split with 10K+ images)

---

## 🧩 Key Features  
✅ Real-time fruit freshness detection  
✅ Transfer learning with MobileNetV2  
✅ Interactive confidence visualization (Plotly)  
✅ Streamlit-based modern UI  
✅ Fully deployable web application  

---

## 📊 Results  
| Metric | Training Accuracy | Validation Accuracy | Validation Loss |
|--------|-------------------|---------------------|----------------|
| Value  | 95.6%             | **98.7%**           | 0.0396         |

---

## 🖥️ App Preview  
🌐 **Live Demo:** (https://machine-learning-freshness-prediction.streamlit.app)
📦 **Model File:** `fruit_classifier_model.h5`

---

## 🛠️ Tech Stack  
- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- Matplotlib / Plotly  
- Streamlit  

---

## 🧾 How to Run Locally  
```bash
# Clone repository
git clone https://github.com/Yansoun/machine-learning/fruit-freshness-classifier.git
cd fruit-freshness-classifier

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 💡 Lessons Learned  
- The importance of transfer learning for small datasets  
- Balancing model accuracy with real-time inference performance  
- Optimizing CNN models for deployment in lightweight web environments  

---

## 👤 Author  
**Yessine Zouari**  
Machine Learning Engineer  
🔗 [https://www.linkedin.com/in/yessine-zouari-a84b9a349]
