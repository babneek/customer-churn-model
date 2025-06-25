# 📊 Customer Churn Prediction Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-model-crlrpp9heeq6sp8vlxgieb.streamlit.app/)

**Try it online:**  
[https://customer-churn-model-crlrpp9heeq6sp8vlxgieb.streamlit.app/](https://customer-churn-model-crlrpp9heeq6sp8vlxgieb.streamlit.app/)

---

## 📖 Overview
This project predicts whether a customer is likely to churn (leave) based on behavioral and demographic data. It helps businesses proactively retain customers using machine learning.

---

## 🚀 Features
- Preprocessing and feature engineering included
- Uses Logistic Regression or Random Forest Classifier
- Built-in user interface using Streamlit
- Lightweight and easy to use

---

## 🧠 Tech Stack
- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- joblib

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/babneek/customer-churn-model.git
cd customer-churn-model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run streamlit_app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🧩 How to Use
1. Input customer data in the web app.
2. Click "Predict" to see churn probability.
3. Analyze the results to make retention decisions.

---

## 📁 Folder Structure
```
customer-churn-model/
├── streamlit_app.py
├── models/
│   └── churn_model.pkl
├── data/
│   └── churn_data.csv
├── requirements.txt
```

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙏 Acknowledgements
- [Streamlit](https://streamlit.io/) for the web app framework
- [Python](https://www.python.org/) for the programming language
