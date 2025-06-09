import streamlit as st
import joblib
import pandas as pd

# Load model and label encoders
model = joblib.load('models/churn_model.pkl')
le_dict = joblib.load('models/label_encoder.pkl')

st.title("Customer Churn Prediction")

# User inputs
def user_input_features():
    gender = st.selectbox('Gender', ['Female', 'Male'])
    SeniorCitizen = st.selectbox('Senior Citizen', [0, 1])
    Partner = st.selectbox('Partner', ['Yes', 'No'])
    Dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=5)
    PhoneService = st.selectbox('Phone Service', ['Yes', 'No'])
    MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
    PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=350.0)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    return data

input_data = user_input_features()

# Preprocess input (encode categorical features)
for col in input_data:
    if col in le_dict:
        try:
            input_data[col] = le_dict[col].transform([input_data[col]])[0]
        except ValueError:
            st.error(f"Value '{input_data[col]}' not recognized for feature '{col}'. Please choose a valid option.")
            st.stop()

df_input = pd.DataFrame([input_data])

# Predict button
if st.button('Predict Churn'):
    prediction = model.predict(df_input)[0]
    if prediction == 1:
        st.error("Customer will likely churn.")
    else:
        st.success("Customer is unlikely to churn.")
