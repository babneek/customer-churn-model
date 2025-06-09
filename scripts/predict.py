import pandas as pd
from joblib import load

def predict_single(customer_data):
    model = load('models/churn_model.pkl')
    le_dict = load('models/label_encoder.pkl')

    for col in customer_data:
        if col in le_dict:
            customer_data[col] = le_dict[col].transform([customer_data[col]])[0]
    df = pd.DataFrame([customer_data])
    return model.predict(df)[0]
