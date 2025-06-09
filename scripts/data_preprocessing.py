# scripts/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    return df

def encode_features(df):
    le_dict = {}
    # Encode all object type columns except target 'Churn'
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    # Encode target column 'Churn' as binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df, le_dict

# Example usage:
if __name__ == "__main__":
    df = load_and_clean_data()
    df, le_dict = encode_features(df)
    print(df.head())
