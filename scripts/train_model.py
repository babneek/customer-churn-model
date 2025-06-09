import os

def train():
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # rest of your code ...


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from data_preprocessing import load_and_clean_data, encode_features

def train():
    # Load and preprocess data
    df = load_and_clean_data()
    df, le_dict = encode_features(df)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split into training and testing sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model and label encoders
    dump(model, 'models/churn_model.pkl')
    dump(le_dict, 'models/label_encoder.pkl')

    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train()
    print("Model trained and saved successfully!")
