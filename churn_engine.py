# churn_engine.py
from sklearn.ensemble import RandomForestClassifier

def train_churn_model(df, feature_cols, target_col):
    """Trains a classifier to predict customer churn."""
    model = RandomForestClassifier(n_estimators=100)
    X = df[feature_cols]
    y = df[target_col]
    model.fit(X, y)
    return model

def predict_churn(model, customer_data):
    """Predicts churn status for new customer data."""
    return model.predict(customer_data)