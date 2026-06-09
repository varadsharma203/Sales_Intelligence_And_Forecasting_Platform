import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_sales_forecast_model(df, target_column, features, model_type='random_forest', time_column='date'):
    model_df = df.copy()
    
    # 1. Feature Engineering (as before)
    # ... (Keep your existing feature engineering code here) ...
    
    # 2. DEFINITIVE SYNC: Drop ALL rows with ANY missing data in inputs or target
    final_cols = features + [target_column]
    model_df = model_df.dropna(subset=final_cols)
    
    X = model_df[features]
    y = model_df[target_column]
    
    # 3. Chronological Split
    split_index = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # 4. Train
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model.fit(X_train, y_train)
    
    # 5. PREDICTION SYNC: Force the prediction output to align with the X_test index
    predictions = model.predict(X_test)
    
    # 6. RETURN SYNC: Use y_test's index to create a series that is guaranteed to match
    results_df = pd.DataFrame({
        'Actual': y_test.values, # Use .values to strip the index and force alignment
        'Predicted': predictions
    })
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2 Score": round(r2, 4)}
    
    return model, metrics, results_df