import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_sales_forecast_model(df, target_column, features, model_type='random_forest', time_column='date'):
    # 1. Create a deep copy to keep original data safe
    df = df.copy()
    
    # 2. Add time_index explicitly
    df['time_index'] = range(len(df))
    if 'time_index' not in features:
        features.append('time_index')
    
    # 3. Define the subset of columns we are working with
    final_cols = features + [target_column]
    data = df[final_cols].copy()
    
    # 4. CRITICAL: Force numeric and drop rows with ANY bad data
    # This turns any non-numeric text into NaN, then drops those rows
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 5. Check if training data exists
    if data.shape[0] < 2:
        raise ValueError(f"Not enough valid data (Found {data.shape[0]} rows). Check for non-numeric data in columns: {features}")
    
    X = data[features]
    y = data[target_column]
    
    # 6. Train the model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
        
    model.fit(X, y)
    
    # 7. Calculate metrics based on the cleaned 'data'
    predictions = model.predict(X)
    metrics = {
        "MAE": round(mean_absolute_error(y, predictions), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y, predictions)), 2),
        "R2 Score": round(r2_score(y, predictions), 4)
    }
    results_df = pd.DataFrame({'Actual': y, 'Predicted': predictions})
    return model, metrics, results_df