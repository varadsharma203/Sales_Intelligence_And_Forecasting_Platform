import pandas as pd
import numpy as np

def generate_forecast(model, last_known_data, months_to_predict, feature_cols):
    """
    Predicts future sales with shape validation and trend handling.
    """
    # 1. Prepare data
    input_df = last_known_data.copy()
    
    # 2. Add 'time_index' if missing to allow for trend prediction
    if 'time_index' in feature_cols and 'time_index' not in input_df.columns:
        input_df['time_index'] = range(len(input_df))
    
    # 3. Clean the input (prevents NaN/Inf crashes)
    try:
        current_input = input_df[feature_cols].copy().iloc[[-1]]
        current_input = current_input.apply(pd.to_numeric, errors='coerce').fillna(0)
    except KeyError as e:
        raise KeyError(f"Feature column missing: {e}. Ensure the data has the same features used during training.")

    forecast = []
    
    # 4. Generate forecast loop
    for _ in range(months_to_predict):
        pred = model.predict(current_input)
        forecast.append(float(pred[0]))
        
        # Update features for next iteration
        if 'sales_lag_1' in current_input.columns:
            current_input['sales_lag_1'] = pred[0]
            
        if 'time_index' in current_input.columns:
            current_input['time_index'] += 1
            
        if 'month' in current_input.columns:
            current_input['month'] = (current_input['month'] % 12) + 1
            
    return forecast