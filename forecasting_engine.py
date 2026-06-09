def generate_forecast(model, last_known_data, months_to_predict, feature_cols):
    """
    Predicts future sales. 
    Added 'feature_cols' to ensure we only pass what the model learned!
    """
    forecast = []
    # ONLY select the columns the model was trained on
    current_input = last_known_data[feature_cols].copy()

    for _ in range(months_to_predict):
        pred = model.predict(current_input)
        forecast.append(float(pred[0]))
        
        # Update ONLY the lag feature
        if 'sales_lag_1' in current_input.columns:
            current_input['sales_lag_1'] = pred[0]
            
    return forecast