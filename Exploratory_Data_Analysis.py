import pandas as pd

def get_summary_statistics(df):
    """Generates numerical and categorical summary statistics."""
    # Numerical Summary
    num_summary = df.describe().T.reset_index()
    num_summary = num_summary.rename(columns={'index': 'Feature'})
    
    # Categorical Summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        cat_summary = pd.DataFrame({
            'Feature': cat_cols,
            'Unique Values': [df[col].nunique() for col in cat_cols],
            'Most Frequent': [df[col].mode()[0] if not df[col].mode().empty else "N/A" for col in cat_cols]
        })
    else:
        cat_summary = pd.DataFrame()
        
    return num_summary, cat_summary

def get_data_quality_report(df):
    """Generates a raw data quality report showing missing values and data types."""
    quality_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    }).reset_index(drop=True)
    return quality_df

def analyze_revenue_trends(df, date_col, revenue_col):
    """Identifies monthly revenue trends and seasonal patterns."""
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[date_col])
    
    # Monthly Trend
    df_copy['Month'] = df_copy[date_col].dt.month_name()
    df_copy['Month_Num'] = df_copy[date_col].dt.month
    monthly_trend = df_copy.groupby(['Month_Num', 'Month'])[revenue_col].sum().reset_index()
    monthly_trend = monthly_trend.sort_values('Month_Num').drop(columns=['Month_Num'])
    
    # Seasonal Trend (Quarterly)
    df_copy['Quarter'] = df_copy[date_col].dt.to_period('Q').astype(str)
    seasonal_trend = df_copy.groupby('Quarter')[revenue_col].sum().reset_index()
    
    return monthly_trend, seasonal_trend

def analyze_product_performance(df, product_col, revenue_col):
    """Calculates total revenue and units sold per product."""
    performance = df.groupby(product_col).agg(
        Total_Revenue=(revenue_col, 'sum'),
        Units_Sold=(product_col, 'count')
    ).reset_index().sort_values('Total_Revenue', ascending=False)
    return performance

def analyze_customer_behavior(df, customer_col, revenue_col):
    """Analyzes customer purchasing frequency and average order value."""
    behavior = df.groupby(customer_col).agg(
        Total_Spent=(revenue_col, 'sum'),
        Purchase_Frequency=(customer_col, 'count')
    ).reset_index()
    behavior['Average_Order_Value'] = (behavior['Total_Spent'] / behavior['Purchase_Frequency']).round(2)
    return behavior.sort_values('Total_Spent', ascending=False)

def analyze_regional_performance(df, region_col, revenue_col):
    """Analyzes sales volume and revenue by region."""
    regional = df.groupby(region_col).agg(
        Total_Revenue=(revenue_col, 'sum'),
        Number_of_Transactions=(region_col, 'count')
    ).reset_index().sort_values('Total_Revenue', ascending=False)
    return regional