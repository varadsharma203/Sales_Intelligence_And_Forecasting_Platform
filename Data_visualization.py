import plotly.express as px
import pandas as pd

def plot_revenue_trends(df, date_col, revenue_col):
    """Line chart showing revenue over the exact timeline."""
    timeline_df = df.groupby(date_col)[revenue_col].sum().reset_index()
    fig = px.line(timeline_df, x=date_col, y=revenue_col, 
                  title="Revenue Trends", markers=True, template="plotly_dark")
    return fig

def plot_monthly_sales(df, date_col, revenue_col):
    """Bar chart aggregating sales by month."""
    df_copy = df.copy()
    # Safely extract just the Year-Month
    df_copy['Month'] = pd.to_datetime(df_copy[date_col], errors='coerce').dt.to_period('M').astype(str)
    monthly_df = df_copy.groupby('Month')[revenue_col].sum().reset_index()
    fig = px.bar(monthly_df, x='Month', y=revenue_col, 
                 title="Monthly Sales Performance", template="plotly_dark", color_discrete_sequence=['#00CC96'])
    return fig

def plot_category_analysis(df, category_col, revenue_col):
    """Donut chart showing revenue breakdown by category."""
    cat_df = df.groupby(category_col)[revenue_col].sum().reset_index()
    fig = px.pie(cat_df, names=category_col, values=revenue_col, 
                 title="Product Category Analysis", hole=0.4, template="plotly_dark")
    return fig

def plot_top_products(df, product_col, revenue_col, top_n=10):
    """Horizontal bar chart of the best-selling products."""
    prod_df = df.groupby(product_col)[revenue_col].sum().reset_index()
    prod_df = prod_df.sort_values(by=revenue_col, ascending=True).tail(top_n) # Get top N and sort for Plotly
    fig = px.bar(prod_df, x=revenue_col, y=product_col, orientation='h', 
                 title=f"Top {top_n} Selling Products", template="plotly_dark", color_discrete_sequence=['#EF553B'])
    return fig

def plot_regional_sales(df, region_col, revenue_col):
    """Bar chart comparing sales across different regions."""
    reg_df = df.groupby(region_col)[revenue_col].sum().reset_index()
    # Sort descending so the biggest region is on the left
    reg_df = reg_df.sort_values(by=revenue_col, ascending=False)
    fig = px.bar(reg_df, x=region_col, y=revenue_col, 
                 title="Regional Sales Comparison", template="plotly_dark", color=region_col)
    return fig

def plot_purchase_frequency(df, customer_col):
    """Histogram showing how often customers make a purchase."""
    # Count how many times each customer ID appears
    freq_df = df[customer_col].value_counts().reset_index()
    freq_df.columns = [customer_col, 'Purchase Count']
    fig = px.histogram(freq_df, x='Purchase Count', 
                       title="Customer Purchase Frequency (Transactions per Customer)", 
                       template="plotly_dark", nbins=20)
    return fig

def plot_actual_vs_predicted(results_df):
    """Scatter plot comparing actual sales to model predictions."""
    # Force alignment and drop any incomplete rows
    clean_df = results_df.dropna(subset=['Actual', 'Predicted']).reset_index(drop=True)
    
    # Create the plot using the cleaned, perfectly aligned data
    fig = px.scatter(
        clean_df, x='Actual', y='Predicted', 
        title="Accuracy: Actual vs. Predicted",
        template="plotly_dark",
        opacity=0.7,
        color_discrete_sequence=['#00CC96']
    )
    
    # Add the perfect prediction line
    min_val = min(clean_df['Actual'].min(), clean_df['Predicted'].min())
    max_val = max(clean_df['Actual'].max(), clean_df['Predicted'].max())
    
    fig.add_shape(
        type="line", line=dict(dash='dash', color='white', width=2),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    return fig
def plot_feature_importance(model, feature_names, model_type):
    """Bar chart showing which features influenced the model most."""
    import numpy as np # Ensure numpy is available
    
    # Extract importance based on the algorithm type
    if model_type == 'random_forest':
        importances = model.feature_importances_
    else: # Linear Regression uses coefficients
        importances = np.abs(model.coef_)
        
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Level': importances
    }).sort_values(by='Importance Level', ascending=True)
    
    fig = px.bar(
        feat_df, x='Importance Level', y='Feature', orientation='h',
        title="AI Clues: Feature Importance",
        template="plotly_dark",
        color_discrete_sequence=['#EF553B']
    )
    return fig