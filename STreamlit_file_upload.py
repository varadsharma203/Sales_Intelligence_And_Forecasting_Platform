import streamlit as st
import pandas as pd 
import numpy as np
from Data_cleaning import clean_sales_data
from ml_modle import train_sales_forecast_model 
from Data_visualization import (
    plot_revenue_trends, plot_monthly_sales, plot_category_analysis, 
    plot_top_products, plot_regional_sales, plot_purchase_frequency,
    plot_actual_vs_predicted, plot_feature_importance 
)
# Replace your old import line with this:
from GenAI_Insights import get_ai_chat_response, draft_full_executive_report
import plotly.express as px

from Exploratory_Data_Analysis import (
    get_summary_statistics, get_data_quality_report, analyze_revenue_trends,
    analyze_product_performance, analyze_customer_behavior, analyze_regional_performance
)
import forecasting_engine as engine
import sidebar_logic as sb # Import your new sidebar file
# Configure the page layout to span the whole screen for a better dashboard experience
import churn_engine as churn


# 1. Page Configuration (Must be first)
st.set_page_config(layout="wide", page_title="Sales Forecasting Platform")

# 2. Define the hero function
def render_landing_page():
    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, #1E90FF, #00BFFF);
        padding: 50px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .hero-title { font-size: 3rem; font-weight: 800; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.25rem; opacity: 0.9; margin-bottom: 30px; }
    </style>
    <div class="hero-container">
        <div class="hero-title">Unlock Your Sales Potential</div>
        <div class="hero-subtitle">Forecasting, Churn Analysis, and AI-Powered Insights in one unified dashboard.</div>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy", "94.2%", "+2.1%")
    m2.metric("AI Response Time", "< 2s")
    m3.metric("Data Processing", "Real-time")
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🔮 Predictive Forecasting")
        st.write("Leverage advanced ML models to anticipate revenue trends.")
    with c2:
        st.subheader("⚡ Churn Analysis")
        st.write("Identify at-risk customers instantly.")
    with c3:
        st.subheader("🧠 Gemini AI Insights")
        st.write("Get data-driven answers from your custom AI assistant.")
    st.divider()
    st.info("🚀 **Ready to start?** Use the uploader in the sidebar to load your dataset.")
    st.caption("Developed by Varad | [View on GitHub](https://github.com/varadandro-lang/Sales_Intelligence_And_Forecasting_Platform)")

# 3. Define the File Uploader
st.title("📊 SALES INTELLIGENCE DASHBOARD")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# 4. Sidebar Logic
current_page, selected_regions = sb.render_sidebar(
    uploaded_file, 
    st.session_state.get('cleaned_data'), 
    'revenue'
)

# 5. Handle the landing page vs. dashboard logic
if uploaded_file is None:
    render_landing_page()
    st.stop() # Hides the rest of the app until a file is uploaded# Use current_page to switch logic
if current_page == "Dashboard Overview":
    st.write("Welcome to the Dashboard!")
# ... and so on
# 1. Single File Uploader
#uploaded_file = st.file_uploader(
 #   "Choose a CSV or Excel file", 
  #  type=["csv", "xlsx"]
#)

# 2. Process the file if it exists
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name
        
        # Step A: Read the raw file into a DataFrame
        if file_name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
    # Try using the 'calamine' engine
            raw_df = pd.read_excel(uploaded_file, engine='calamine')
        else:
            st.error("Unsupported file type!")
            st.stop()
            
        st.success(f"{file_name} successfully loaded!")
        
        # Display the RAW data
        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head(5)) 
        
        
        # ==========================================
        # 1. THE CLEANING PIPELINE
        # ==========================================
        if st.button("🧼 Clean Data"):
            cleaned_df, quality_report = clean_sales_data(raw_df)
            st.session_state['cleaned_data'] = cleaned_df
            st.session_state['cleaning_report'] = quality_report
            
        if 'cleaned_data' in st.session_state and 'cleaning_report' in st.session_state:
            final_df = st.session_state['cleaned_data']
            report = st.session_state['cleaning_report']
            
            st.success("Data successfully cleaned and formatted!")
            
            # Display Data Quality Report
            st.subheader("📊 Data Quality Report")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Starting Rows", report["initial_rows"])
            col_b.metric("Duplicates Removed", report["duplicates_removed"])
            col_c.metric("Missing Values Fixed", report["missing_values_filled"])
            col_d.metric("Outliers Capped", report["outliers_capped"])
                
            # Display the cleaned data
            st.subheader("Cleaned Data Preview")
            st.dataframe(final_df)

            st.download_button(
                label="📥 Download Cleaned Data as CSV",
                data=final_df.to_csv(index=False).encode('utf-8'),
                file_name=f"cleaned_{file_name}",
                mime="text/csv"
            )
            # ==========================================
            # EXPLORATORY DATA ANALYSIS (EDA) STUDIO
            # ==========================================
            st.divider()
            st.header("🧮 Exploratory Data Analysis (EDA)")
            st.write("Deep dive into the raw mathematical summaries and tabular reports of your dataset.")
            
            # Re-grab columns for mapping
            all_cols = final_df.columns.tolist()
            numeric_cols = final_df.select_dtypes(include=['number']).columns.tolist()
            text_cols = final_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            with st.expander("⚙️ EDA Configuration Mapping", expanded=False):
                st.write("Map your dataset columns to generate the EDA tables.")
                e1, e2, e3 = st.columns(3)
                eda_rev = e1.selectbox("EDA Revenue Metric", numeric_cols, key="eda_rev")
                eda_date = e2.selectbox("EDA Date Metric", all_cols, index=all_cols.index('date') if 'date' in all_cols else 0, key="eda_date")
                eda_prod = e3.selectbox("EDA Product Metric", text_cols if text_cols else all_cols, key="eda_prod")
                
                e4, e5 = st.columns(2)
                eda_reg = e4.selectbox("EDA Region Metric", text_cols if text_cols else all_cols, key="eda_reg")
                eda_cust = e5.selectbox("EDA Customer Metric", all_cols, key="eda_cust")

            if st.button("Generate EDA Reports"):
                eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Summary & Quality", "Trends & Regions", "Products & Customers"])
                
                with eda_tab1:
                    st.subheader("Data Quality Report")
                    st.dataframe(get_data_quality_report(final_df), use_container_width=True)
                    
                    st.subheader("Numeric Summary Statistics")
                    num_stat, cat_stat = get_summary_statistics(final_df)
                    st.dataframe(num_stat, use_container_width=True)
                    
                    if not cat_stat.empty:
                        st.subheader("Categorical Summary Statistics")
                        st.dataframe(cat_stat, use_container_width=True)
                        
                with eda_tab2:
                    st.subheader("Revenue Trend Analysis")
                    try:
                        monthly, seasonal = analyze_revenue_trends(final_df, eda_date, eda_rev)
                        col_m, col_s = st.columns(2)
                        col_m.write("**Monthly Performance**")
                        col_m.dataframe(monthly, use_container_width=True)
                        col_s.write("**Quarterly/Seasonal Performance**")
                        col_s.dataframe(seasonal, use_container_width=True)
                    except Exception as e:
                        st.warning("Could not generate temporal trends. Ensure the Date column is valid.")
                        
                    st.subheader("Regional Sales Performance")
                    st.dataframe(analyze_regional_performance(final_df, eda_reg, eda_rev), use_container_width=True)
                    
                with eda_tab3:
                    st.subheader("Product Performance Analysis")
                    st.dataframe(analyze_product_performance(final_df, eda_prod, eda_rev), use_container_width=True)
                    
                    st.subheader("Customer Purchasing Behavior")
                    st.dataframe(analyze_customer_behavior(final_df, eda_cust, eda_rev), use_container_width=True)

            # ==========================================
            # 2. ADVANCED VISUALIZATION ENGINE
            # ==========================================
            st.divider()
            st.header("📈 Advanced Visualization Engine")
            st.write("Configure your dashboard by matching your data columns to the chart requirements.")
            
            all_cols = final_df.columns.tolist()
            numeric_cols = final_df.select_dtypes(include=['number']).columns.tolist()
            text_cols = final_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            with st.expander("⚙️ Dashboard Configuration Mapping", expanded=True):
                c1, c2, c3 = st.columns(3)
                rev_col = c1.selectbox("Revenue/Sales Metric", numeric_cols)
                date_col = c2.selectbox("Date/Time Metric", all_cols, index=all_cols.index('date') if 'date' in all_cols else 0)
                prod_col = c3.selectbox("Product Name/ID", text_cols if text_cols else all_cols)
                
                c4, c5, c6 = st.columns(3)
                cat_col = c4.selectbox("Product Category", text_cols if text_cols else all_cols)
                reg_col = c5.selectbox("Region/Location", text_cols if text_cols else all_cols)
                cust_col = c6.selectbox("Customer Name/ID", all_cols)

            if st.button("📊 Generate Dashboard"):
                st.success("Dashboard generated successfully!")
                
                # --- ROW 1: Trends ---
                st.subheader("Performance Over Time")
                row1_col1, row1_col2 = st.columns(2)
                try:
                    row1_col1.plotly_chart(plot_revenue_trends(final_df, date_col, rev_col), use_container_width=True)
                    row1_col2.plotly_chart(plot_monthly_sales(final_df, date_col, rev_col), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate time series charts. Error: {e}")

                # --- ROW 2: Product Insights ---
                st.subheader("Product & Category Insights")
                row2_col1, row2_col2 = st.columns(2)
                try:
                    row2_col1.plotly_chart(plot_category_analysis(final_df, cat_col, rev_col), use_container_width=True)
                    row2_col2.plotly_chart(plot_top_products(final_df, prod_col, rev_col), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate product charts. Error: {e}")

                # --- ROW 3: Demographics & Frequency ---
                st.subheader("Regional & Customer Metrics")
                row3_col1, row3_col2 = st.columns(2)
                try:
                    row3_col1.plotly_chart(plot_regional_sales(final_df, reg_col, rev_col), use_container_width=True)
                    row3_col2.plotly_chart(plot_purchase_frequency(final_df, cust_col), use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate demographic charts. Error: {e}")

            # ==========================================
            # 3. MACHINE LEARNING STUDIO
            # ==========================================
            # ==========================================
            # 3. MACHINE LEARNING STUDIO
            # ==========================================
            st.divider()
            st.header("🤖 Machine Learning Studio")
            st.subheader("Data Requirements")

            st.markdown("""
| Column Category | Fields |
| :--- | :--- |
| **Target Variable** | Sales / Revenue |
| **Numeric Features** | Price, Quality |
| **Categorical Features** | Category, Region |
|**Date/Time Column** | Date, Month, Year |
""")
            
            ml_col1, ml_col2 = st.columns(2)
            
            with ml_col1:
                target_col = st.selectbox("Select Target Variable (e.g., Sales)", all_cols)
                model_choice = st.selectbox("Select Algorithm", ["Random Forest", "Linear Regression"])
                model_type_str = "random_forest" if model_choice == "Random Forest" else "linear"

            with ml_col2:
                time_col = st.selectbox("Select ML Date/Time Column", all_cols, index=all_cols.index('date') if 'date' in all_cols else 0)
                available_features = [col for col in all_cols if col != target_col and col != time_col]
                numeric_feat_cols = final_df[available_features].select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = st.multiselect("Select Feature Columns (Must be numbers)", numeric_feat_cols, default=numeric_feat_cols)

            if st.button("🚀 Train Forecasting Model"):
                if not feature_cols:
                    st.warning("Please select at least one feature column.")
                else:
                    with st.spinner("Training..."):
                        trained_model, metrics, results_df = train_sales_forecast_model(
                            df=final_df, target_column=target_col, features=feature_cols, 
                            model_type=model_type_str, time_column=time_col
                        )
                        st.session_state.update({'ml_metrics': metrics, 'ml_results_df': results_df, 
                                                'ml_model': trained_model, 'ml_target_col': target_col, 
                                                'ml_feature_cols': feature_cols, 'ml_model_type_str': model_type_str})
                        st.success("Model trained successfully!")

            # ==========================================
            # 3b. FORECASTING ENGINE (Moved OUTSIDE the training button)
            # ==========================================
            st.divider()
            st.header("🔮 Sales Forecasting Engine")
            months_to_forecast = st.slider("Forecast Horizon (Months)", 1, 12, 6)

            if st.button("🚀 Generate Future Forecast"):
                if 'ml_model' in st.session_state:
        # Debugging: These lines will now show you the mismatch BEFORE the crash happens
                    #st.write("--- DEBUG INFO ---")
                    #st.write("Columns in current dataframe:", final_df.columns.tolist())
                    #st.write("Features model expects:", st.session_state['ml_feature_cols'])
        
        # Use the SAME feature list used during training
                    features_used = st.session_state['ml_feature_cols']
        
                    try:
                        forecast = engine.generate_forecast(
                        st.session_state['ml_model'], 
                        final_df, 
                        months_to_forecast,
                        features_used
                        )
                        st.session_state['forecast_result'] = forecast
                    except Exception as e:
                        st.error(f"Forecast Engine Crashed: {e}")
                if st.session_state.get('forecast_result'):
                    st.write("First 5 forecast predictions:", st.session_state['forecast_result'][:5])
            if st.session_state.get('forecast_result') is not None:
                forecast_df = pd.DataFrame(st.session_state['forecast_result'], columns=['Forecasted Revenue'])
                forecast_df['Month'] = range(1, len(forecast_df) + 1)
    
                # 2. Create Plotly chart with a zoomed Y-axis
                fig = px.line(forecast_df, x='Month', y='Forecasted Revenue', markers=True)
    
                # 3. CRITICAL: Set the Y-axis range to "zoom in" on your data
                # This forces the chart to start at your minimum value, not 0
                min_val = forecast_df['Forecasted Revenue'].min()
                max_val = forecast_df['Forecasted Revenue'].max()
                fig.update_yaxes(range=[min_val * 0.99, max_val * 1.01])
    
                st.plotly_chart(fig, use_container_width=True)
    
                if st.button("Clear Forecast"):
                    st.session_state['forecast_result'] = None
                    st.rerun()
            
            
            # ==========================================
            # CHECK MEMORY BEFORE DISPLAYING ML RESULTS & AI
            # ==========================================
            if 'ml_metrics' in st.session_state:
                # Retrieve from memory
                metrics = st.session_state['ml_metrics']
                results_df = st.session_state['ml_results_df']
                trained_model = st.session_state['ml_model']
                target_col = st.session_state['ml_target_col']
                feature_cols = st.session_state['ml_feature_cols']
                model_type_str = st.session_state['ml_model_type_str']

                # Display the metrics
                st.subheader("📈 Model Performance Metrics")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("MAE", metrics["MAE"])
                m_col2.metric("RMSE", metrics["RMSE"])
                m_col3.metric("R² Score", f"{metrics['R2 Score'] * 100:.1f}%")
                
                # Display the ML Visualizations
                st.divider()
                st.subheader("👁️ AI Analysis Visualizations")
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    fig_acc = plot_actual_vs_predicted(results_df)
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                with v_col2:
                    fig_feat = plot_feature_importance(trained_model, feature_cols, model_type_str)
                    st.plotly_chart(fig_feat, use_container_width=True)
                    
                    
            
                
                # ==========================================
                # 4. GEN AI BUSINESS ANALYST
                # ==========================================
               # ==========================================
                # 4. GEN AI BUSINESS ANALYST (INTERACTIVE)
                # ==========================================
                st.divider()
                st.header("🧠 AI Data Assistant")
                
                # Initialize chat history in session state
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                # Display existing chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input box
                if prompt := st.chat_input("Ask about your data or ML results..."):
                    # Display user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate AI response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response, st.session_state.chat_history = get_ai_chat_response(
                                final_df, metrics, target_col, prompt, st.session_state.chat_history
                            )
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                st.divider()
                st.subheader("📄 Quick Executive Summary")
                if st.button("Generate Summary"):
                    with st.spinner("Generating executive summary..."):
                        report = draft_full_executive_report(final_df, metrics, target_col)
                        st.session_state['report'] = report
                    if 'report' in st.session_state:
                        st.info(st.session_state['report'])
                        st.download_button("📥 Download", st.session_state['report'], "Summary.txt")                       
            # ==========================================
            # ⚡ NEW: CHURN PREDICTION MODULE
            # ==========================================
            # ==========================================
            st.header("⚡ Churn Prediction Module")
            st.markdown("---")

            with st.expander("🛠️ Configure & Run Churn Analysis"):
                c1, c2 = st.columns(2)
                churn_target = c1.selectbox("Select Target Column (e.g., Churned)", final_df.columns, key="churn_target")
                churn_features = c2.multiselect("Select Feature Columns", final_df.columns, default=final_df.select_dtypes(include=[np.number]).columns.tolist(), key="churn_features")

                if st.button("🚀 Train Churn Model"):
                    with st.spinner("Training Random Forest Classifier..."):
                        churn_model = churn.train_churn_model(final_df, churn_features, churn_target)
                        st.session_state['churn_model'] = churn_model
                        st.success("Churn model trained successfully!")


            if 'churn_model' in st.session_state:
                st.write("### Prediction Results")
                predictions = churn.predict_churn(st.session_state['churn_model'], final_df[churn_features])
                final_df['Churn_Prediction'] = predictions
                st.dataframe(final_df.head(10)) # Show the first 10 predictions
    except Exception as e:
        # A catch-all for any major file reading errors
        st.error(f"A general error occurred: {e}")

        
else:
    st.info("Please upload a file to begin.")
def render_footer():
    st.divider()
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 5px;
        font-size: 10px;
        border-top: 1px solid #333;
    }
    </style>
    <div class="footer">
        <p>Sales Intelligence & Forecasting Platform </p>
        <p>Powered by  Google Gemini AI </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Call this at the very end of your script
render_footer()
