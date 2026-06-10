import streamlit as st

def render_sidebar(uploaded_file, df=None, rev_col=None):
    with st.sidebar:
        st.title("🛠️ Control Center")
        
        # 1. Status & Reset
        if uploaded_file is not None:
            st.success("Data Ready")
            if st.button("🔄 Reset Session"):
                st.session_state.clear()
                st.rerun()
        else:
            st.warning("Please upload a file")
        
        st.divider()
            
        # 2. Performance Metrics
        if df is not None and rev_col is not None:
            st.subheader("🚀 Quick Overview")
            st.metric("Total Revenue", f"₹{df[rev_col].sum():,.2f}")
            st.metric("Transactions", df.shape[0])
            
            # Model Status Indicator
            if 'ml_model' in st.session_state:
                st.caption("✅ Model: Trained & Ready")
            else:
                st.caption("⚠️ Model: Needs Training")
            st.divider()
            
        # 3. Navigation
        st.subheader("📍 Navigation")
        options = ["Dashboard Overview", "Sales Analytics", "Product Performance", "ML & AI Studio"]
        page = st.radio("Select View:", options, 
                        index=options.index(st.session_state.get('page', options[0])),
                        disabled=(df is None))
        st.session_state.page = page
        
        # 4. Export Hub (New!)
        st.divider()
        st.subheader("💾 Export Hub")
        if df is not None:
            st.download_button("📥 Export Cleaned Data", df.to_csv(index=False), "data.csv")
            
        return page, None