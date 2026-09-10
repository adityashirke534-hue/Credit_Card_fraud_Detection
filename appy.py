import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Page Setup
st.set_page_config(
    page_title="FraudGuard AI | Credit Card Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .fraud-alert {
        background-color: #fff2f2;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #dc3545;
    }
    .safe-alert {
        background-color: #f2fff4;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #198754;
    }
    </style>
""", unsafe_allow_html=True)

# Model & Scaler Loader
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('isolation_forrest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

iso_model, scaler = load_assets()

# Sidebar - System Info & Batch Processing
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=70)
    st.title("FraudGuard AI")
    st.caption("Engineered with Isolation Forest & PCA")
    
    st.divider()
    st.subheader("⚙️ System Status")
    if iso_model is not None and scaler is not None:
        st.success("Model Status: **Loaded & Active**")
    else:
        st.error("Model Status: **Files Not Found**")
        st.info("Ensure `isolation_forest_model.pkl` and `scaler.pkl` are in the app root folder.")
    
    st.divider()
    st.subheader("📊 Batch Testing")
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    
    st.caption("Developed for Credit Card Anomaly Analytics")

# Main Header
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("Real-time anomaly detection system powered by Machine Learning.")

if iso_model is None or scaler is None:
    st.warning("⚠️ Please resolve the model file missing error to proceed.")
    st.stop()

# Tabs Interface
tab1, tab2 = st.tabs(["🔍 Single Transaction Predictor", "📈 Batch CSV Analysis"])

# TAB 1: Single Prediction
with tab1:
    st.subheader("Transaction Parameter Inputs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📍 Distance Factors")
        dist_home = st.number_input("Distance From Home (km)", min_value=0.0, value=10.5, step=1.0)
        dist_last_trans = st.number_input("Distance From Last Transaction (km)", min_value=0.0, value=1.2, step=0.1)

    with col2:
        st.markdown("##### 💵 Financial Ratios")
        ratio_median = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=1.8, step=0.1)
        repeat_retailer = st.radio("Repeat Retailer?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    with col3:
        st.markdown("##### 💳 Verification Signals")
        used_chip = st.radio("Used Chip?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        used_pin = st.radio("Used PIN?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        online_order = st.radio("Online Order?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    st.divider()
    
    if st.button("🚀 Evaluate Risk Level", use_container_width=True):
        # Format input vector
        raw_features = np.array([[
            dist_home,
            dist_last_trans,
            ratio_median,
            repeat_retailer,
            used_chip,
            used_pin,
            online_order
        ]])
        
        # Preprocess and inference
        scaled_features = scaler.transform(raw_features)
        prediction = iso_model.predict(scaled_features)[0]
        
        # Output Handling (-1 = Anomaly/Fraud, 1 = Normal)
        if prediction == -1:
            st.markdown(f"""
                <div class="fraud-alert">
                    <h3 style="color: #dc3545; margin:0;">🚨 High Risk: Fraudulent Transaction Detected</h3>
                    <p style="margin-top:10px; color: #555;">The pattern of this transaction significantly deviates from standard customer purchase behaviors.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="safe-alert">
                    <h3 style="color: #198754; margin:0;">✅ Low Risk: Normal Transaction Verified</h3>
                    <p style="margin-top:10px; color: #555;">Transaction parameters match legitimate purchasing metrics.</p>
                </div>
            """, unsafe_allow_html=True)

# TAB 2: Batch Analysis
with tab2:
    st.subheader("Batch CSV Processing")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        required_cols = [
            'distance_from_home', 'distance_from_last_transaction', 
            'ratio_to_median_purchase_price', 'repeat_retailer', 
            'used_chip', 'used_pin_number', 'online_order'
        ]
        
        if all(col in batch_df.columns for col in required_cols):
            X_batch = batch_df[required_cols]
            X_batch_scaled = scaler.transform(X_batch)
            preds = iso_model.predict(X_batch_scaled)
            
            # Map predictions
            batch_df['Prediction'] = np.where(preds == -1, 'Fraud', 'Normal')
            
            # Key Summary Metrics
            total_trans = len(batch_df)
            fraud_count = (batch_df['Prediction'] == 'Fraud').sum()
            normal_count = total_trans - fraud_count
            fraud_pct = (fraud_count / total_trans) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Transactions", f"{total_trans:,}")
            m2.metric("Normal Transactions", f"{normal_count:,}")
            m3.metric("Flagged Fraud", f"{fraud_count:,}", delta=f"{fraud_pct:.2f}% risk", delta_color="inverse")
            m4.metric("Fraud Rate", f"{fraud_pct:.1f}%")
            
            st.divider()
            
            # Visualization
            fig = px.pie(
                batch_df, 
                names='Prediction', 
                title='Transaction Breakdown',
                color='Prediction',
                color_discrete_map={'Normal': '#198754', 'Fraud': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### Detailed Prediction Results")
            st.dataframe(batch_df, use_container_width=True)
        else:
            st.error(f"Missing required columns. Please ensure your CSV includes: {', '.join(required_cols)}")
    else:
        st.info("💡 Upload a CSV file in the left sidebar to run batch processing.")