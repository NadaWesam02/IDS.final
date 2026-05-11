# app.py - Intrusion Detection System Web App
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="IDS - Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Network Intrusion Detection System")
st.markdown("### Machine Learning based IDS to detect cyber attacks")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('ids_model_final.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')
    return model, scaler, encoders

try:
    model, scaler, encoders = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.info("Please make sure all model files are in the same directory")
    st.stop()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📡 Network Connection Features")
    st.markdown("Enter the network connection details below:")
    
    # Input fields (main features)
    duration = st.number_input("Duration (seconds)", min_value=0, value=0, step=1)
    
    protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
    service = st.selectbox("Service", ["http", "private", "smtp", "ftp", "telnet", "other"])
    flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR"])
    
    src_bytes = st.number_input("Source Bytes", min_value=0, value=0, step=100)
    dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0, step=100)
    
    with st.expander("Advanced Features (Optional)"):
        st.info("Leave default values if not sure")
        land = st.number_input("Land", value=0)
        wrong_fragment = st.number_input("Wrong Fragment", value=0)
        urgent = st.number_input("Urgent", value=0)

with col2:
    st.subheader("🔍 Detection Result")
    st.markdown("Click the button below to analyze the connection")
    
    # Predict button
    if st.button("🚨 DETECT INTRUSION", type="primary", use_container_width=True):
        # Prepare features
        try:
            # Encode categorical features
            protocol_encoded = encoders['protocol_type'].transform([protocol_type])[0]
            service_encoded = encoders['service'].transform([service])[0]
            flag_encoded = encoders['flag'].transform([flag])[0]
            
            # Create feature array (41 features total)
            features = np.array([
                duration, protocol_encoded, service_encoded, flag_encoded,
                src_bytes, dst_bytes, land, wrong_fragment, urgent,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]).reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # Show result
            st.markdown("---")
            
            if prediction == 1:
                st.error("🚨 ALERT: INTRUSION DETECTED!")
                st.metric("Confidence", f"{probability[1]*100:.2f}%")
                st.warning("⚠️ This connection appears to be malicious!")
                
                # Add visual indicator
                st.progress(int(probability[1]*100))
            else:
                st.success("✅ NORMAL TRAFFIC")
                st.metric("Confidence", f"{probability[0]*100:.2f}%")
                st.info("This connection appears to be normal.")
                
            # Show additional details
            with st.expander("📊 Detailed Analysis"):
                st.write(f"**Prediction:** {'Attack' if prediction==1 else 'Normal'}")
                st.write(f"**Confidence:** {max(probability)*100:.2f}%")
                st.write(f"**Attack Probability:** {probability[1]*100:.2f}%")
                st.write(f"**Normal Probability:** {probability[0]*100:.2f}%")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Make sure all fields are filled correctly")

# Footer
st.markdown("---")
st.markdown("### 📌 How it works")
st.markdown("""
This IDS uses a **Random Forest** model trained on the NSL-KDD dataset.
- **Normal (✅)**: Regular network traffic
- **Attack (🚨)**: Malicious activity (DoS, Probe, R2L, U2R)

The model achieves **98.67% attack detection rate (Recall)**.
""")

# Sidebar info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.markdown("## About")
    st.markdown("""
    **Intrusion Detection System**
    
    - Model: Random Forest
    - Accuracy: 98.1%
    - Recall: 98.67%
    - Dataset: NSL-KDD
    
    ---
    **Features used:**
    - 41 network features
    - Protocol, Service, Flag
    - Bytes, Duration, etc.
    """)