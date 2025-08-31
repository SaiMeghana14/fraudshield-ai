import streamlit as st
from streamlit_option_menu import option_menu
from modules import fraud_detection, scam_detector, visualizations
import pandas as pd
import time

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

# Lottie animation loader
from streamlit_lottie import st_lottie
import json
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

fraud_anim = load_lottie("assets/animations/fraud.json")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "FraudShield AI",
        ["🏠 Home", "📊 Trading Fraud Detection", "📱 Investor FraudShield", "📈 Reports"],
        icons=["house", "graph-up", "shield-check", "file-earmark-text"],
        menu_icon="cast",
        default_index=0
    )

# Home Page
if selected == "🏠 Home":
    st.title("🛡️ FraudShield AI")
    st_lottie(fraud_anim, height=300, key="fraud")
    st.markdown("### Protecting Every Trade, Securing Every Investor.")
    st.info("An AI-powered platform for fraud detection & investor protection aligned with SEBI’s mandate.")

# Trading Fraud Detection
elif selected == "📊 Trading Fraud Detection":
    st.header("📊 Real-Time Trading Fraud Detection")
    st.write("Upload trading dataset to detect anomalies.")

    uploaded_file = st.file_uploader("Upload Trades CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        with st.spinner("Analyzing trade patterns..."):
            anomalies = fraud_detection.detect_anomalies(df)
            time.sleep(2)
        st.success("Analysis Complete ✅")

        st.subheader("🚨 Anomalous Trades")
        st.dataframe(anomalies)

        st.subheader("📉 Fraud Detection Visuals")
        visualizations.plot_trade_volume(df)
        visualizations.plot_price_anomalies(df, anomalies)

# Investor FraudShield
elif selected == "📱 Investor FraudShield":
    st.header("📱 Investor FraudShield – Scam Message Detector")
    user_msg = st.text_area("Paste SMS/Email content here:")
    if st.button("Check Fraud Risk"):
        result = scam_detector.detect_scam(user_msg)
        if result["label"] == "scam":
            st.error(f"🚨 Scam Detected! Confidence: {result['score']:.2f}")
        else:
            st.success("✅ Looks Safe")

    st.subheader("Sample Scam Messages")
    scam_samples = pd.read_csv("data/phishing_samples.csv")
    st.table(scam_samples.head())

# Reports
elif selected == "📈 Reports":
    st.header("📈 Market Fraud Analysis Report")
    st.write("Automated insights from trade & scam detection.")
    visualizations.generate_report()
    st.download_button("⬇️ Download Report", "report.pdf")
app.py
