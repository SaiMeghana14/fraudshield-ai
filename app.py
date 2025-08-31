import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import time
import json
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from streamlit_lottie import st_lottie

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

# -------------------- LOAD ANIMATION --------------------
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

fraud_anim = load_lottie("assets/animations/fraud.json")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_trades():
    return pd.read_csv("data/trades.csv")

@st.cache_data
def load_phishing():
    return pd.read_csv("data/phishing_samples.csv")

# -------------------- FRAUD DETECTION --------------------
def detect_anomalies(df):
    anomalies = pd.DataFrame()

    # normalize column names
    df.columns = [c.lower() for c in df.columns]

    # numeric anomaly detection
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        threshold = df[col].mean() + 3 * df[col].std()
        col_anomalies = df[df[col] > threshold]
        if not col_anomalies.empty:
            anomalies = pd.concat([anomalies, col_anomalies])

    # if "price" exists, check pct_change anomalies
    if "price" in df.columns:
        price_anoms = df[df["price"].pct_change().abs() > 0.2]
        anomalies = pd.concat([anomalies, price_anoms])

    return anomalies.drop_duplicates()

# -------------------- SCAM DETECTOR --------------------
@st.cache_resource
def train_scam_model():
    phishing_data = load_phishing()
    X = phishing_data["message"]
    y = phishing_data["label"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_scam_model()

def detect_scam(message):
    X_test = vectorizer.transform([message])
    y_pred = model.predict(X_test)[0]
    y_prob = model.predict_proba(X_test)[0, model.classes_.tolist().index(y_pred)]
    return {"label": y_pred, "score": y_prob}

# -------------------- VISUALIZATIONS --------------------
def plot_trade_volume(df):
    if "stock" in df.columns and "amount" in df.columns:
        fig = px.bar(df, x="stock", y="amount", color="stock", title="📊 Trade Volume per Stock")
        st.plotly_chart(fig, use_container_width=True)

def plot_price_anomalies(df, anomalies):
    if "price" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="lines+markers", name="Price"))
        if not anomalies.empty and "price" in anomalies.columns:
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies["price"],
                mode="markers", marker=dict(color="red", size=12),
                name="Anomalies 🚨"
            ))
        fig.update_layout(title="📉 Price Trends & Anomalies")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- EXTRA FRAUD CHARTS --------------------
def plot_fraud_rate(df, anomalies):
    fraud_rate = len(anomalies) / len(df) * 100 if len(df) > 0 else 0
    st.metric("📊 Fraud Rate", f"{fraud_rate:.2f}%")

def plot_fraud_by_location(df):
    if "location" in df.columns:
        fig = px.histogram(df, x="location", title="🌍 Fraud by Location", color="location")
        st.plotly_chart(fig, use_container_width=True)

def plot_amount_distribution(df):
    if "amount" in df.columns:
        fig = px.histogram(df, x="amount", nbins=30, title="💰 Transaction Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- REPORT --------------------
def generate_report():
    st.markdown("""
    ### 📈 Market Fraud Analysis Report
    - Detected anomalies in trading volumes & prices  
    - Scam message classifier trained on phishing samples  
    - Real-time investor safety tools  
    """)
    st.success("✅ Report Generated")

# -------------------- SIDEBAR NAV --------------------
with st.sidebar:
    selected = option_menu(
        "FraudShield AI",
        ["🏠 Home", "📊 Trading Fraud Detection", "📱 Investor FraudShield", "📈 Reports"],
        icons=["house", "graph-up", "shield-check", "file-earmark-text"],
        menu_icon="cast",
        default_index=0
    )

# -------------------- HOME --------------------
if selected == "🏠 Home":
    st.title("🛡️ FraudShield AI")
    st_lottie(fraud_anim, height=300, key="fraud")
    st.markdown("### Protecting Every Trade, Securing Every Investor.")
    st.info("An AI-powered platform for fraud detection & investor protection aligned with SEBI’s mandate.")

# -------------------- TRADING FRAUD DETECTION --------------------
elif selected == "📊 Trading Fraud Detection":
    st.header("📊 Real-Time Trading Fraud Detection")

    uploaded_file = st.file_uploader("Upload Trades CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_trades()

    # normalize column names for consistency
    df.columns = [c.lower() for c in df.columns]

    st.dataframe(df.head())

    with st.spinner("🔍 Analyzing trade patterns..."):
        st.write("Columns in dataset:", df.columns.tolist())
        anomalies = detect_anomalies(df)
        time.sleep(2)
    st.success("Analysis Complete ✅")

    st.subheader("🚨 Anomalous Trades")
    if anomalies.empty:
        st.success("No anomalies detected ✅")
    else:
        st.dataframe(anomalies)

    st.subheader("📉 Fraud Detection Visuals")
    plot_trade_volume(df)
    plot_price_anomalies(df, anomalies)

    # NEW FRAUD CHARTS
    st.subheader("📊 Fraud Insights")
    plot_fraud_rate(df, anomalies)
    plot_fraud_by_location(anomalies)
    plot_amount_distribution(anomalies)

# -------------------- INVESTOR FRAUDSHIELD --------------------
elif selected == "📱 Investor FraudShield":
    st.header("📱 Investor FraudShield – Scam Message Detector")
    user_msg = st.text_area("Paste SMS/Email content here:")
    if st.button("Check Fraud Risk"):
    if not user_msg.strip():
        st.warning("⚠️ Please enter a message first.")
    else:
        result = detect_scam(user_msg)
        
    if result == detect_scam(user_msg):
        st.error(f"🚨 Scam Detected! Confidence: {result['score']:.2f}")
    else:
        st.success(f"✅ Looks Safe (Confidence: {result['score']:.2f})")

    st.subheader("📋 Sample Scam Messages")
    scam_samples = load_phishing()
    st.table(scam_samples.head())

# -------------------- REPORT --------------------
elif selected == "📈 Reports":
    st.header("📈 Market Fraud Analysis Report")
    generate_report()
    st.download_button(
    label="⬇️ Download Report",
    data="Automated Fraud Report Generated.",
    file_name="fraud_report.txt"
)

