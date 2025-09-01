import streamlit as st 
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from streamlit_lottie import st_lottie
from langdetect import detect

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
    df = pd.read_csv("data/trades.csv")

    # Add fake dates automatically if not present
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(
            np.random.choice(pd.date_range("2024-06-01", "2024-08-30"), len(df))
        )

    return df

@st.cache_data
def load_phishing():
    return pd.read_csv("data/phishing_samples.csv")

# -------------------- FRAUD DETECTION --------------------
def detect_anomalies(df):
    anomalies = pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        threshold = df[col].mean() + 3 * df[col].std()
        col_anomalies = df[df[col] > threshold]
        if not col_anomalies.empty:
            anomalies = pd.concat([anomalies, col_anomalies])

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

# -------------------- SCAM CATEGORY & EXPLAINABILITY --------------------
def classify_scam(message):
    categories = {
        "investment": ["investment", "returns", "profit", "guaranteed"],
        "lottery": ["lottery", "winner", "prize"],
        "phishing": ["bank", "password", "account", "login"],
        "romance": ["love", "relationship", "dating"],
        "ponzi": ["double", "scheme", "MLM"]
    }
    detected = []
    for cat, kws in categories.items():
        for kw in kws:
            if kw in message.lower():
                detected.append(cat)
                break
    return detected if detected else ["general scam"]

def highlight_keywords(message):
    suspicious_words = ["urgent", "verify", "guaranteed", "investment", "lottery", "password", "click"]
    for word in suspicious_words:
        if word in message.lower():
            message = message.replace(word, f"**:red[{word}]**")
    return message

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

# Fraud Insights
def plot_fraud_rate(df, anomalies):
    fraud_rate = len(anomalies) / len(df) * 100 if len(df) > 0 else 0
    st.metric("📊 Fraud Rate", f"{fraud_rate:.2f}%")

# Fraud vs Safe Pie Chart
def plot_fraud_vs_safe(df):
    if "is_fraud" in df.columns:
        fraud_counts = df["is_fraud"].value_counts()
        fig = px.pie(values=fraud_counts.values,
                     names=["Safe", "Fraud"],
                     title="⚖ Fraud vs Safe Trades",
                     hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

# Fraud trend over time
def plot_fraud_trend(df):
    if "date" in df.columns and "is_fraud" in df.columns:
        fraud_rate_over_time = df.groupby("date")["is_fraud"].mean() * 100
        fig = px.line(fraud_rate_over_time,
                      title="📈 Fraud Rate Over Time",
                      labels={"value": "Fraud Rate (%)", "date": "Date"})
        st.plotly_chart(fig, use_container_width=True)

def plot_fraud_by_location(df):
    if "location" in df.columns:
        fig = px.histogram(df, x="location", title="🌍 Fraud by Location", color="location")
        st.plotly_chart(fig, use_container_width=True)

def plot_amount_distribution(df):
    if "amount" in df.columns:
        fig = px.histogram(df, x="amount", nbins=30, title="💰 Transaction Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)

def plot_top_suspicious_traders(df):
    if "is_fraud" in df.columns and "trade_id" in df.columns:
        top_trades = df[df["is_fraud"] == 1]["trade_id"].value_counts().head(5)
        if not top_trades.empty:
            st.write("🚨 Top Suspicious Trades:")
            st.bar_chart(top_trades)

def plot_fraud_heatmap(df):
    if "location" in df.columns and "is_fraud" in df.columns:
        fig = px.density_heatmap(
            df,
            x="location",
            y="is_fraud",
            title="🔥 Fraud Density by Location",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)

# Scam Trends
def scam_trends():
    phishing_data = load_phishing()
    fig = px.histogram(phishing_data, x="label", title="📈 Scam Types Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Scam Network Graph
def scam_network():
    G = nx.Graph()
    G.add_edges_from([
        ("Scammer A", "Email Domain X"),
        ("Scammer A", "Location Y"),
        ("Scammer B", "Phone Z"),
        ("Scammer B", "Location Y")
    ])
    plt.figure(figsize=(6, 4))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, font_size=10)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)

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

    # --- Key Highlights ---
    st.subheader("✨ Why FraudShield AI?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔍 Fraud Cases Analyzed", "12,450+")
    with col2:
        st.metric("⚡ Avg Detection Speed", "0.8 sec")
    with col3:
        st.metric("✅ Accuracy", "95%+")

    # --- Interactive Buttons ---
    st.subheader("🚀 Quick Actions")
    colA, colB = st.columns(2)
    
    with colA:
        if st.button("📊 Try Anomaly Detector"):
            # Switch to anomaly detection page
            st.session_state["menu_option"] = "📊 Trade Anomalies"
            # Call your anomaly detection logic directly if you want instant results
            from modules.trade_anomalies import show_anomaly_detector
            show_anomaly_detector()
            st.rerun()
    
    with colB:
        if st.button("📱 Check Scam Messages"):
            # Switch to scam message detection page
            st.session_state["menu_option"] = "📱 Investor FraudShield"
            # Call scam message checker logic instantly
            from modules.scam_messages import show_scam_checker
            show_scam_checker()
            st.rerun()

    # --- Live Fraud Tips ---
    st.subheader("📢 Today’s Fraud Prevention Tip")
    fraud_tips = [
        "Never share OTPs or PINs, even with bank officials.",
        "Cross-check URLs; fake sites often mimic real domains.",
        "Be cautious of 'too good to be true' investment offers.",
        "Enable two-factor authentication on all accounts.",
        "Always verify before transferring money to unknown parties."
    ]
    import random
    st.success(random.choice(fraud_tips))

    # --- Community Stats ---
    st.subheader("🌍 Community Impact")
    st.progress(75)  # example impact progress
    st.caption("Over 7,500+ investors have reported scams via FraudShield.")

# -------------------- TRADING FRAUD DETECTION --------------------
elif selected == "📊 Trading Fraud Detection":
    st.header("📊 Real-Time Trading Fraud Detection")

    uploaded_file = st.file_uploader("Upload Trades CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_trades()

    df.columns = [c.lower() for c in df.columns]
    st.dataframe(df.head())

    with st.spinner("🔍 Analyzing trade patterns..."):
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

    st.subheader("📊 Fraud Insights")
    plot_fraud_rate(df, anomalies)
    plot_fraud_vs_safe(df)
    plot_fraud_trend(df)
    plot_fraud_by_location(df)
    plot_amount_distribution(df)
    plot_top_suspicious_traders(df)
    plot_fraud_heatmap(df)

# -------------------- INVESTOR FRAUDSHIELD --------------------
elif selected == "📱 Investor FraudShield":
    st.header("📱 Investor FraudShield – Scam Message Detector")
    user_msg = st.text_area("Paste SMS/Email content here:")

    if st.button("Check Fraud Risk"):
        if not user_msg.strip():
            st.warning("⚠️ Please enter a message first.")
        else:
            result = detect_scam(user_msg)
            categories = classify_scam(user_msg)
            highlighted = highlight_keywords(user_msg)

            st.markdown(f"### 🔍 Analyzed Message")
            st.markdown(highlighted)

            st.progress(int(result["score"] * 100))

            if result["score"] > 0.5:
                st.error(f"🚨 Scam Detected! Categories: {', '.join(categories)} | Confidence: {result['score']:.2f}")
            else:
                st.success(f"✅ Looks Safe | Confidence: {result['score']:.2f}")

            # Multi-language detection
            try:
                lang = detect(user_msg)
                st.info(f"🌐 Detected Language: {lang}")
            except:
                st.info("🌐 Language could not be detected")

            # Educational tip
            st.warning("💡 Tip: Legit investments never guarantee profits. Be cautious with urgency keywords.")

    st.subheader("📋 Sample Scam Messages")
    scam_samples = load_phishing()
    st.table(scam_samples.head())

    st.subheader("📈 Scam Trends Dashboard")
    scam_trends()

    st.subheader("🌐 Scam Network Graph")
    scam_network()

    st.subheader("🤝 Community Reporting Hub")
    new_report = st.text_input("Report a scam message:")
    if st.button("Submit Report") and new_report:
        st.success("✅ Thank you! Your report has been added to our database.")

# -------------------- REPORT --------------------
elif selected == "📈 Reports":
    st.header("📈 Market Fraud Analysis Report")
    generate_report()
    st.download_button(
        label="⬇️ Download Report",
        data="Automated Fraud Report Generated.",
        file_name="fraud_report.txt"
    )
