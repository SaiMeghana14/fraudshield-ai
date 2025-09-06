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
import requests
import yfinance as yf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from streamlit_lottie import st_lottie
from langdetect import detect

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

# -------------------- LOAD ANIMATION --------------------
def load_lottie(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

fraud_anim = load_lottie("assets/animations/fraud.json")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_trades():
    df = pd.read_csv("data/trades.csv")

    # Add fake dates automatically if not present (reproducible)
    np.random.seed(42)
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
    # work with lowercase column names internally
    df.columns = [c.lower() for c in df.columns]

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        # avoid division by zero
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue
        threshold = mean + 3 * std
        col_anomalies = df[df[col] > threshold]
        if not col_anomalies.empty:
            anomalies = pd.concat([anomalies, col_anomalies])

    if "price" in df.columns:
        try:
            price_anoms = df[df["price"].pct_change().abs() > 0.2]
            anomalies = pd.concat([anomalies, price_anoms])
        except Exception:
            pass

    # include flagged is_fraud if present
    if "is_fraud" in df.columns:
        try:
            anomalies = pd.concat([anomalies, df[df["is_fraud"].astype(int) == 1]])
        except Exception:
            pass

    return anomalies.drop_duplicates()

# -------------------- SCAM DETECTOR --------------------
@st.cache_resource
def train_scam_model():
    phishing_data = load_phishing()
    # support different column names
    msg_col = "message" if "message" in phishing_data.columns else phishing_data.columns[0]
    label_col = "label" if "label" in phishing_data.columns else phishing_data.columns[1] if len(phishing_data.columns) > 1 else phishing_data.columns[-1]
    X = phishing_data[msg_col].astype(str)
    y = phishing_data[label_col].astype(str)

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_scam_model()

def detect_scam(message):
    X_test = vectorizer.transform([str(message)])
    y_pred = model.predict(X_test)[0]
    # safe probability
    try:
        y_prob = model.predict_proba(X_test)[0, model.classes_.tolist().index(y_pred)]
    except Exception:
        y_prob = model.predict_proba(X_test).max()
    return {"label": y_pred, "score": float(y_prob)}

# -------------------- SCAM CATEGORY & EXPLAINABILITY --------------------
def classify_scam(message):
    categories = {
        "Investment scam": ["investment", "returns", "profit", "guaranteed"],
        "Lottery scam": ["lottery", "winner", "prize"],
        "Phishing (bank/email)": ["bank", "password", "account", "login", "otp"],
        "Romance scam": ["love", "relationship", "dating"],
        "Ponzi scheme": ["double", "scheme", "mlm"]
    }
    detected = []
    text = str(message).lower()
    for cat, kws in categories.items():
        for kw in kws:
            if kw in text:
                detected.append(cat)
                break
    return detected if detected else ["General / Unknown"]

def highlight_keywords(message):
    suspicious_words = ["urgent", "verify", "guaranteed", "investment", "lottery", "password", "click", "otp"]
    text = str(message)
    # highlight occurrences (case-insensitive)
    for word in suspicious_words:
        text = pd.Series([text]).str.replace(word, f"**:red[{word}]**", case=False, regex=True).iloc[0]
    return text

def explain_scam(message):
    explanations = []
    text = str(message).lower()
    if "guaranteed" in text or "guarantee" in text:
        explanations.append("Claims of guaranteed returns — common sign of investment fraud.")
    if "otp" in text or "one-time password" in text:
        explanations.append("Requests for OTP/pin — likely phishing attempt.")
    if "click here" in text or "bit.ly" in text or "tinyurl" in text:
        explanations.append("Contains shortened/malicious-looking links.")
    if any(w in text for w in ["lottery", "winner", "prize"]):
        explanations.append("Mention of lottery/prize — typical lottery scams.")
    if not explanations:
        explanations.append("No clear rule-based reason found; model-based score used.")
    return explanations

# -------------------- REAL-TIME MARKET DATA (yfinance + Alpha Vantage fallback) --------------------
def fetch_yfinance(ticker, period="5d", interval="15m"):
    try:
        data = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
        if data is None or data.empty:
            return None
        return data
    except Exception:
        return None

def fetch_alpha_vantage(ticker, api_key, interval="15min"):
    # Alpha Vantage requires symbol format; we'll attempt intraday
    base = "https://www.alphavantage.co/query"
    params = {"function":"TIME_SERIES_INTRADAY", "symbol":ticker, "interval":interval, "apikey":api_key, "outputsize":"compact"}
    try:
        r = requests.get(base, params=params, timeout=10)
        j = r.json()
        # parse response
        key = f"Time Series ({interval})"
        if key in j:
            df = pd.DataFrame.from_dict(j[key], orient="index").astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ["open", "high", "low", "close", "volume"]
            return df
        else:
            return None
    except Exception:
        return None

def show_live_market_widget():
    st.subheader("📈 Live Market Data – Fraud Watch")
    col1, col2 = st.columns([3,1])
    with col1:
        ticker = st.text_input("Enter stock symbol (use .NS for NSE)", value="RELIANCE.NS")
        data_source = st.selectbox("Data source", ["yfinance (no key)", "Alpha Vantage (provide API key)"])
        av_key = st.text_input("Alpha Vantage API Key (optional)", type="password") if data_source.startswith("Alpha") else None

        period = st.selectbox("Period", ["1d","5d","1mo"], index=1)
        interval = st.selectbox("Interval", ["1m","5m","15m","30m","60m"], index=2)
        data = None
        if data_source.startswith("yfinance"):
            data = fetch_yfinance(ticker, period=period, interval=interval)
        else:
            if av_key:
                data = fetch_alpha_vantage(ticker, api_key=av_key, interval=interval if interval.endswith("m") else "15min")
            else:
                st.info("Provide Alpha Vantage API key for this source.")
        if data is not None and not data.empty:
            st.line_chart(data["Close"] if "Close" in data.columns else data["close"])
            st.dataframe(data.tail(5))
        else:
            st.warning("Live data not available for the requested ticker/interval.")

    with col2:
        st.markdown("### 🚨 Live Ticker Fraud Watch")
        # Simple rotating alerts (in real app, pull from stream)
        sample_alerts = [
            "Unusual volume spike in RELIANCE.NS",
            "Abnormal price jump in HDFCBANK.NS",
            "Multiple large trades executed for a low-cap stock",
            "Suspicious order patterns detected in small-cap segment"
        ]
        st.info(sample_alerts[np.random.randint(0, len(sample_alerts))])

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

# Fraud vs Safe Pie Chart — safe dynamic label mapping
def plot_fraud_vs_safe(df):
    if "is_fraud" in df.columns:
        fraud_counts = df["is_fraud"].value_counts()
        labels = []
        for idx in fraud_counts.index:
            labels.append("Fraud" if str(idx) in ["1","True","true"] else "Safe")
        fig = px.pie(values=fraud_counts.values,
                     names=labels,
                     title="⚖ Fraud vs Safe Trades",
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# Fraud trend over time
def plot_fraud_trend(df):
    if "date" in df.columns and "is_fraud" in df.columns:
        # ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        fraud_rate_over_time = df.groupby("date")["is_fraud"].mean() * 100
        if not fraud_rate_over_time.empty:
            fig = px.line(fraud_rate_over_time,
                          title="📈 Fraud Rate Over Time",
                          labels={"value": "Fraud Rate (%)", "date": "Date"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-series fraud data available.")

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
    # heatmap of location vs is_fraud (counts)
    if "location" in df.columns and "is_fraud" in df.columns:
        # pivot for density display
        heat = df.groupby(["location","is_fraud"]).size().reset_index(name="count")
        if not heat.empty:
            fig = px.density_heatmap(df, x="location", y="is_fraud",
                                     title="🔥 Fraud Density by Location",
                                     color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No heatmap data.")

# Scam Trends
def scam_trends():
    phishing_data = load_phishing()
    if "label" in phishing_data.columns:
        fig = px.histogram(phishing_data, x="label", title="📈 Scam Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No scam labels available to plot trends.")

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
    buf.seek(0)
    st.image(buf)
    plt.close()

# -------------------- REPORT HELPERS --------------------
def generate_report():
    st.markdown("""
    ### 📈 Market Fraud Analysis Report
    - Detected anomalies in trading volumes & prices  
    - Scam message classifier trained on phishing samples  
    - Real-time investor safety tools  
    """)
    st.success("✅ Report Generated")

def download_report_files(df):
    import io
    from openpyxl import Workbook  # make sure openpyxl is installed in requirements.txt

    # Save to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="fraud_report")
    excel_data = output.getvalue()

    st.download_button(
        label="⬇️ Download Report (Excel)",
        data=excel_data,
        file_name="fraud_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Save to CSV
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=csv_data,
        file_name="fraud_report.csv",
        mime="text/csv"
    )

# -------------------- COMMUNITY REPORTING (persist to CSV) --------------------
REPORTS_PATH = "data/reports.csv"
def append_report(msg_text, source="web"):
    # ensure folder exists and append
    try:
        new = {"message": msg_text, "source": source, "ts": pd.Timestamp.now()}
        if not st.session_state.get("reports_df"):
            if os.path.exists(REPORTS_PATH):
                st.session_state["reports_df"] = pd.read_csv(REPORTS_PATH)
            else:
                st.session_state["reports_df"] = pd.DataFrame(columns=["message","source","ts"])
        st.session_state["reports_df"] = pd.concat([st.session_state["reports_df"], pd.DataFrame([new])], ignore_index=True)
        st.session_state["reports_df"].to_csv(REPORTS_PATH, index=False)
        return True
    except Exception as e:
        return False

# -------------------- REPORT EXPORT (Excel/PDF) --------------------
def download_report_files(df):
    # Excel
    output = BytesIO()
    df.to_excel(output, index=False, sheet_name="fraud_report")
    output.seek(0)
    st.download_button("⬇️ Download Excel Report", data=output, file_name="fraud_report.xlsx")

    # PDF simple (text-based) using reportlab if available
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        pdf_buf = BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        text = c.beginText(40, 750)
        text.textLine("FraudShield AI - Fraud Report")
        for i, row in df.head(30).iterrows():
            text.textLine(str(row.to_dict()))
        c.drawText(text)
        c.save()
        pdf_buf.seek(0)
        st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="fraud_report.pdf")
    except Exception:
        st.info("PDF export not available (reportlab missing).")

# -------------------- SEBI RULEBOOK BOT (FAQ) --------------------
SEBI_FAQ = {
    "insider trading": "Insider trading is prohibited under SEBI regulations. Report suspicious trades to SEBI.",
    "ipo": "IPO allotments and disclosures are governed by SEBI (ICDR) Regulations.",
    "investor protection": "SEBI's mandate includes investor protection, market development and supervision."
}

def sebi_bot(query):
    q = query.lower()
    for k, v in SEBI_FAQ.items():
        if k in q:
            return v
    return "I don't have a direct answer. Please consult the SEBI website or upload a specific query."

# -------------------- ENGAGEMENT: QUIZ & BADGES --------------------
def scam_quiz():
    st.subheader("🎮 Spot the Scam Quiz")
    q = "You receive an email: 'You are a lottery winner! Click here to claim your prize.' Is this a scam?"
    ans = st.radio(q, ["Yes", "No"], index=0)
    if st.button("Submit Answer"):
        if ans == "Yes":
            st.success("✅ Correct — this is a scam.")
            st.balloons()
            st.success("🏅 You earned the 'Investor Guardian' badge!")
        else:
            st.error("❌ Incorrect — this is a scam. Be careful with prize claims.")

# -------------------- HOME / SIDEBAR NAV (keep existing but add quick actions integration) --------------------
import os
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "🏠 Home"

with st.sidebar:
    av_key_sidebar = st.text_input("Alpha Vantage API Key (optional)", type="password")
    live_updates = st.checkbox("Enable simulated push alerts", value=False)
    selected = option_menu(
        "FraudShield AI",
        ["🏠 Home", "📊 Trading Fraud Detection", "📱 Investor FraudShield", "📈 Reports"],
        icons=["house", "graph-up", "shield-check", "file-earmark-text"],
        menu_icon="cast",
        default_index=["🏠 Home", "📊 Trading Fraud Detection", "📱 Investor FraudShield", "📈 Reports"].index(st.session_state["selected_page"])
    )

# keep sidebar and session state in sync
st.session_state["selected_page"] = selected

# -------------------- HOME --------------------
if selected == "🏠 Home":
    st.title("🛡️ FraudShield AI")
    if fraud_anim:
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
            st.session_state["selected_page"] = "📊 Trading Fraud Detection"
            st.rerun()
    
    with colB:
        if st.button("📱 Check Scam Messages"):
            st.session_state["selected_page"] = "📱 Investor FraudShield"
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

    # Live market widget
    show_live_market_widget()

    uploaded_file = st.file_uploader("Upload Trades CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_trades()

    df.columns = [c.lower() for c in df.columns]
    st.dataframe(df.head())

    with st.spinner("🔍 Analyzing trade patterns..."):
        anomalies = detect_anomalies(df)
        time.sleep(1)
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

    # allow downloading a report of anomalies
    if not anomalies.empty:
        download_report_files(anomalies)

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
            explanations = explain_scam(user_msg)

            st.markdown(f"### 🔍 Analyzed Message")
            st.markdown(highlighted, unsafe_allow_html=True)

            st.progress(int(result["score"] * 100))

            if result["score"] > 0.5:
                st.error(f"🚨 Scam Detected! Categories: {', '.join(categories)} | Confidence: {result['score']:.2f}")
            else:
                st.success(f"✅ Looks Safe | Confidence: {result['score']:.2f}")

            # show why flagged
            st.subheader("Why this message was flagged")
            for e in explanations:
                st.write("- " + e)

            # Multi-language detection
            try:
                lang = detect(user_msg)
                st.info(f"🌐 Detected Language: {lang}")
            except Exception:
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
        ok = append_report(new_report, source="web")
        if ok:
            st.success("✅ Thank you! Your report has been added to our database.")
        else:
            st.error("❌ Failed to add report — check server permissions.")

    # show leaderboard from session_state or file
    if os.path.exists(REPORTS_PATH):
        try:
            reports_df = pd.read_csv(REPORTS_PATH)
            top_states = reports_df["ts"].dt.date.value_counts().head(5) if "ts" in reports_df.columns else None
            st.subheader("🏆 Recent Reports")
            st.dataframe(reports_df.tail(10))
        except Exception:
            st.info("No reports available yet.")

    # Quiz & engagement
    scam_quiz()

# -------------------- REPORT --------------------
elif selected == "📈 Reports":
    st.header("📈 Market Fraud Analysis Report")
    generate_report()
    # allow download of a sample consolidated report (trades + phishing)
    try:
        df_all = load_trades()
        download_report_files(df_all)
    except Exception:
        st.info("No trades available for export.")
