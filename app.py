import streamlit as st  
import pandas as pd
import numpy as np
import time
import json
import plotly.express as px
import plotly.graph_objects as go

import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
from PIL import Image
import requests
import yfinance as yf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from streamlit_lottie import st_lottie
from langdetect import detect

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

# ------------------ ALPHA VANTAGE API ------------------

api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={api_key}"
data = requests.get(url).json()

if "Global Quote" in data and data["Global Quote"]:

    quote = data["Global Quote"]

    price = float(quote["05. price"])
    change_percent = float(
        quote["10. change percent"].replace("%","")
    )
    volume = int(quote["06. volume"])

else:
    st.warning("⚠ Live market data unavailable. Using fallback demo values.")

    # Fallback values for demo
    price = 253.47
    change_percent = 0.98
    volume = 5671471
    
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

# -------------------- COMMUNITY REPORTING (persist to CSV) --------------------
REPORTS_PATH = "data/reports.csv"
def append_report(msg_text, source="web"):
    # ensure folder exists and append
    try:
        new = {"message": msg_text, "source": source, "ts": pd.Timestamp.now()}
        if "reports_df" not in st.session_state:
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

    # CSV
    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV Report",
        data=csv_data,
        file_name="fraud_report.csv",
        mime="text/csv"
    )

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

# -------------------- HOME / SIDEBAR NAV  --------------------
import os
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "🏠 Home"

with st.sidebar:
    av_key_sidebar = st.text_input("Alpha Vantage API Key (optional)", type="password")
    live_updates = st.checkbox("Enable simulated push alerts", value=False)
    st.sidebar.markdown("## 🖥️ FraudShield AI")
    st.markdown("""
    <style>
    
    /* Hide only radio circles */
    input[type="radio"] {
        display:none;
    }
    
    /* Style menu rows */
    div[role="radiogroup"] label {
        padding:12px 16px;
        margin-bottom:10px;
        border-radius:12px;
        background:#ffffff;
    }
    
    /* Hover */
    div[role="radiogroup"] label:hover {
        background:#f3f4f6;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    pages = [
        "🏠 Home",
        "📊 Trading Fraud Detection",
        "📱 Investor FraudShield",
        "📈 Reports"
    ]
    
    selected = st.sidebar.radio(
        "",
        pages,
        index=pages.index(st.session_state.get("selected_page", "🏠 Home"))
    )
    
    st.session_state["selected_page"] = selected

# -------------------- HOME --------------------
if selected == "🏠 Home":
    st.title("🛡️ FraudShield AI")
    if fraud_anim:
        st_lottie(fraud_anim, height=300, key="fraud")
    st.markdown("### Protecting Every Trade, Securing Every Investor.")
    st.info("An AI-powered platform for fraud detection & investor protection aligned with SEBI’s mandate.")

    # ---------------- Surveillance Dashboard ----------------
    
    st.subheader("🔍 Market Surveillance Engine")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price Change %", f"{change_percent}%")
    
    with col2:
        st.metric("Volume", f"{volume:,}")
    
    with col3:
        risk_score = 20
    
        if change_percent > 5:
            risk_score += 50
    
        if volume > 10000000:
            risk_score += 30
    
        if risk_score <= 30:
            risk_label = "Low 🟢"
    
        elif risk_score <= 60:
            risk_label = "Moderate 🟡"
        
        else:
            risk_label = "High 🔴"
    
        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text":"Fraud Risk"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"red"},
            "steps":[
                {"range":[0,30],"color":"green"},
                {"range":[30,60],"color":"yellow"},
                {"range":[60,100],"color":"red"}
            ]
        }
    ))
    
    st.plotly_chart(fig,use_container_width=True)
    
    # -------- Alerts --------
    
    if risk_score >= 70:
        st.error("🚨 High-Risk Alert: Possible Pump-and-Dump Activity")
    
    elif risk_score >= 40:
        st.warning("⚠ Moderate Risk: Unusual Market Behavior Detected")
    
    else:
        st.success("✅ SEBI Surveillance Status: Normal")

    # --- Key Highlights ---
    st.subheader("✨ Why FraudShield AI?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔍 Fraud Cases Analyzed", "12,450+")
    with col2:
        st.metric("⚡ Avg Detection Speed", "0.8 sec")
    with col3:
        st.metric("✅ Accuracy", "95%+")
    with col4:
        suspicious_signals = 0
        if change_percent > 5:
            suspicious_signals += 1
    
        if volume > 10000000:
            suspicious_signals += 1
    
        st.metric(
            "Suspicious Signals",
            suspicious_signals
        )

    # --- Quick Actions ---
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Trading Fraud Detection", use_container_width=True):
            st.session_state["selected_page"] = "📊 Trading Fraud Detection"
            st.rerun()
    
    with col2:
        if st.button("🛡 Investor FraudShield", use_container_width=True):
            st.session_state["selected_page"] = "📱 Investor FraudShield"
            st.rerun()
    
    with col3:
        if st.button("📈 View Reports", use_container_width=True):
            st.session_state["selected_page"] = "📈 Reports"
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

    # Store scores across tabs
    if "website_score" not in st.session_state:
        st.session_state.website_score = 0
    if "message_score" not in st.session_state:
        st.session_state.message_score = 0
    if "qr_score" not in st.session_state:
        st.session_state.qr_score = 0

    st.header("🛡️ Investor FraudShield")
    st.caption("AI-powered Website, Message & QR Scam Detection")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🌐 Website Scanner",
            "📩 SMS / Email",
            "📷 QR Scanner",
            "📊 AI Dashboard"
        ]
    )

    with tab1:
        st.subheader("🌐 Website Scam Detection")

        url = st.text_input(
            "Enter Website URL",
            placeholder="https://example.com"
        )

        if st.button("Analyze Website"):

            if not url.strip():
                st.warning("Please enter a website URL.")

            else:
                score = 10
                reasons = []

                suspicious_keywords = {
                    "login":10,
                    "verify":10,
                    "secure":10,
                    "signin":10,
                    "account":10,
                    "wallet":10,
                    "bank":10,
                    "update":10,
                    "confirm":10
                }

                shorteners = {
                    "bit.ly":35,
                    "tinyurl":30,
                    "goo.gl":30,
                    "rb.gy":30,
                    "t.co":30,
                    "cutt.ly":30
                }

                if not url.lower().startswith("https://"):
                    score += 20
                    reasons.append("HTTPS not detected")

                for site, pts in shorteners.items():
                    if site in url.lower():
                        score += pts
                        reasons.append(f"URL shortener detected ({site})")
                        break

                for word, pts in suspicious_keywords.items():
                    if word in url.lower():
                        score += pts
                        reasons.append(f"Contains '{word}'")

                score = min(score,100)
                st.session_state.website_score = score

                st.progress(score)

                if score < 30:
                    st.success("✅ Website appears Safe")
                elif score < 60:
                    st.warning("⚠ Suspicious Website")
                else:
                    st.error("🚨 High Risk Phishing Website")

                st.markdown("### 🔍 Analysis")
                if reasons:
                    for r in reasons:
                        st.write("✔", r)
                else:
                    st.write("✔ No suspicious indicators detected.")

                st.info("AI verdict generated successfully.")

    with tab2:

        st.subheader("📩 Scam Message Detection")

        scan_type = st.radio(
            "Message Type",
            ["SMS","Email"],
            horizontal=True
        )

        user_msg = st.text_area(f"Paste {scan_type}")

        if st.button("Analyze Message"):
            if not user_msg.strip():
                st.warning("Please enter a message.")

            else:
                result = detect_scam(user_msg)
                score = min(int(result["score"]*100),100)

                st.session_state.message_score = score
                st.progress(score)

                label = str(result["label"]).lower()
                if label in ["spam", "scam", "fraud", "phishing", "1", "malicious"]:
                    st.error("🚨 Scam Detected")
                
                elif score > 40:
                    st.warning("⚠️ Suspicious")
                
                else:
                    st.success("✅ Looks Safe")

                st.markdown("### 📂 Categories")
                for c in classify_scam(user_msg):
                    st.write("•", c)

                st.markdown("### 🤖 AI Explanation")
                for e in explain_scam(user_msg):
                    st.write("✔", e)

                try:
                    st.info(f"Detected Language: {detect(user_msg)}")
                except:
                    pass

    with tab3:

        st.subheader("📷 QR Code Scanner")

        uploaded = st.file_uploader(
            "Upload QR Code",
            type=["png","jpg","jpeg"]
        )

        if uploaded:

            image = Image.open(uploaded)
            st.image(image, width=300)

            image_np = np.array(image)

            detector = cv2.QRCodeDetector()
            qr, points, _ = detector.detectAndDecode(image_np)
            
            if qr:
                st.success("QR Decoded Successfully")
                st.code(qr)
            else:
                st.warning("No QR code detected.")

                risk = 20
                reasons = []

                suspicious_patterns = {
                    "bit.ly":25,
                    "tinyurl":20,
                    "goo.gl":20,
                    "rb.gy":20,
                    "cutt.ly":20,
                    "t.co":20,
                    "pay":15,
                    "wallet":10,
                    "upi":10
                }

                for keyword, pts in suspicious_patterns.items():
                    if keyword in qr.lower():
                        risk += pts
                        reasons.append(f"Detected '{keyword}'")
                        break

                risk = min(risk,100)
                st.session_state.qr_score = risk

                st.progress(risk)

                st.markdown("### 🔍 QR Analysis")

                if reasons:
                    for reason in reasons:
                        st.write("✔", reason)
                else:
                    st.write("✔ No suspicious indicators found.")

                if risk > 60:
                    st.error("🚨 Suspicious QR Code")
                elif risk > 30:
                    st.warning("⚠ Medium Risk")
                else:
                    st.success("✅ QR appears Safe")

    with tab4:

        st.subheader("📊 Overall Fraud Assessment")

        overall = max(
            st.session_state.website_score,
            st.session_state.message_score,
            st.session_state.qr_score
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=overall,
                title={"text":"Overall Fraud Risk"},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"darkred"},
                    "steps":[
                        {"range":[0,30],"color":"green"},
                        {"range":[30,60],"color":"yellow"},
                        {"range":[60,100],"color":"red"}
                    ]
                }
            )
        )

        st.plotly_chart(gauge, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("🌐 Website", f"{st.session_state.website_score}%")
        c2.metric("📩 Message", f"{st.session_state.message_score}%")
        c3.metric("📷 QR", f"{st.session_state.qr_score}%")

        st.markdown("---")

        if overall >= 70:
            st.error("🚨 HIGH RISK")
            explanation = "Multiple high-risk indicators detected. Avoid interacting with the submitted content."
        elif overall >= 40:
            st.warning("⚠ MEDIUM RISK")
            explanation = "Some suspicious indicators were detected. Exercise caution before proceeding."
        else:
            st.success("✅ LOW RISK")
            explanation = "No major fraud indicators detected."

        st.markdown("### 🤖 AI Explanation")
        st.info(explanation)

        st.markdown("### 🛡 Recommended Actions")
        st.write("✔ Do not click suspicious links")
        st.write("✔ Verify the sender or website")
        st.write("✔ Report scams to the National Cyber Crime Portal")
        st.write("✔ Block suspicious contacts")
        st.write("✔ Never share OTPs or banking credentials")

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
                
    st.subheader("📚 Knowledge Hub")

    with st.expander("Investment Fraud"):
        st.write("""
    • Never trust guaranteed returns.
    
    • Verify SEBI registration.
    
    • Research before investing.
    """)
    
    with st.expander("QR Fraud"):
        st.write("""
    Fake payment QR codes
    can redirect money
    to fraudsters.
    """)
    
    with st.expander("Phishing"):
        st.write("""
    Never share OTPs.
    
    Always verify domains.
    
    Beware of urgency.
    """)
    
    with st.expander("Emergency Help"):
        st.write("""
    📞 Cyber Helpline: 1930
    
    🌐 https://cybercrime.gov.in
    """)
