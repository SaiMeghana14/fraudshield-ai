import plotly.express as px
import streamlit as st
import pandas as pd
from . import report_generator

# -------------------- Trading Visuals --------------------

def plot_trade_volume(df: pd.DataFrame):
    """Bar chart showing stock volumes traded."""
    fig = px.bar(
        df, x="Stock", y="Quantity",
        title="📊 Trade Volumes by Stock",
        color="Stock", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_price_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame):
    """Scatter plot showing stock prices with anomalies highlighted."""
    fig = px.scatter(
        df, x="Stock", y="Price",
        size="Quantity", color="Stock",
        title="📉 Stock Prices with Anomalies"
    )
    if not anomalies.empty:
        fig.add_scatter(
            x=anomalies["Stock"], y=anomalies["Price"],
            mode="markers", marker=dict(size=15, color="red"),
            name="🚨 Anomalies"
        )
    st.plotly_chart(fig, use_container_width=True)


# -------------------- Fraud Detection Visuals --------------------

def fraud_rate_chart(df: pd.DataFrame):
    """Bar chart of fraud vs legit counts."""
    counts = df["is_fraud"].value_counts().reset_index()
    counts.columns = ["is_fraud", "count"]
    counts["is_fraud"] = counts["is_fraud"].map({0: "✅ Legit", 1: "🚨 Fraud"})
    fig = px.bar(
        counts, x="is_fraud", y="count", color="is_fraud",
        title="🔍 Overall Fraud Rate", text="count", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def fraud_by_location_chart(df: pd.DataFrame):
    """Pie chart showing fraudulent trades by location."""
    fraud_counts = df[df["is_fraud"] == 1]["location"].value_counts().reset_index()
    fraud_counts.columns = ["location", "count"]
    if fraud_counts.empty:
        st.info("✅ No fraudulent trades detected by location.")
    else:
        fig = px.pie(
            fraud_counts, names="location", values="count",
            title="🌍 Fraudulent Trades by Location"
        )
        st.plotly_chart(fig, use_container_width=True)


def transaction_amount_distribution(df: pd.DataFrame):
    """Histogram of trade amounts split by fraud flag."""
    fig = px.histogram(
        df, x="amount", color="is_fraud",
        nbins=20, title="💰 Transaction Amount Distribution (Fraud vs Legit)"
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------- Report Generation --------------------

def generate_report():
    """Generate a fraud analysis PDF report with download button."""
    st.write("📄 Generating Fraud Analysis Report...")
    report_path = report_generator.create_pdf()
    with open(report_path, "rb") as f:
        st.download_button(
            "⬇️ Download Report PDF", f, file_name="fraud_report.pdf"
        )
