import plotly.express as px
import streamlit as st
import pandas as pd
from . import report_generator

def plot_trade_volume(df: pd.DataFrame):
    fig = px.bar(df, x="Stock", y="Quantity", title="📊 Trade Volumes by Stock",
                 color="Stock", height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_price_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame):
    fig = px.scatter(df, x="Stock", y="Price", size="Quantity", color="Stock",
                     title="📉 Stock Prices with Anomalies")
    if not anomalies.empty:
        fig.add_scatter(x=anomalies["Stock"], y=anomalies["Price"],
                        mode="markers", marker=dict(size=15, color="red"),
                        name="Anomalies")
    st.plotly_chart(fig, use_container_width=True)

def generate_report():
    st.write("📄 Generating Fraud Analysis Report...")
    report_path = report_generator.create_pdf()
    with open(report_path, "rb") as f:
        st.download_button("⬇️ Download Report PDF", f, file_name="fraud_report.pdf")
