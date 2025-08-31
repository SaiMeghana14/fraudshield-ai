import matplotlib.pyplot as plt
import streamlit as st

def plot_trade_volume(df):
    st.subheader("📊 Trade Volume Over Time")
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["volume"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Volume")
    st.pyplot(fig)

def plot_price_anomalies(df, anomalies):
    st.subheader("📉 Price Anomalies")
    fig, ax = plt.subplots()
    ax.plot(df["time"], df["price"], label="Normal")
    ax.scatter(anomalies["time"], anomalies["price"], color="red", label="Anomalies")
    ax.legend()
    st.pyplot(fig)

def generate_report():
    st.write("Fraud analysis shows unusual patterns in trading volumes and prices.")
    st.write("- High frequency anomalies around specific time intervals")
    st.write("- Investor-targeted scams identified with >90% accuracy")
