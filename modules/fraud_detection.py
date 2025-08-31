import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    model = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly"] = model.fit_predict(df[["price", "volume"]])
    anomalies = df[df["anomaly"] == -1]
    return anomalies
