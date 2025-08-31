import pandas as pd

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    anomalies = []

    for idx, row in df.iterrows():
        # flag trade if quantity is abnormally high
        if row["Quantity"] > df["Quantity"].mean() * 5:
            anomalies.append(idx)
        # flag trade if price deviates drastically
        elif abs(row["Price"] - df["Price"].mean()) > 3 * df["Price"].std():
            anomalies.append(idx)

    return df.loc[anomalies]
