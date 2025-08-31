import re
import pandas as pd

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in stock trades based on abnormal amount values.
    Works with datasets containing 'amount' column.
    """

    # If 'amount' column missing, return empty with error message
    if "amount" not in df.columns:
        return pd.DataFrame({"error": ["Missing 'amount' column in dataset"]})

    anomalies = []

    mean_amount = df["amount"].mean()
    std_amount = df["amount"].std()

    for idx, row in df.iterrows():
        # flag trade if amount is abnormally high
        if row["amount"] > mean_amount * 5:
            anomalies.append(idx)
        # flag trade if amount deviates drastically
        elif abs(row["amount"] - mean_amount) > 3 * std_amount:
            anomalies.append(idx)

    return df.loc[anomalies]


# -------------------- Fraud Detection --------------------

def detect_fraudulent_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects potentially fraudulent transactions based on simple rules.
    Adds an 'is_fraud' column to the dataframe (1 = fraud, 0 = legit).
    """
    df = df.copy()
    df["is_fraud"] = 0  # default legit

    # Rule 1: Very large transaction amount
    if "amount" in df.columns:
        threshold = df["amount"].mean() + 3 * df["amount"].std()
        df.loc[df["amount"] > threshold, "is_fraud"] = 1

    # Rule 2: Suspicious locations (example: offshore accounts)
    if "location" in df.columns:
        suspicious_locations = ["Offshore", "Unknown", "Cayman Islands"]
        df.loc[df["location"].isin(suspicious_locations), "is_fraud"] = 1

    # Rule 3: Unusual trading times (midnight trades flagged)
    if "time" in df.columns:
        df.loc[df["time"].str.contains("00:"), "is_fraud"] = 1

    return df


# -------------------- Phishing Detection --------------------

def detect_phishing(messages: pd.Series) -> pd.DataFrame:
    """
    Detect phishing attempts in messages or URLs.
    Returns a DataFrame with phishing flag and reason.
    """
    results = []

    suspicious_keywords = ["verify your account", "urgent", "password", "click here", "bank", "lottery", "reset"]
    suspicious_domains = ["bit.ly", "tinyurl", "freegift", "login-now"]

    for idx, msg in messages.items():
        flagged = False
        reason = []

        # Check for suspicious keywords
        for kw in suspicious_keywords:
            if re.search(kw, msg, re.IGNORECASE):
                flagged = True
                reason.append(f"Keyword: {kw}")

        # Check for suspicious domains
        for domain in suspicious_domains:
            if domain in msg.lower():
                flagged = True
                reason.append(f"Suspicious domain: {domain}")

        results.append({
            "message": msg,
            "is_phishing": 1 if flagged else 0,
            "reason": ", ".join(reason) if flagged else "Legit"
        })

    return pd.DataFrame(results)
