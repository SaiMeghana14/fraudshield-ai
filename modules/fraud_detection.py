import re
import pandas as pd
import streamlit as st   # needed for st.warning

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal anomaly detector that supports different dataset schemas.
    - Looks for numeric columns in order: 'amount' -> 'quantity' -> 'price'
    - Flags rows where value > mean * 5 OR abs(value-mean) > 3*std
    - Also includes rows with is_fraud == 1 (if present)
    - Returns flagged rows (duplicates removed). Never raises KeyError.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    # normalize column name lookup: map lowercase -> actual name
    col_map = {c.strip().lower(): c for c in df.columns}

    numeric_cols = []
    for key in ["amount", "quantity", "price"]:
        if key in col_map:
            numeric_cols.append(col_map[key])

    if not numeric_cols and "is_fraud" in df.columns:
        try:
            return df[df["is_fraud"].astype(int) == 1].copy()
        except Exception:
            return pd.DataFrame(columns=df.columns)

    if not numeric_cols:
        st.warning("No numeric columns found for anomaly detection (expected 'amount', 'Quantity', or 'Price').")
        return pd.DataFrame(columns=df.columns)

    # convert numeric columns safely
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = pd.Series(False, index=df.index)

    # anomaly detection per numeric column
    for c in numeric_cols:
        s = df[c]
        mean, std = s.mean(skipna=True), s.std(skipna=True)

        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue

        mask |= s > mean * 5
        mask |= (s - mean).abs() > 3 * std

    if "is_fraud" in df.columns:
        try:
            mask |= df["is_fraud"].astype(int) == 1
        except Exception:
            pass

    anomalies = df[mask].drop_duplicates().copy()
    return anomalies


# -------------------- Fraud Detection --------------------

def detect_fraudulent_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects potentially fraudulent transactions based on simple rules.
    Adds an 'is_fraud' column to the dataframe (1 = fraud, 0 = legit).
    """
    df = df.copy()
    df["is_fraud"] = 0

    col_map = {c.strip().lower(): c for c in df.columns}

    # Rule 1: Very large transaction amount
    if "amount" in col_map:
        c = col_map["amount"]
        threshold = df[c].mean() + 3 * df[c].std()
        df.loc[df[c] > threshold, "is_fraud"] = 1
    elif "quantity" in col_map:
        c = col_map["quantity"]
        threshold = df[c].mean() + 3 * df[c].std()
        df.loc[df[c] > threshold, "is_fraud"] = 1
    elif "price" in col_map:
        c = col_map["price"]
        threshold = df[c].mean() + 3 * df[c].std()
        df.loc[df[c] > threshold, "is_fraud"] = 1

    # Rule 2: Suspicious locations
    if "location" in col_map:
        suspicious_locations = ["Offshore", "Unknown", "Cayman Islands"]
        df.loc[df[col_map["location"]].isin(suspicious_locations), "is_fraud"] = 1

    # Rule 3: Unusual trading times
    if "time" in col_map:
        try:
            df.loc[df[col_map["time"]].astype(str).str.contains("00:"), "is_fraud"] = 1
        except Exception:
            pass

    return df


# -------------------- Phishing Detection --------------------

def detect_phishing(messages: pd.Series) -> pd.DataFrame:
    results = []
    suspicious_keywords = ["verify your account", "urgent", "password", "click here", "bank", "lottery", "reset"]
    suspicious_domains = ["bit.ly", "tinyurl", "freegift", "login-now"]

    for idx, msg in messages.items():
        flagged = False
        reason = []

        for kw in suspicious_keywords:
            if re.search(kw, msg, re.IGNORECASE):
                flagged = True
                reason.append(f"Keyword: {kw}")

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
