import re
import pandas as pd

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal anomaly detector that supports different dataset schemas.
    - Looks for numeric columns in order: 'amount' -> 'Quantity' -> 'Price'
    - Flags rows where value > mean * 5 OR abs(value-mean) > 3*std
    - Also includes rows with is_fraud == 1 (if present)
    - Returns flagged rows (duplicates removed). Never raises KeyError.
    """
    # handle empty input
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    # normalize column name lookup: map lowercase -> actual name
    col_map = {c.strip().lower(): c for c in df.columns}

    # choose numeric columns to examine (prefer amount, then quantity, then price)
    numeric_cols = []
    if 'amount' in col_map:
        numeric_cols.append(col_map['amount'])
    if 'quantity' in col_map and col_map['quantity'] not in numeric_cols:
        numeric_cols.append(col_map['quantity'])
    if 'price' in col_map and col_map['price'] not in numeric_cols:
        numeric_cols.append(col_map['price'])

    # if no numeric column found, but `is_fraud` exists, use that to return flagged rows
    if not numeric_cols and 'is_fraud' in df.columns:
        try:
            return df[df['is_fraud'].astype(int) == 1].copy()
        except Exception:
            return pd.DataFrame(columns=df.columns)

    if not numeric_cols:
        # no useful numeric columns — warn and return empty set
        st.warning("No numeric columns found for anomaly detection (expected 'amount' or 'Quantity' or 'Price').")
        return pd.DataFrame(columns=df.columns)

    # coerce numeric columns to numbers safely
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    mask = pd.Series(False, index=df.index)

    # flag anomalies per numeric column
    for c in numeric_cols:
        s = df[c]
        mean = s.mean(skipna=True)
        std = s.std(skipna=True)

        # skip if we can't compute stats
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue

        mask |= s > mean * 5
        mask |= (s - mean).abs() > 3 * std

    # include explicit is_fraud column if present
    if 'is_fraud' in df.columns:
        try:
            mask |= df['is_fraud'].astype(int) == 1
        except Exception:
            # if conversion fails, ignore it
            pass

    anomalies = df[mask].copy()
    anomalies = anomalies.drop_duplicates()

    return anomalies


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
