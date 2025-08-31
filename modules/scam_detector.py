import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_phishing_data(path="data/phishing_samples.csv"):
    df = pd.read_csv(path)

    # Normalize column names just in case
    df.columns = df.columns.str.strip().str.lower()

    # Ensure correct columns exist
    if "message" not in df.columns or "label" not in df.columns:
        raise ValueError("phishing_samples.csv must contain 'message' and 'label' columns.")

    return df

def train_scam_model(path="data/phishing_samples.csv"):
    df = load_phishing_data(path)

    X = df["message"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer

def predict_scam(text, model, vectorizer):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]
