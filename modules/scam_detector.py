import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("data/phishing_samples.csv")

# Train a simple text classifier
X, y = data["message"], data["label"]
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression())
])
pipeline.fit(X, y)

# Save model for reusability
joblib.dump(pipeline, "modules/scam_model.pkl")

def detect_scam(text: str):
    if not text.strip():
        return {"label": "safe", "score": 0.0}

    model = joblib.load("modules/scam_model.pkl")
    prob = model.predict_proba([text])[0]
    label = "scam" if prob[1] > 0.5 else "safe"
    return {"label": label, "score": float(max(prob))}
