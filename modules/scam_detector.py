from transformers import pipeline

nlp_model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

def detect_scam(message: str):
    result = nlp_model(message)[0]
    return {"label": result["label"].lower(), "score": result["score"]}
