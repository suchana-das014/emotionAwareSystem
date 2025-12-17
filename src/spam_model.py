import re
import joblib
from src.preprocess import clean_text

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/spam_vectorizer.pkl")


def _rule_based_spam(text: str) -> bool:
    # simple heuristics for obvious spam: URLs, common spam words, long digit sequences, or standalone emails
    if re.search(r"http[s]?://|www\\.|click here|free|win|prize|claim|buy now|subscribe", text, re.I):
        return True
    if re.search(r"\\d{7,}", text):
        return True
    # (removed standalone-email heuristic per user request)
    return False


def predict_spam(text):
    """Return 1 for spam, 0 for ham.

    Uses a small rule-based fallback for obvious spam, then the trained model.
    """
    raw = text
    if _rule_based_spam(raw):
        return 1

    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    # Normalize model output to 0/1
    if isinstance(pred, (int, float)):
        return int(pred)
    pred_str = str(pred).lower()
    if pred_str in {"spam", "1", "true", "t"}:
        return 1
    return 0
