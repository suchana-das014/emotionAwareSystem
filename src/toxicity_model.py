import re
import joblib
from src.preprocess import clean_text

# Load trained toxicity model and vectorizer
toxicity_model = joblib.load("models/toxicity_model.pkl")
toxicity_vectorizer = joblib.load("models/toxicity_vectorizer.pkl")

def _rule_based_toxicity(text: str) -> bool:
    """
    Simple rule-based detection for obvious toxic/hateful content.
    Returns True if message appears toxic.
    """
    t = text.lower()
    
    # Hateful/dehumanizing language
    hateful_patterns = [
        r"deadly sin",  # religious intolerance
        r"satanistic",  # religious intolerance
        r"go to hell",
        r"you're a.*gay",  # direct slurs/insults targeting sexual orientation
        r"gay.*sin",  # religious condemnation
        r"god damn",
        r"fuck.*you",
        r"you.*idiot",
        r"you.*stupid",
        r"you.*useless",
        r"kill.*yourself",
        r"kys\b",
        r"deserve.*die",
        r"should.*die",
    ]
    
    for pattern in hateful_patterns:
        if re.search(pattern, t, re.IGNORECASE):
            return True
    
    return False

def predict_toxicity(text: str) -> int:
    """
    Predict toxicity of a message (binary: 0=not toxic, 1=toxic).
    Uses rule-based heuristics first, then falls back to the trained model.
    """
    # Check rule-based heuristics first (catches obvious hateful content)
    if _rule_based_toxicity(text):
        return 1
    
    # Fall back to trained model
    cleaned_text = clean_text(text)
    vector = toxicity_vectorizer.transform([cleaned_text])
    prediction = toxicity_model.predict(vector)[0]
    
    # Normalize to 0/1
    if isinstance(prediction, (int, float)):
        return int(prediction)
    pred_str = str(prediction).lower()
    if pred_str in {"1", "true", "toxic"}:
        return 1
    return 0
