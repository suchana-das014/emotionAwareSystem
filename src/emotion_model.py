import joblib
from src.preprocess import clean_text

# Load trained emotion model and vectorizer
emotion_model = joblib.load("models/emotion_model.pkl")
emotion_vectorizer = joblib.load("models/emotion_vectorizer.pkl")

def predict_emotion(text: str, toxicity: int = None) -> str:
    """
    Predict emotion of a non-spam message.
    Possible outputs:
    joy, sadness, anger, fear, neutral, and others from go_emotions
    
    Args:
        text: The message text to analyze
        toxicity: Optional toxicity flag (0 or 1). If toxicity=1, return "anger"
                 since toxic messages almost always convey anger/hostility.
    """
    # If message is toxic, override with "anger" for accuracy
    if toxicity == 1:
        return "anger"
    
    cleaned_text = clean_text(text)
    vector = emotion_vectorizer.transform([cleaned_text])
    prediction = emotion_model.predict(vector)[0]
    return prediction
