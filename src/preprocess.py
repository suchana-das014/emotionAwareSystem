import re
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)
