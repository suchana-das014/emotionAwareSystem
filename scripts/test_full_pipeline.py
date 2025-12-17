import sys
sys.path.insert(0, '.')

from src.emotion_model import predict_emotion
from src.toxicity_model import predict_toxicity
from src.stress_index import compute_stress_index

messages = [
    "feeling ill",
    "If you doubt words of the Bible, that homosexuality is a deadly sin, make a pentagram tatoo on your forehead go to the satanistic masses with your gay pals!",
    "First and last warning, you fucking gay - I won't appreciate if any more nazi shwain would write in my page!"
]

for msg in messages:
    print(f"Message: {msg[:60]}...")
    tox = predict_toxicity(msg)
    emo = predict_emotion(msg, toxicity=tox)
    stress = compute_stress_index(text=msg, toxicity=tox, emotion=emo)
    print(f"  Toxicity: {tox}, Emotion: {emo}, Stress: {stress}")
    print()
