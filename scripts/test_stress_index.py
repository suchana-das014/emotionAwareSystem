from src.toxicity_model import predict_toxicity
from src.emotion_model import predict_emotion
from src.stress_index import compute_stress_index

samples=[
"We've noticed repeated mistakes in your work; let's discuss this tomorrow",
"This is urgent — submit the final report by 9 AM or it will be escalated.",
"If this isn't fixed by tomorrow, it could affect your position here."
]

for s in samples:
    tox = predict_toxicity(s)
    emo = predict_emotion(s)
    stress = compute_stress_index(text=s, toxicity=tox, emotion=emo)
    print(s)
    print('tox->', tox, 'emo->', emo, 'stress->', stress)
