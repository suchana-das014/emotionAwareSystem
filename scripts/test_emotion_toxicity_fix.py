import sys
sys.path.insert(0, '.')

from src.emotion_model import predict_emotion
from src.toxicity_model import predict_toxicity

toxic_msg = "First and last warning, you fucking gay - I won't appreciate if any more nazi shwain would write in my page! I don't wish to talk to you anymore!"
normal_msg = "I love this amazing day!"

print("Testing toxic message:")
tox = predict_toxicity(toxic_msg)
emo = predict_emotion(toxic_msg, toxicity=tox)
print(f"Toxicity: {tox}, Emotion: {emo}")
print()

print("Testing normal message:")
tox = predict_toxicity(normal_msg)
emo = predict_emotion(normal_msg, toxicity=tox)
print(f"Toxicity: {tox}, Emotion: {emo}")
