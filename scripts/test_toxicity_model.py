import sys
sys.path.insert(0, '.')

from src.toxicity_model import predict_toxicity

hateful_msg = "If you doubt words of the Bible, that homosexuality is a deadly sin, make a pentagram tatoo on your forehead go to the satanistic masses with your gay pals!"
mild_msg = "feeling ill"

print("Testing hateful message:")
result = predict_toxicity(hateful_msg)
print(f"Toxicity: {result}")
print()

print("Testing mild message:")
result = predict_toxicity(mild_msg)
print(f"Toxicity: {result}")
