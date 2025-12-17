from src.toxicity_model import predict_toxicity
samples=[
"We've noticed repeated mistakes in your work; let's discuss this tomorrow",
"This is urgent — submit the final report by 9 AM or it will be escalated.",
"If this isn't fixed by tomorrow, it could affect your position here."
]
for s in samples:
    print(s)
    print('->', predict_toxicity(s))
