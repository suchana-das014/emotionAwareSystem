def compute_stress_index(text=None, toxicity=None, emotion=None):
    """
    Compute a simple stress index (0 or 1).

    Priority:
    - If `toxicity` is numeric, use it (int).
    - If `toxicity` is a string like "Stress" or "Toxic", map to 1.
    - Otherwise, fall back to lightweight heuristics on the message text and emotion:
      look for urgent/escalation/deadline language or negative emotions.
    """
    # Use explicit numeric toxicity when available
    if isinstance(toxicity, (int, float)):
        return int(toxicity)

    # Handle string labels from the toxicity model
    if isinstance(toxicity, str):
        if toxicity.lower() in {"stress", "toxic", "severe"}:
            return 1
        try:
            return int(float(toxicity))
        except Exception:
            return 0

    # Fallback heuristics using message text/emotion
    if text:
        t = text.lower()
        # urgent language that often causes stress
        urgent_keywords = ["urgent", "escalat", "deadline", "submit by", "or else", "affect your", "impact your", "final report", "immediately", "asap", "by 9 am"]
        if any(k in t for k in urgent_keywords):
            return 1

    # Emotion-based heuristic
    if emotion:
        if isinstance(emotion, str) and emotion.lower() in {"anger", "fear", "sadness", "annoyance", "annoyed"}:
            return 1

    return 0
