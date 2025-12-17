import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.spam_model import predict_spam

examples = [
    "Congratulations! You've won $1,000,000! Claim now",
    "free money click here",
    "Hi, how are you?"
]

for e in examples:
    print(repr(e), '->', predict_spam(e))
