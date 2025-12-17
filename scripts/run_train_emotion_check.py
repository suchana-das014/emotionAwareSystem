import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocess import clean_text

csv_path = os.path.join(project_root, 'datasets', 'emotions', 'go_emotions_dataset.csv')
print('Loading', csv_path)
df = pd.read_csv(csv_path)
print('Rows:', len(df), 'Columns:', list(df.columns)[:10])

# basic preprocessing
print('Cleaning text (first 2000 rows)...')
limit = min(2000, len(df))
df_small = df.iloc[:limit].copy()
df_small['cleaned'] = df_small['text'].apply(clean_text)

exclude_cols = {'id', 'text', 'example_very_unclear'}
emotion_cols = [c for c in df_small.columns if c not in exclude_cols]
print('Detected emotion columns count:', len(emotion_cols))

# create label
if 'emotion' not in df_small.columns:
    df_small['emotion'] = df_small[emotion_cols].idxmax(axis=1)
    no_label = df_small[emotion_cols].sum(axis=1) == 0
    df_small.loc[no_label, 'emotion'] = 'neutral'

print('Label distribution (top 10):')
print(df_small['emotion'].value_counts().head(10))

# vectorize and train on small sample
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_small['cleaned'])
y = df_small['emotion']
print('Feature shape', X.shape)

model = LogisticRegression(max_iter=1000)
print('Training model on', len(y), 'samples...')
model.fit(X, y)
print('Training complete.')

# save small models to models/check_
out_dir = os.path.join(project_root, 'models')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
joblib.dump(model, os.path.join(out_dir, 'emotion_model_check.pkl'))
joblib.dump(tfidf, os.path.join(out_dir, 'emotion_vectorizer_check.pkl'))
print('Saved check models to', out_dir)
