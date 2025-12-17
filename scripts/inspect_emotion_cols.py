import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'emotions', 'go_emotions_dataset.csv')
print('Loading', csv_path)
df = pd.read_csv(csv_path)
print('\nColumns:', df.columns.tolist())

exclude_cols = {'id', 'text', 'example_very_unclear'}
emotion_cols = [c for c in df.columns if c not in exclude_cols]
print('\nDetected emotion columns (count {}):'.format(len(emotion_cols)))
print(emotion_cols)

print('\nDtypes for emotion columns:')
print(df[emotion_cols].dtypes)

print('\nFirst 5 rows of emotion columns:')
print(df[emotion_cols].head(5))

print('\nAttempting idxmax...')
try:
    labels = df[emotion_cols].idxmax(axis=1)
    print('idxmax succeeded. Sample labels:')
    print(labels.head(10))
except Exception as e:
    import traceback
    print('idxmax raised exception:')
    traceback.print_exc()
