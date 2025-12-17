import pandas as pd
import os

csv_path = r"c:\Users\SUCHANA\Desktop\emotion_aware_nlp\datasets\toxic\archive (5)\train.csv"
df = pd.read_csv(csv_path, nrows=5)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
