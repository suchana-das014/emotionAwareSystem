🧠 Emotion-Aware NLP System

An Emotion-Aware Natural Language Processing (NLP) system that analyzes text to detect emotion, toxicity, spam, and compute a stress index using Machine Learning and rule-based logic.
The project demonstrates a complete end-to-end NLP pipeline with training, inference, testing, and an interactive UI.

📌 Project Overview

This project is designed to automatically analyze user-provided text and provide insights related to:

Emotional state detection

Toxic or abusive language detection

Spam message detection

Mental health stress risk assessment

The system combines classical NLP techniques, TF-IDF feature extraction, Logistic Regression models, and rule-based stress evaluation to ensure fast, interpretable, and ethical predictions.

🚀 Features

Text preprocessing and normalization

Emotion classification

Toxicity detection

Spam detection

Stress index computation

Interactive Streamlit user interface

Modular and scalable project structure

Easily extendable for future enhancements

🧩 Technology Stack

Programming Language: Python

NLP Techniques: Text Cleaning, TF-IDF

Machine Learning: Logistic Regression

Libraries:

scikit-learn

pandas

numpy

nltk

joblib

streamlit

📁 Project Structure
emotion_aware_nlp/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
│
├── datasets/              # Training datasets
│
├── models/                # Trained models and vectorizers
│
├── notebooks/             # Model training notebooks
│
├── scripts/               # Testing and validation scripts
│
└── src/                   # Core source code
    ├── preprocess.py
    ├── emotion_model.py
    ├── toxicity_model.py
    ├── spam_model.py
    ├── stress_index.py
    └── __init__.py

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/suchana-das014/emotionAwareSystem.git
cd emotion_aware_nlp

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


The application will open in your browser and allow you to input text for analysis.

🧪 Model Training

Model training is performed using Jupyter Notebooks located in the notebooks/ folder:

Emotion model training

Toxicity model training

Spam model training

Trained models and vectorizers are saved in the models/ directory and reused during inference.

🧠 Stress Index Logic

The stress index is computed using rule-based logic by combining:

Predicted emotional state

Toxicity detection result

This approach ensures:

Interpretability

Ethical transparency

Avoidance of black-box mental health decisions


👤 Collaborators

Name: Suchana Rani Das

Name: Anika Tahsin
