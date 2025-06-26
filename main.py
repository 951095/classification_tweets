import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_text
from src.model import train_and_evaluate  # ou src.modeling selon ta structure

# Charger les données
df = pd.read_csv('data/tweets.csv')

# Appliquer le preprocessing sur toute la colonne 'text' avant l'entraînement
df['text'] = df['text'].apply(preprocess_text)

# Entraîner et évaluer le modèle sur le texte pré-traité
model, vectorizer, metrics = train_and_evaluate(df)

print(metrics)
