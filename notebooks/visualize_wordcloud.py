
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import pandas as pd
import os
from src.preprocessing import preprocess_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger les données
csv_path = os.path.join(os.path.dirname(__file__), '../data/tweets.csv')
df = pd.read_csv(csv_path)

# Appliquer le prétraitement et concaténer tous les tokens
texts = df['text'].dropna().apply(preprocess_text)
all_text = ' '.join(texts)

# Générer le WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Afficher
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud des tokens les plus fréquents')
plt.show()
