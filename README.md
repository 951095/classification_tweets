# Tweet Project – TP Test Unitaire

## Objectif
Ce projet vise à appliquer des techniques de prétraitement, de modélisation et de tests unitaires sur un jeu de données de tweets pour la détection d'événements.

## Structure du projet
- `src/` : code source (prétraitement, modélisation)
- `tests/` : tests unitaires (Pytest)
- `data/` : données (ex : `tweets.csv`)
- `notebooks/` : analyses exploratoires et modélisation
- `main.py` : script principal

## Installation
1. Cloner le dépôt
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Télécharger les ressources NLTK (à faire une seule fois) :
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Lancer les tests
Dans le dossier `tweet_project/` :
```bash
pytest
```

## Lancer le script principal
```bash
python main.py
```

## Notes
- Le prétraitement gère les cas de texte vide ou non textuel.
- Les tests vérifient la robustesse des fonctions et la qualité des données.
# classification_tweets
