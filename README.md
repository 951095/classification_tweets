
# Tweet Project – Classification automatique de Tweets (TP Test Unitaire)

## 🚀 Objectif
Construire un pipeline complet de Data Science pour la classification automatique de tweets (catastrophes vs normaux), en respectant les bonnes pratiques : nettoyage, modélisation, tests unitaires, reproductibilité et dockerisation.

## 🗂️ Structure du projet

```
tweet_project/
├── data/                # Données brutes (ex : tweets.csv)
├── notebooks/           # Analyses exploratoires et modélisation
│   ├── 01_EDA.ipynb     # Analyse exploratoire des données
│   └── 02_Preprocessing_Modeling.ipynb  # Prétraitement & Modélisation
├── src/                 # Code source (prétraitement, pipeline sklearn)
│   ├── preprocessing.py # Fonctions de nettoyage et NLP
│   └── modeling.py      # Pipeline sklearn et évaluation
├── tests/               # Tests unitaires (Pytest)
├── main.py              # Script principal (exécution du pipeline complet)
├── requirements.txt     # Dépendances Python
├── Dockerfile           # Conteneurisation du projet
└── README.md            # Ce fichier
```

## 📦 Installation & Prérequis
1. **Cloner le dépôt**
   ```bash
   git clone <url-du-repo>
   cd tweet_project
   ```
2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
3. **Télécharger les ressources NLTK** (à faire une seule fois)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```
4. **(Optionnel) Utiliser Docker**
   ```bash
   docker build -t tweet_project .
   docker run -it tweet_project
   ```

## 🔎 Fonctionnalités principales
- **Exploratory Data Analysis (EDA)** :
  - Statistiques descriptives, analyse des classes, valeurs manquantes, distribution des longueurs de texte, visualisations (WordCloud, histogrammes…)
- **Prétraitement du texte** :
  - Nettoyage (minuscule, suppression URLs, ponctuation, chiffres, mots courts)
  - Suppression des stopwords (anglais + personnalisés)
  - Tokenisation, pipeline complet prêt pour la modélisation
- **Modélisation & Pipeline** :
  - Pipeline scikit-learn (TfidfVectorizer + LogisticRegression)
  - Séparation train/test, évaluation (accuracy, precision, recall, f1-score)
  - Prise en charge des cas particuliers (texte vide, classes déséquilibrées…)
- **Tests unitaires** :
  - Couverture complète du prétraitement, robustesse du pipeline, intégration sur données réelles
- **Reproductibilité & Qualité** :
  - Code modulaire, tests automatisés, dockerisation

## 🧪 Lancer les tests
Dans le dossier du projet :
```bash
pytest
```

## ▶️ Exécuter le pipeline principal
```bash
python main.py
```

## 📓 Utilisation des notebooks
Ouvrir les notebooks dans VS Code ou Jupyter pour explorer :
- `notebooks/01_EDA.ipynb` : analyse exploratoire
- `notebooks/02_Preprocessing_Modeling.ipynb` : pipeline de prétraitement et modélisation

## 📝 Exemple d’utilisation du pipeline (dans main.py)
```python
from src.modeling import train_and_evaluate
import pandas as pd
df = pd.read_csv("data/tweets.csv")
pipeline, metrics = train_and_evaluate(df)
print(metrics)
```

## 🛠️ Personnalisation
- Ajouter vos propres stopwords dans `src/preprocessing.py` (variable `CUSTOM_STOPWORDS`)
- Modifier le modèle ou la vectorisation dans `src/modeling.py`
- Ajouter d’autres tests dans le dossier `tests/`

## 📚 Ressources utiles
- [Documentation scikit-learn](https://scikit-learn.org/stable/)
- [NLTK](https://www.nltk.org/)
- [Pytest](https://docs.pytest.org/)

## 👨‍💻 Auteurs
Projet réalisé dans le cadre du TP de Test Unitaire (M1 Data Science)
