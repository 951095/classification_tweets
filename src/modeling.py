

# === src/modeling.py ===

# Import des librairies nécessaires pour la modélisation
from sklearn.feature_extraction.text import TfidfVectorizer  # Pour transformer le texte en vecteurs numériques (TF-IDF)
from sklearn.linear_model import LogisticRegression  # Modèle de classification supervisée
from sklearn.pipeline import Pipeline  # Pour enchaîner plusieurs étapes (vectorisation + modèle)
from sklearn.model_selection import train_test_split  # Pour séparer les données en train/test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Pour évaluer les performances du modèle
import pandas as pd  # Pour manipuler les DataFrames

def create_pipeline():
    """
    Crée un pipeline scikit-learn avec deux étapes :
    1. TfidfVectorizer : transforme le texte en vecteurs numériques (pondération TF-IDF)
    2. LogisticRegression : applique un modèle de régression logistique pour la classification
    Retourne :
        pipeline (sklearn.Pipeline)
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),  # Étape 1 : vectorisation du texte
        ("clf", LogisticRegression())  # Étape 2 : classification
    ])
    return pipeline

def train_and_evaluate(df, text_column="text", target_column="target"):
    """
    Entraîne le pipeline sur les données et retourne le modèle + les métriques d'évaluation.
    Paramètres :
        df : DataFrame contenant les données
        text_column : nom de la colonne contenant le texte à vectoriser
        target_column : nom de la colonne cible (classe à prédire)
    Étapes :
        - Sépare les données en train/test (80%/20%)
        - Entraîne le pipeline sur le train
        - Prédit sur le test
        - Calcule accuracy, precision, recall, f1-score
    Retourne :
        pipeline (entraîné), metrics (dict des scores)
    """
    X = df[text_column].astype(str)  # S'assure que la colonne texte est bien en string
    y = df[target_column]  # Colonne cible (0 ou 1)

    # Séparation train/test (80% pour l'entraînement, 20% pour le test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = create_pipeline()  # Crée le pipeline (vectorisation + modèle)
    pipeline.fit(X_train, y_train)  # Entraîne le pipeline sur les données d'entraînement
    y_pred = pipeline.predict(X_test)  # Prédit les classes sur les données de test

    # Calcul des métriques d'évaluation
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),  # Taux de bonne classification
        "precision": precision_score(y_test, y_pred),  # Précision (positifs prédits corrects)
        "recall": recall_score(y_test, y_pred),  # Rappel (proportion de vrais positifs retrouvés)
        "f1_score": f1_score(y_test, y_pred),  # Score F1 (moyenne harmonique précision/rappel)
    }

    return pipeline, metrics  # Retourne le pipeline entraîné et les métriques d'évaluation
