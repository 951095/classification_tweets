

# === Import des librairies nécessaires ===
import re  # Pour les expressions régulières (nettoyage de texte)
import nltk  # Pour les outils NLP (Natural Language Processing)
from nltk.corpus import stopwords  # Liste des mots vides (stopwords)
from nltk.stem import PorterStemmer  # Pour le stemming (réduction des mots à leur racine)
from nltk.tokenize import word_tokenize  # Pour découper le texte en tokens (mots)
import string  # Pour la gestion de la ponctuation


# Téléchargement robuste des ressources NLTK nécessaires (stopwords et tokenizers)
# Vérifie si les ressources sont déjà présentes, sinon les télécharge automatiquement
try:
    nltk.data.find('corpora/stopwords')  # Vérifie la présence des stopwords
except LookupError:
    nltk.download('stopwords')  # Télécharge si absent
try:
    nltk.data.find('tokenizers/punkt')  # Vérifie la présence du tokenizer
except LookupError:
    nltk.download('punkt')  # Télécharge si absent


# Chargement des stopwords anglais de NLTK
from nltk.corpus import stopwords as nltk_stopwords
STOPWORDS = set(nltk_stopwords.words("english"))  # Liste de base des stopwords anglais
# Ajout de stopwords personnalisés si besoin (ex : mots fréquents non pertinents pour votre cas)
CUSTOM_STOPWORDS = set(["fast", "omg", "etc"])  # Ajoutez ici d'autres mots à filtrer
STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)  # Fusionne les deux listes


# Initialisation du stemmer (pour le stemming des mots)
stemmer = PorterStemmer()  # Permet de ramener les mots à leur racine (non utilisé ici mais prêt pour extension)


def clean_text(text: str) -> str:
    """
    Nettoie un tweet :
    - Met en minuscule
    - Supprime les URLs
    - Supprime la ponctuation et les chiffres
    - Supprime les mots de moins de 3 lettres
    """
    if not isinstance(text, str):  # Vérifie que l'entrée est bien une chaîne de caractères
        return ""  # Si ce n'est pas le cas, retourne une chaîne vide
    text = text.lower()  # Met tout en minuscule pour uniformiser
    text = re.sub(r"http\S+|www\S+", "", text)  # Supprime les URLs (liens web)
    text = re.sub(r"[^a-z\s]", "", text)  # Supprime tout sauf les lettres et les espaces (enlève chiffres et ponctuation)
    words = text.split()  # Découpe le texte en mots (séparés par des espaces)
    words = [w for w in words if len(w) >= 3]  # Garde uniquement les mots de 3 lettres ou plus (filtre les mots trop courts)
    return " ".join(words)  # Recompose le texte nettoyé en une seule chaîne


def tokenize(text):
    """
    Tokenise le texte :
    - Met en minuscule
    - Supprime les URLs
    - Supprime la ponctuation
    - Découpe en tokens (mots)
    - Supprime les stopwords et mots courts (<3)
    """
    if not isinstance(text, str):  # Vérifie que l'entrée est une chaîne
        return []  # Si ce n'est pas le cas, retourne une liste vide
    text = text.lower()  # Met en minuscule
    text = re.sub(r"http\S+|www.\S+", "", text)  # Supprime les URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Supprime la ponctuation
    tokens = word_tokenize(text)  # Découpe le texte en tokens (mots)
    # Filtre : garde les tokens de 3 lettres ou plus et qui ne sont pas des stopwords
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def preprocess_text(text: str) -> str:
    """
    Pipeline complet de prétraitement :
    - Nettoie le texte (voir clean_text)
    - Tokenise le texte (voir tokenize)
    - Reconstitue le texte nettoyé (tokens séparés par des espaces)
    """
    if not isinstance(text, str):  # Vérifie que l'entrée est une chaîne
        return ""  # Si ce n'est pas le cas, retourne une chaîne vide
    cleaned = clean_text(text)  # Étape 1 : nettoyage du texte brut
    tokens = tokenize(cleaned)  # Étape 2 : tokenisation du texte nettoyé
    return " ".join(tokens)  # Étape 3 : recompose le texte final à partir des tokens
