import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

# Téléchargement robuste des ressources NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords as nltk_stopwords

STOPWORDS = set(nltk_stopwords.words("english"))
CUSTOM_STOPWORDS = set(["fast", "omg", "etc"])  # ajoute les mots que tu veux filtrer en plus
STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Nettoie un tweet : minuscule, suppression URLs, ponctuations, chiffres, mots courts (< 3 lettres)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"[^a-z\s]", "", text)        # Ponctuation et chiffres
    words = text.split()
    words = [w for w in words if len(w) >= 3]
    return " ".join(words)

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens

def preprocess_text(text: str) -> str:
    """
    Pipeline complet : nettoyage + tokenisation + reconstitution du texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return " ".join(tokens)
