import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_df():
    return pd.read_csv("data/tweets.csv")

def test_dataframe_shape(sample_df):
    assert sample_df.shape[0] > 0, "Le dataset ne contient aucune ligne"
    assert "text" in sample_df.columns, "Colonne 'text' absente"
    assert "target" in sample_df.columns, "Colonne 'target' absente"

def test_no_empty_text(sample_df):
    assert sample_df['text'].fillna('').str.strip().eq('').sum() < 0.01 * len(sample_df), "Trop de textes vides"

def test_target_classes(sample_df):
    unique_targets = set(sample_df['target'].dropna().unique())
    assert unique_targets <= {0, 1}, f"Valeurs de 'target' inattendues : {unique_targets}"

def test_no_duplicate_rows(sample_df):
    duplicates = sample_df.duplicated().sum()
    assert duplicates < 0.01 * len(sample_df), f"Trop de doublons : {duplicates}"

def test_text_length_stats(sample_df):
    sample_df['text'] = sample_df['text'].fillna("")
    lengths = sample_df['text'].apply(len)
    assert lengths.min() >= 5, "Des tweets sont trop courts"
    assert lengths.mean() > 20, "Longueur moyenne anormalement basse"


from src.preprocessing import clean_text, tokenize, preprocess_text

def test_clean_text_basic():
    text = "Hello! This is a test... 1234 :)"
    result = clean_text(text)
    assert result == "hello this test", f"Résultat inattendu: {result}"

def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_tokenize_basic():
    text = "Floods are coming FAST!!!"
    tokens = tokenize(text)
    assert all(isinstance(t, str) for t in tokens)
    assert all(len(t) >= 3 for t in tokens), "Des tokens trop courts"
    assert "fast" not in tokens, "Stopword non supprimé"

def test_preprocess_pipeline():
    text = "1234 OMG!!! FLOODS! Again... www.alert.com"
    processed = preprocess_text(text)
    assert isinstance(processed, str)
    assert "flood" in processed or "floods" in processed, "Mot-clé manquant"
    assert "www" not in processed
    assert len(processed.split()) > 0
