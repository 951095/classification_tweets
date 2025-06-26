import pytest
from src.preprocessing import preprocess_text, tokenize, clean_text

def test_preprocess_text_empty():
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text(123) == ""

def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []
    assert tokenize(123) == []

def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_preprocess_text_short():
    # Mots trop courts supprimés
    assert preprocess_text("a b c") == ""
    assert preprocess_text("ok wow") == "wow"

def test_preprocess_text_with_url():
    # 'this' est un stopword, donc supprimé
    assert preprocess_text("Check this http://example.com") == "check"

def test_preprocess_text_with_punctuation():
    assert preprocess_text("Wow!!! Amazing, really.") == "wow amazing really"

def test_preprocess_text_with_custom_stopwords():
    # "omg" est dans CUSTOM_STOPWORDS
    assert "omg" not in preprocess_text("omg this is fast")

@pytest.mark.parametrize("input_text,expected", [
    ("Flood in NYC!", "flood nyc"),
    ("OMG fire!!!", "fire"),
    ("", ""),
    (None, ""),
    ("a b c", ""),
    ("This is fast", ""),  # tous les mots sont des stopwords
    ("Check www.site.com now", "check"),  # 'now' est un stopword
])
def test_preprocess_text_param(input_text, expected):
    assert preprocess_text(input_text) == expected

def test_preprocess_text_no_exception_on_weird_input():
    class Weird:
        def __str__(self):
            return "strange object"
    try:
        preprocess_text(Weird())
    except Exception:
        pytest.fail("preprocess_text doit gérer les objets exotiques sans lever d'exception")

def test_tokenize_behavior():
    # Les stopwords et mots courts sont supprimés
    tokens = tokenize("This is a test omg fast")
    assert "omg" not in tokens  # custom stopword
    assert "fast" not in tokens  # custom stopword
    assert all(len(t) >= 3 for t in tokens)
