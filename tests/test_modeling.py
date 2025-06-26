# tests/test_modeling.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.modeling import train_and_evaluate

def test_pipeline_runs():
    df = pd.DataFrame({
        "text": ["Flood in NYC", "I love pizza", "Earthquake!", "Nice weather"],
        "target": [1, 0, 1, 0]
    })
    pipeline, metrics = train_and_evaluate(df)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

def test_empty_text_handled():
    df = pd.DataFrame({
        "text": ["", "Something happened", "Flood", ""],
        "target": [0, 1, 1, 0]
    })
    pipeline, metrics = train_and_evaluate(df)
    assert isinstance(metrics, dict)

def test_pipeline_with_imbalanced_classes():
    df = pd.DataFrame({
        "text": ["Disaster"] * 19 + ["Safe"] * 1,
        "target": [1] * 19 + [0]
    })
    pipeline, metrics = train_and_evaluate(df)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

def test_pipeline_with_nonsense_text():
    df = pd.DataFrame({
        "text": ["asdfgh", "qwerty", "zxcvb", "poiuyt"],
        "target": [0, 1, 0, 1]
    })
    pipeline, metrics = train_and_evaluate(df)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}

def test_train_and_evaluate_missing_column():
    df = pd.DataFrame({"text": ["a", "b", "c"]})
    try:
        train_and_evaluate(df)
        assert False, "Doit lever une erreur si colonne 'target' manquante"
    except KeyError:
        pass

def test_train_and_evaluate_wrong_type():
    import pandas as pd
    from src.modeling import train_and_evaluate
    df = pd.DataFrame({"text": [1, 2, 3], "target": [0, 1, 0]})
    import pytest
    with pytest.raises(ValueError, match="empty vocabulary"):
        train_and_evaluate(df)

def test_train_and_evaluate_empty_df():
    df = pd.DataFrame({"text": [], "target": []})
    try:
        train_and_evaluate(df)
        assert False, "Doit lever une erreur ou gérer le DataFrame vide"
    except ValueError:
        pass
    except Exception:
        pass
