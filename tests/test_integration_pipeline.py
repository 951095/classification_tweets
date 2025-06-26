import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.preprocessing import preprocess_text
from src.modeling import train_and_evaluate

def test_full_pipeline():
    df = pd.DataFrame({
        "text": [
            "Flood in Paris! http://flood.com",
            "No disaster here.",
            "OMG fire in the city!",
            "Just a sunny day"
        ],
        "target": [1, 0, 1, 0]
    })
    df["text"] = df["text"].apply(preprocess_text)
    pipeline, metrics = train_and_evaluate(df)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
