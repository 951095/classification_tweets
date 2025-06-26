import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import os
import pytest
from src.preprocessing import preprocess_text
from src.modeling import train_and_evaluate


def test_pipeline_on_real_csv():
    csv_path = os.path.join(os.path.dirname(__file__), '../data/tweets.csv')
    if not os.path.exists(csv_path):
        pytest.skip("tweets.csv non trouvé")
    df = pd.read_csv(csv_path).head(20)  # Prend un petit échantillon
    df['text'] = df['text'].apply(preprocess_text)
    pipeline, metrics = train_and_evaluate(df)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
