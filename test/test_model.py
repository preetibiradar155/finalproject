import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import joblib


def test_model_artifact_exists():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, 'models', 'best_model.joblib')
    assert os.path.exists(path), f'Missing model at {path}'


def test_model_loads():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, 'models', 'best_model.joblib')
    model = joblib.load(path)
    assert model is not None
