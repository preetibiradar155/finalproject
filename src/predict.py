import os
import json

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')
ART_DIR = os.path.join(ROOT, 'artifacts')


class Predictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # prefer models/, fallback to artifacts/
            cand1 = os.path.join(MODELS_DIR, 'best_model.joblib')
            cand2 = os.path.join(ART_DIR, 'best_model.joblib')
            model_path = cand1 if os.path.exists(cand1) else cand2
        self.model = joblib.load(model_path)
        self.model_path = model_path
    
    def predict(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        preds = self.model.predict(X)
        return preds


if __name__ == '__main__':
    pred = Predictor()
    # dummy
    print(pred.predict({}))
