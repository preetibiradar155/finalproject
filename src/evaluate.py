import os
import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preparation import load_data, prepare_features

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')
MET_DIR = os.path.join(ROOT, 'reports', 'metrics')
FIG_DIR = os.path.join(ROOT, 'reports', 'figures')

os.makedirs(MET_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def evaluate():
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipe = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    preds = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    res = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'n_test': int(len(y_test)),
        'evaluated_at': datetime.utcnow().isoformat()
    }
    with open(os.path.join(MET_DIR, 'evaluation_test.json'), 'w') as f:
        json.dump(res, f, indent=2)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual G3')
    plt.ylabel('Predicted G3')
    plt.title('Actual vs Predicted (Test)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'actual_vs_predicted.png'))
    plt.close()
    
    plt.figure(figsize=(6, 4))
    residuals = y_test - preds
    plt.hist(residuals, bins=15)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'residuals.png'))
    plt.close()
    
    print(json.dumps(res, indent=2))
    return res


if __name__ == '__main__':
    evaluate()
