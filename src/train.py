import os
import json
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preparation import load_data, prepare_features

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, 'models')
MET_DIR = os.path.join(ROOT, 'reports', 'metrics')
ART_DIR = os.path.join(ROOT, 'artifacts')
MLRUNS_DIR = os.path.join(ROOT, 'mlruns')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)


def build_pipeline(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features


def train():
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    mlflow.set_tracking_uri(f'file:{MLRUNS_DIR}')
    mlflow.set_experiment('student_marks_prediction')
    
    df = load_data()
    X, y = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor, num_feats, cat_feats = build_pipeline(X_train)
    
    candidates = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    results = {}
    best_name = None
    best_score = -1e18
    best_pipe = None
    
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mlflow.log_param('model', name)
            mlflow.log_metric('mae', mae)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('r2', r2)
            mlflow.log_param('num_features', len(num_feats))
            mlflow.log_param('cat_features', len(cat_feats))
            try:
                mlflow.sklearn.log_model(pipe, 'model')
            except Exception as e:
                print('mlflow log_model skipped:', e)
            results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
            if r2 > best_score:
                best_score = r2
                best_name = name
                best_pipe = pipe
    
    # save best
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(best_pipe, model_path)
    artifact_path = os.path.join(ART_DIR, 'best_model.joblib')
    joblib.dump(best_pipe, artifact_path)
    
    summary = {
        'best_model': best_name,
        'best_r2': float(best_score),
        'results': results,
        'trained_at': datetime.utcnow().isoformat(),
        'test_size': 0.2,
        'random_state': 42
    }
    with open(os.path.join(MET_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    train()
