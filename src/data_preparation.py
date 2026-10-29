import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
FIG_DIR = os.path.join(REPORT_DIR, 'figures')
MET_DIR = os.path.join(REPORT_DIR, 'metrics')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)


def load_data(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'student-mat.csv')
    df = pd.read_csv(path, sep=',')
    return df


def validate_data(df):
    report = {
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {k: str(v) for k, v in df.dtypes.items()},
        'missing': df.isna().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'target_stats': {}
    }
    if 'G3' in df.columns:
        report['target_stats'] = {
            'mean': float(df['G3'].mean()),
            'std': float(df['G3'].std()),
            'min': int(df['G3'].min()),
            'max': int(df['G3'].max())
        }
    with open(os.path.join(MET_DIR, 'data_validation.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report


def eda(df):
    # correlation with target
    numeric_df = df.select_dtypes(include=[np.number])
    if 'G3' in numeric_df.columns:
        corr = numeric_df.corr()['G3'].sort_values(ascending=False)
        corr.to_csv(os.path.join(MET_DIR, 'correlation_with_G3.csv'))
        plt.figure(figsize=(8, 6))
        corr.drop('G3').plot(kind='bar')
        plt.title('Correlation with final grade G3')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'corr_with_g3.png'))
        plt.close()
    # distribution
    if 'G3' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['G3'], bins=20, kde=True)
        plt.title('Distribution of G3')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'g3_distribution.png'))
        plt.close()
    # studytime vs g3
    if 'studytime' in df.columns and 'G3' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='studytime', y='G3', data=df)
        plt.title('Study time vs G3')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'studytime_vs_g3.png'))
        plt.close()
    meta = {
        'generated_at': datetime.utcnow().isoformat(),
        'notes': 'EDA plots saved to reports/figures'
    }
    with open(os.path.join(MET_DIR, 'eda_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def prepare_features(df):
    df = df.copy()
    # target
    y = df['G3'].copy()
    X = df.drop(columns=['G3'])
    return X, y


if __name__ == '__main__':
    df = load_data()
    rep = validate_data(df)
    print('shape', rep['shape'])
    eda(df)
    print('done')
