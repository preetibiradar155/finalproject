# Student Marks Prediction - End-to-End ML Capstone

## Overview
This project predicts a student's final grade (G3) using demographic, social, and school-related features from the UCI Student Performance dataset (Math course). It covers data prep, EDA, training/eval with MLflow tracking, FastAPI inference, Dockerization, and tests.

## Problem statement
Regression task: predict final grade G3 (numeric) based on input features. Use case - early identification/intervention for at-risk students.

## Dataset
- Source: UCI Student Performance dataset (student-mat.csv). In case of download issues, a reproducible synthetic dataset matching schema is generated as fallback (logged).
- Target: G3 (final grade)
- Features: 33 columns (see UCI description) including school, sex, age, Medu, Fedu, studytime, failures, absences, G1, G2, etc.
- Stored in `data/`.

## Tech stack
Python, pandas, scikit-learn, matplotlib, seaborn, MLflow, FastAPI, Uvicorn, Pydantic, pytest, joblib, Docker.

## Project structure
```
project/
├── data/
├── src/
│   ├── data_preparation.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── app.py
├── models/
├── reports/figures/, reports/metrics/
├── tests/
├── mlruns/
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── Makefile
├── run_all.ps1
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Data
```bash
cd data && python getdata.py
python -c "from src import data_preparation; df=data_preparation.load_data(); data_preparation.validate_data(df); data_preparation.eda(df)"
```

## Train & Evaluate
```bash
python -c "from src import train; train.train()"
python -c "from src import evaluate; evaluate.evaluate()"
```
MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`

## FastAPI
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
Endpoints: GET /, GET /health, GET /model-info, POST /predict

Example payload:
```json
{
  "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3", "Pstatus": "T",
  "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher", "reason": "course", "guardian": "mother",
  "traveltime": 2, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
  "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
  "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1, "Walc": 1, "health": 3, "absences": 4,
  "G1": 10, "G2": 11
}
```

## Docker
```bash
docker build -t student-marks-prediction:latest .
docker run -p 8000:8000 student-marks-prediction:latest
```
(Docker may not be available in this environment; config is provided.)

## Tests
```bash
pytest -v
```

## All-in-one
```bash
.\run_all.ps1  # Windows
make all        # if make available
```

## Known limitations
- If network access blocked, synthetic data used as fallback (clearly noted). 
- MLflow model registry support depends on backend; local file store used.
- Docker execution not verified if daemon unavailable.
- Predictions are model-based estimates, not guarantees.
