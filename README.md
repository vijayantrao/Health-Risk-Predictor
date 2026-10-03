# 🩺 Diabetes Risk Prediction Dashboard

A machine learning web application that predicts an individual's diabetes risk from key health parameters, built with scikit-learn and presented as an interactive Streamlit dashboard.

> **Disclaimer:** This is a learning/portfolio project, not a medical diagnostic tool.

## Problem Statement

Early identification of diabetes risk enables preventive action before clinical diagnosis. This project trains a classification model on real clinical records and wraps it in an accessible dashboard where users enter their own health metrics and receive an instant, probability-scored risk assessment.

## Dataset

- **Source:** [Diabetes Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Size:** 100,000 patient records
- **Features:** gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level
- **Target:** `diabetes` (0 = no, 1 = yes)
- **Class imbalance:** only ~8.5% of records are diabetic, so accuracy alone is misleading

## Model & Performance

| Metric (diabetic class) | Score |
|---|---|
| Accuracy | 96.0% |
| Precision | 0.77 |
| Recall | 0.75 |
| F1 | 0.76 |

- **Algorithm:** Random Forest Classifier (100 trees)
- **Imbalance handling:** `class_weight='balanced'` plus a stratified 80/20 train/test split
- **Why not just accuracy?** A model that always predicts "no diabetes" would already be ~91.5% accurate, so precision and recall on the minority class are the honest measures.

Exact figures are regenerated into `metrics.json` every time `model.py` runs, and the dashboard reads them from there.

## Tech Stack

Python · pandas · scikit-learn · Streamlit · Plotly

## Features

- Real-time risk prediction with a probability score
- Dataset exploration: age and BMI distributions by diabetes status
- Feature-importance chart showing which factors drive predictions most

## Project Structure

```
health-risk-predictor/
├── app.py            # Streamlit dashboard
├── model.py          # trains + evaluates the model, saves artifacts
├── requirements.txt
├── metrics.json      # evaluation results from the latest training run
├── data/diabetes_prediction_dataset.csv
└── *.pkl             # trained model, encoders, column order
```

## Run Locally

```bash
git clone https://github.com/<your-username>/health-risk-predictor.git
cd health-risk-predictor
python -m venv venv
venv\Scripts\activate        # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python model.py              # retrains and saves the model
streamlit run app.py         # opens the dashboard at localhost:8501
```

## Live Demo

_Add your Streamlit Community Cloud link here after deploying._

## Future Improvements

- Tune the decision threshold or try SMOTE / XGBoost to raise recall
- Hyperparameter search with `GridSearchCV`
- Add SHAP explanations for individual predictions
