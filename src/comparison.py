# src/comparison.py

"""
This file demonstrates a simplistic approach to comparing multiple algorithms
based on their lexical and semantic features extracted using our pipeline.

While `train_model.py` does the actual comparison and training, this file can be
used as a reference or a utility to compare various sklearn models quickly.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def compare_models(X, y):
    """
    Compare different classifiers using cross-validation and return the best model.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_score = -1
    best_model = None

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        mean_score = scores.mean()
        print(f"{name} Mean F1 Score: {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model
