import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

ARTIFACT_DIR = 'tourism_project/artifacts'
DEPLOY_DIR   = 'tourism_project/deployment'

def load_splits():
    X_train = pd.read_csv(ARTIFACT_DIR + '/X_train.csv')
    X_test  = pd.read_csv(ARTIFACT_DIR + '/X_test.csv')
    y_train = pd.read_csv(ARTIFACT_DIR + '/y_train.csv').squeeze()
    y_test  = pd.read_csv(ARTIFACT_DIR + '/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def get_models_and_params():
    return {
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            {'max_depth': [3, 5, 7, 10, None],
             'min_samples_split': [2, 5, 10],
             'criterion': ['gini', 'entropy']}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            {'n_estimators': [100, 200, 300],
             'max_depth': [5, 10, 20, None],
             'min_samples_split': [2, 5],
             'max_features': ['sqrt', 'log2']}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [100, 200],
             'learning_rate': [0.05, 0.1, 0.2],
             'max_depth': [3, 4, 5],
             'subsample': [0.8, 1.0]}
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {'n_estimators': [50, 100, 200],
             'learning_rate': [0.5, 1.0, 1.5]}
        ),
        'XGBoost': (
            XGBClassifier(random_state=42, eval_metric='logloss',
                          scale_pos_weight=4, use_label_encoder=False),
            {'n_estimators': [100, 200, 300],
             'max_depth': [3, 5, 7],
             'learning_rate': [0.05, 0.1, 0.2],
             'subsample': [0.7, 0.8, 1.0]}
        )
    }

def train_and_tune(X_train, X_test, y_train, y_test):
    models    = get_models_and_params()
    results   = {}
    best_f1   = 0
    best_model = None
    best_name  = ''
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    for name, (model, params) in models.items():
        print('--- Tuning', name, '---')
        search = RandomizedSearchCV(
            model, params, n_iter=15, cv=5,
            scoring='f1', n_jobs=-1, random_state=42, verbose=0
        )
        search.fit(X_train, y_train)
        tuned  = search.best_estimator_
        y_pred = tuned.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, tuned.predict_proba(X_test)[:, 1])

        print('  Best params :', search.best_params_)
        print('  Accuracy    :', round(acc, 4))
        print('  F1 Score    :', round(f1, 4))
        print('  ROC-AUC     :', round(auc, 4))
        print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

        results[name] = {
            'best_params': search.best_params_,
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'roc_auc':  round(auc, 4)
        }
        if f1 > best_f1:
            best_f1    = f1
            best_model = tuned
            best_name  = name

    model_path = DEPLOY_DIR + '/best_model.pkl'
    joblib.dump(best_model, model_path)
    print('Best model:', best_name, '| F1:', round(best_f1, 4), '| saved to', model_path)

    with open(ARTIFACT_DIR + '/experiment_log.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Experiment log saved.')
    return results, best_name, best_model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_splits()
    train_and_tune(X_train, X_test, y_train, y_test)
