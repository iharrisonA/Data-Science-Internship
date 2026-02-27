"""
Iris Flower Classification — Model Training
============================================
Trains and evaluates multiple classifiers, saves the best model.

Usage:
    python src/train.py --data data/IRIS.csv
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)

from preprocess import load_data, split_and_scale, FEATURES, TARGET


def build_models():
    return {
        'Logistic Regression':  LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree':        DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':        RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=200,
                                                            learning_rate=0.05,
                                                            random_state=42),
        'SVM':                  SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'K-Nearest Neighbors':  KNeighborsClassifier(n_neighbors=5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/IRIS.csv')
    parser.add_argument('--output', default='outputs/best_model.pkl')
    args = parser.parse_args()

    print(f"📂 Loading: {args.data}")
    df = load_data(args.data)
    print(f"   Shape: {df.shape}  |  Classes: {df[TARGET].unique().tolist()}")

    X_train, X_test, y_train, y_test, scaler, le = split_and_scale(df)
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

    models = build_models()
    cv     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    print("\n📊 Cross-Validation Results (10-Fold)")
    print("=" * 55)
    print(f"  {'Model':<25} {'CV Acc':>8}  {'±':>7}")
    print("-" * 55)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        results[name] = {'scores': scores, 'model': model}
        print(f"  {name:<25} {scores.mean():>8.4f}  {scores.std():>7.4f}")

    best_name  = max(results, key=lambda k: results[k]['scores'].mean())
    best_model = results[best_name]['model']
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\n✅ Best Model: {best_name}")
    print(f"   Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'model':    best_model,
            'name':     best_name,
            'scaler':   scaler,
            'encoder':  le,
            'features': FEATURES,
            'all_results': {k: v['scores'] for k, v in results.items()},
        }, f)
    print(f"💾 Model saved to: {args.output}")


if __name__ == '__main__':
    main()
