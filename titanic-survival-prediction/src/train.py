"""
Titanic Survival Prediction — Model Training
=============================================
Trains and evaluates multiple ML models on the Titanic dataset.
Saves the best model to outputs/best_model.pkl

Usage:
    python src/train.py --data data/tested.csv
"""

import argparse
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from preprocess import engineer_features, FEATURES


def train_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
        'SVM':                 SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n📊 Cross-Validation Results (5-Fold)")
    print("=" * 50)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = (scores, model)
        print(f"  {name:25s}  {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k][0].mean())
    best_model = results[best_name][1]
    best_model.fit(X, y)

    y_pred = best_model.predict(X)
    print(f"\n✅ Best Model: {best_name}")
    print(f"   Training Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"\n{classification_report(y, y_pred, target_names=['Not Survived', 'Survived'])}")

    return best_name, best_model, results


def main():
    parser = argparse.ArgumentParser(description='Train Titanic survival prediction models')
    parser.add_argument('--data', type=str, default='data/tested.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='outputs/best_model.pkl',
                        help='Path to save the best model')
    args = parser.parse_args()

    print(f"📂 Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"   Shape: {df.shape}")

    df = engineer_features(df)
    X = df[FEATURES]
    y = df['Survived']

    best_name, best_model, results = train_models(X, y)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({'model': best_model, 'name': best_name, 'features': FEATURES}, f)
    print(f"\n💾 Model saved to: {args.output}")


if __name__ == '__main__':
    main()
