"""
Iris Flower Classification — Inference
========================================
Predict the species of new iris flower measurements.

Usage:
    python src/predict.py --sl 5.1 --sw 3.5 --pl 1.4 --pw 0.2
    python src/predict.py --data data/new_flowers.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd

from preprocess import FEATURES


def predict_single(sl, sw, pl, pw, model_path='outputs/best_model.pkl'):
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    X = np.array([[sl, sw, pl, pw]])
    X = saved['scaler'].transform(X)
    pred  = saved['model'].predict(X)[0]
    proba = saved['model'].predict_proba(X)[0]
    label = saved['encoder'].inverse_transform([pred])[0]
    print(f"\n🌸 Predicted Species : {label}")
    print("   Probabilities:")
    for cls, p in zip(saved['encoder'].classes_, proba):
        bar = '█' * int(p * 30)
        print(f"   {cls:<12} {p:.4f}  {bar}")


def predict_csv(data_path, model_path='outputs/best_model.pkl'):
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    df = pd.read_csv(data_path)
    X  = saved['scaler'].transform(df[FEATURES].values)
    preds = saved['encoder'].inverse_transform(saved['model'].predict(X))
    probas = saved['model'].predict_proba(X)
    df['Predicted_Species'] = preds
    for i, cls in enumerate(saved['encoder'].classes_):
        df[f'P({cls})'] = probas[:, i].round(4)
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sl',    type=float, help='Sepal length')
    parser.add_argument('--sw',    type=float, help='Sepal width')
    parser.add_argument('--pl',    type=float, help='Petal length')
    parser.add_argument('--pw',    type=float, help='Petal width')
    parser.add_argument('--data',  type=str,   help='CSV file with measurements')
    parser.add_argument('--model', default='outputs/best_model.pkl')
    args = parser.parse_args()

    if args.data:
        predict_csv(args.data, args.model)
    elif all(v is not None for v in [args.sl, args.sw, args.pl, args.pw]):
        predict_single(args.sl, args.sw, args.pl, args.pw, args.model)
    else:
        print("Provide either --data CSV or all four measurements (--sl --sw --pl --pw)")


if __name__ == '__main__':
    main()
