"""
Movie Rating Prediction — Inference
=====================================
Predict IMDb ratings for new movies using a saved model.

Usage:
    python src/predict.py --data data/new_movies.csv --model outputs/best_model.pkl
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from preprocess import full_pipeline, FEATURES


def predict(data_path: str, model_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, encoding='latin1')

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    X, _, = full_pipeline(df, artefacts=saved['artefacts'], fit=False)
    preds = saved['model'].predict(X)
    preds = np.clip(preds, 1.0, 10.0).round(2)

    out = df[['Name']].copy() if 'Name' in df.columns else pd.DataFrame()
    out['Predicted_Rating'] = preds
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   required=True)
    parser.add_argument('--model',  default='outputs/best_model.pkl')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    results = predict(args.data, args.model)
    print(results.to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n✅ Saved to: {args.output}")


if __name__ == '__main__':
    main()
