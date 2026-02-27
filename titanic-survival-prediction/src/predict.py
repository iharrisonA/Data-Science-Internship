"""
Titanic Survival Prediction — Inference
========================================
Run predictions on new passenger data using a saved model.

Usage:
    python src/predict.py --data data/new_passengers.csv --model outputs/best_model.pkl
"""

import argparse
import pickle
import pandas as pd
from preprocess import engineer_features, FEATURES


def predict(data_path: str, model_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df_eng = engineer_features(df)
    X = df_eng[FEATURES]

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    model = saved['model']
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df['Predicted_Survived'] = preds
    df['Survival_Probability'] = probs.round(4)
    df['Prediction_Label'] = df['Predicted_Survived'].map({1: 'Survived', 0: 'Not Survived'})
    return df[['PassengerId', 'Name', 'Pclass', 'Sex', 'Age',
               'Predicted_Survived', 'Survival_Probability', 'Prediction_Label']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   required=True, help='Path to passenger CSV')
    parser.add_argument('--model',  default='outputs/best_model.pkl')
    parser.add_argument('--output', default=None,  help='Optional: save results to CSV')
    args = parser.parse_args()

    results = predict(args.data, args.model)
    print(results.to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
