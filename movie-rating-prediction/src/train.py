"""
Movie Rating Prediction — Model Training
=========================================
Trains multiple regression models, evaluates with cross-validation,
and saves the best model + encoders to outputs/.

Usage:
    python src/train.py --data data/IMDb_Movies_India.csv
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model    import Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

from preprocess import full_pipeline, FEATURES, TARGET


def evaluate(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                           scoring='neg_mean_squared_error'))
    r2_scores   = cross_val_score(model, X, y, cv=kf, scoring='r2')
    return rmse_scores, r2_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/IMDb_Movies_India.csv')
    parser.add_argument('--output', default='outputs/best_model.pkl')
    args = parser.parse_args()

    print(f"📂 Loading: {args.data}")
    df = pd.read_csv(args.data, encoding='latin1')
    print(f"   Raw shape: {df.shape}")

    X, y, artefacts = full_pipeline(df, fit=True)
    print(f"   Clean shape: {X.shape}  |  Ratings: {len(y)}")
    print(f"   Rating mean: {y.mean():.3f}  std: {y.std():.3f}")

    models = {
        'Ridge Regression':       Ridge(alpha=1.0),
        'Lasso Regression':       Lasso(alpha=0.01),
        'Random Forest':          RandomForestRegressor(n_estimators=200, max_depth=10,
                                                        min_samples_leaf=5, random_state=42,
                                                        n_jobs=-1),
        'Gradient Boosting':      GradientBoostingRegressor(n_estimators=300,
                                                             learning_rate=0.05,
                                                             max_depth=5,
                                                             subsample=0.8,
                                                             random_state=42),
    }

    results = {}
    print("\n📊 Cross-Validation Results (5-Fold)")
    print("=" * 60)
    print(f"  {'Model':<25} {'RMSE':>8}  {'±':>6}  {'R²':>7}  {'±':>6}")
    print("-" * 60)

    for name, model in models.items():
        rmse_cv, r2_cv = evaluate(model, X, y)
        results[name] = {
            'model': model,
            'rmse_mean': rmse_cv.mean(), 'rmse_std': rmse_cv.std(),
            'r2_mean':   r2_cv.mean(),   'r2_std':   r2_cv.std(),
        }
        print(f"  {name:<25} {rmse_cv.mean():>8.4f}  {rmse_cv.std():>6.4f}  "
              f"{r2_cv.mean():>7.4f}  {r2_cv.std():>6.4f}")

    # Best by RMSE
    best_name = min(results, key=lambda k: results[k]['rmse_mean'])
    best_model = results[best_name]['model']
    best_model.fit(X, y)

    # Full training metrics
    y_pred = best_model.predict(X)
    print(f"\n✅ Best Model: {best_name}")
    print(f"   Train RMSE : {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"   Train MAE  : {mean_absolute_error(y, y_pred):.4f}")
    print(f"   Train R²   : {r2_score(y, y_pred):.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'name': best_name,
            'artefacts': artefacts,
            'features': FEATURES,
            'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                            for k, v in results.items()},
        }, f)
    print(f"\n💾 Saved to: {args.output}")
    return results, best_name


if __name__ == '__main__':
    main()
