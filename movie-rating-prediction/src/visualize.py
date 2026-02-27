"""
Movie Rating Prediction — Visualization Dashboard
===================================================
Generates a comprehensive 12-panel analysis dashboard.

Usage:
    python src/visualize.py --data data/IMDb_Movies_India.csv --model outputs/best_model.pkl
"""

import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

from preprocess import full_pipeline, FEATURES, TARGET, clean

# ── Palette ──────────────────────────────────────────────────────────────────
C_NAVY  = "#1a237e"
C_RED   = "#e53935"
C_CYAN  = "#00acc1"
C_GOLD  = "#f9a825"
C_GREEN = "#43a047"
C_LIGHT = "#e8eaf6"
C_WHITE = "#ffffff"
C_GRAY  = "#9e9e9e"
PALETTE = [C_NAVY, C_CYAN, C_GOLD, C_RED, C_GREEN]

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.color':        '#cccccc',
})


def style_ax(ax, title, fontsize=13):
    ax.set_facecolor(C_WHITE)
    ax.set_title(title, fontsize=fontsize, fontweight='bold', color=C_NAVY, pad=10)
    for spine in ax.spines.values():
        spine.set_color('#dddddd')


def build_models():
    return {
        'Ridge':            Ridge(alpha=1.0),
        'Lasso':            Lasso(alpha=0.01),
        'Random Forest':    RandomForestRegressor(n_estimators=200, max_depth=10,
                                                  min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=5, subsample=0.8, random_state=42),
    }


def generate_dashboard(data_path, model_path, output_path):
    # ── Load ─────────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(data_path, encoding='latin1')
    X, y, artefacts = full_pipeline(df_raw, fit=True)
    df = clean(df_raw).dropna(subset=[TARGET])

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    best_model  = saved['model']
    best_name   = saved['name']
    all_results = saved.get('all_results', {})

    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    residuals = y - y_pred

    # CV for all models
    models = build_models()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = {}
    cv_r2   = {}
    for name, m in models.items():
        rmse = np.sqrt(-cross_val_score(m, X, y, cv=kf, scoring='neg_mean_squared_error'))
        r2   = cross_val_score(m, X, y, cv=kf, scoring='r2')
        cv_rmse[name] = rmse
        cv_r2[name]   = r2

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 28), facecolor='#f4f5f9')
    fig.suptitle('IMDb India Movie Rating Prediction — Full Analysis Dashboard',
                 fontsize=22, fontweight='bold', color=C_NAVY, y=0.985)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.48, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.96, bottom=0.03)

    # ── 1. Rating Distribution ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, '① Rating Distribution')
    ax.hist(y, bins=40, color=C_NAVY, edgecolor='white', linewidth=0.6, alpha=0.9)
    ax.axvline(y.mean(), color=C_RED,  lw=2, ls='--', label=f'Mean: {y.mean():.2f}')
    ax.axvline(y.median(), color=C_GOLD, lw=2, ls='--', label=f'Median: {y.median():.2f}')
    ax.set_xlabel('IMDb Rating', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)

    # ── 2. Ratings by Genre (top 10) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, '② Avg Rating by Genre (Top 10)')
    genre_series = df['Genre'].dropna().str.split(', ').explode()
    genre_ratings = (
        df.loc[genre_series.index, TARGET]
        .groupby(genre_series)
        .agg(['mean', 'count'])
        .query('count >= 50')
        .sort_values('mean', ascending=True)
        .tail(10)
    )
    colors_g = [C_CYAN if v >= genre_ratings['mean'].median() else C_RED
                for v in genre_ratings['mean']]
    bars = ax.barh(genre_ratings.index, genre_ratings['mean'],
                   color=colors_g, edgecolor='white', height=0.65)
    for bar, val in zip(bars, genre_ratings['mean']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, color=C_NAVY)
    ax.set_xlabel('Average Rating', fontsize=10)
    ax.set_xlim(0, 8.5)

    # ── 3. Ratings Over Time ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, '③ Average Rating by Decade')
    df_year = df.copy()
    df_year['Year_clean'] = df['Year'].astype(str).str.extract(r'(\d{4})', expand=False).astype(float)
    df_year['Decade'] = (df_year['Year_clean'] // 10 * 10).astype('Int64')
    decade_stats = (df_year.groupby('Decade')[TARGET]
                    .agg(['mean','count'])
                    .query('count >= 20')
                    .reset_index())
    ax.plot(decade_stats['Decade'], decade_stats['mean'],
            marker='o', color=C_NAVY, lw=2.5, markersize=8, markerfacecolor=C_CYAN)
    ax.fill_between(decade_stats['Decade'], decade_stats['mean'],
                    alpha=0.12, color=C_CYAN)
    ax.set_xlabel('Decade', fontsize=10)
    ax.set_ylabel('Average Rating', fontsize=10)
    ax.set_ylim(4, 8)

    # ── 4. Top 10 Directors by Avg Rating ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, '④ Top Directors by Avg Rating (≥5 films)')
    top_dirs = (df.groupby('Director')[TARGET]
                .agg(['mean','count'])
                .query('count >= 5')
                .sort_values('mean', ascending=False)
                .head(10)
                .sort_values('mean', ascending=True))
    bars = ax.barh(top_dirs.index, top_dirs['mean'],
                   color=C_GOLD, edgecolor='white', height=0.65)
    for bar, val in zip(bars, top_dirs['mean']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, color=C_NAVY)
    ax.set_xlabel('Average Rating', fontsize=10)
    ax.set_xlim(0, 10)

    # ── 5. Top 10 Actors by Avg Rating ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, '⑤ Top Lead Actors by Avg Rating (≥5 films)')
    top_actors = (df.groupby('Actor 1')[TARGET]
                  .agg(['mean','count'])
                  .query('count >= 5')
                  .sort_values('mean', ascending=False)
                  .head(10)
                  .sort_values('mean', ascending=True))
    bars = ax.barh(top_actors.index, top_actors['mean'],
                   color=C_CYAN, edgecolor='white', height=0.65)
    for bar, val in zip(bars, top_actors['mean']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, color=C_NAVY)
    ax.set_xlabel('Average Rating', fontsize=10)
    ax.set_xlim(0, 10)

    # ── 6. Votes vs Rating Scatter ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, '⑥ Votes vs Rating')
    df_votes = df.copy()
    df_votes['Votes_num'] = pd.to_numeric(
        df_votes['Votes'].astype(str).str.replace(',','', regex=False), errors='coerce')
    sample = df_votes.dropna(subset=[TARGET, 'Votes_num']).sample(
        min(2000, len(df_votes)), random_state=42)
    sc = ax.scatter(np.log1p(sample['Votes_num']), sample[TARGET],
                    c=sample[TARGET], cmap='RdYlGn',
                    alpha=0.4, s=15, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Rating', shrink=0.8)
    ax.set_xlabel('Log(Votes + 1)', fontsize=10)
    ax.set_ylabel('IMDb Rating', fontsize=10)

    # ── 7. Duration vs Rating ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, '⑦ Duration vs Rating')
    df_dur = df.copy()
    df_dur['Duration_min'] = (df_dur['Duration'].astype(str)
                               .str.extract(r'(\d+)', expand=False).astype(float))
    df_dur = df_dur.dropna(subset=[TARGET, 'Duration_min'])
    df_dur = df_dur[df_dur['Duration_min'].between(30, 240)]
    df_dur['DurBin'] = pd.cut(df_dur['Duration_min'],
                               bins=[30,60,90,105,120,150,240],
                               labels=['30-60','60-90','90-105','105-120','120-150','150+'])
    dur_stats = df_dur.groupby('DurBin', observed=True)[TARGET].mean()
    ax.bar(dur_stats.index, dur_stats.values, color=C_GREEN,
           edgecolor='white', linewidth=1.2, width=0.6)
    for i, val in enumerate(dur_stats.values):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=9,
                fontweight='bold', color=C_NAVY)
    ax.set_xlabel('Duration (minutes)', fontsize=10)
    ax.set_ylabel('Average Rating', fontsize=10)
    ax.set_ylim(0, 8)

    # ── 8. Model RMSE Comparison ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    style_ax(ax, '⑧ Model RMSE Comparison (5-Fold CV)')
    names  = list(cv_rmse.keys())
    means  = [cv_rmse[n].mean() for n in names]
    stds   = [cv_rmse[n].std()  for n in names]
    colors_m = [C_GOLD if n == best_name else C_NAVY for n in names]
    short  = {'Ridge': 'Ridge', 'Lasso': 'Lasso',
               'Random Forest': 'RF', 'Gradient Boosting': 'GB'}
    bars = ax.barh([short[n] for n in names], means, xerr=stds,
                   color=colors_m, edgecolor='white', height=0.5,
                   error_kw=dict(ecolor=C_GRAY, capsize=4))
    for bar, val in zip(bars, means):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold', color=C_NAVY)
    ax.set_xlabel('RMSE (lower is better)', fontsize=10)
    ax.legend(handles=[mpatches.Patch(color=C_GOLD, label='Best Model')], fontsize=9)

    # ── 9. R² Comparison ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    style_ax(ax, '⑨ Model R² Comparison (5-Fold CV)')
    means_r2 = [cv_r2[n].mean() for n in names]
    stds_r2  = [cv_r2[n].std()  for n in names]
    colors_r = [C_GOLD if n == best_name else C_CYAN for n in names]
    bars = ax.barh([short[n] for n in names], means_r2, xerr=stds_r2,
                   color=colors_r, edgecolor='white', height=0.5,
                   error_kw=dict(ecolor=C_GRAY, capsize=4))
    for bar, val in zip(bars, means_r2):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold', color=C_NAVY)
    ax.set_xlabel('R² (higher is better)', fontsize=10)
    ax.set_xlim(0, 1)

    # ── 10. Actual vs Predicted ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    style_ax(ax, f'⑩ Actual vs Predicted — {best_name}')
    sample_idx = np.random.choice(len(y), min(1500, len(y)), replace=False)
    ax.scatter(y.iloc[sample_idx], y_pred[sample_idx],
               alpha=0.3, s=12, color=C_NAVY, edgecolors='none')
    lims = [max(1, y.min()-0.5), min(10, y.max()+0.5)]
    ax.plot(lims, lims, color=C_RED, lw=2, ls='--', label='Perfect fit')
    ax.set_xlabel('Actual Rating', fontsize=10)
    ax.set_ylabel('Predicted Rating', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    r2_val = r2_score(y, y_pred)
    ax.text(0.05, 0.92, f'R² = {r2_val:.4f}', transform=ax.transAxes,
            fontsize=10, color=C_RED, fontweight='bold')

    # ── 11. Residuals Distribution ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    style_ax(ax, '⑪ Residuals Distribution')
    ax.hist(residuals, bins=50, color=C_CYAN, edgecolor='white',
            linewidth=0.5, alpha=0.85)
    ax.axvline(0, color=C_RED, lw=2, ls='--', label='Zero residual')
    ax.axvline(residuals.mean(), color=C_GOLD, lw=2, ls='--',
               label=f'Mean: {residuals.mean():.3f}')
    ax.set_xlabel('Residual (Actual − Predicted)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=9)

    # ── 12. Feature Importance ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    style_ax(ax, f'⑫ Feature Importance — {best_name}')
    if hasattr(best_model, 'feature_importances_'):
        imps = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        imps = np.abs(best_model.coef_)
    else:
        imps = np.ones(len(FEATURES))

    feat_disp = {
        'Year_clean': 'Release Year', 'Duration_min': 'Duration (min)',
        'Votes_log': 'Log Votes', 'Genre_count': 'Genre Count',
        'genre_Drama': 'Genre: Drama', 'genre_Action': 'Genre: Action',
        'genre_Romance': 'Genre: Romance', 'genre_Comedy': 'Genre: Comedy',
        'genre_Thriller': 'Genre: Thriller', 'genre_Crime': 'Genre: Crime',
        'genre_Horror': 'Genre: Horror', 'genre_Family': 'Genre: Family',
        'genre_Musical': 'Genre: Musical', 'genre_Adventure': 'Genre: Adventure',
        'genre_Mystery': 'Genre: Mystery', 'genre_Biography': 'Genre: Biography',
        'Director_enc': 'Director (encoded)', 'Actor1_enc': 'Lead Actor',
        'Actor2_enc': 'Actor 2', 'Actor3_enc': 'Actor 3',
        'Director_avg_rating': 'Director Avg Rating',
        'Actor1_avg_rating': 'Lead Actor Avg Rating',
    }
    fi = pd.Series(imps, index=FEATURES).sort_values(ascending=True).tail(12)
    fi.index = [feat_disp.get(i, i) for i in fi.index]
    colors_fi = [C_CYAN if v >= fi.quantile(0.66) else
                 C_GOLD if v >= fi.quantile(0.33) else C_RED
                 for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='white', height=0.65)
    ax.set_xlabel('Importance Score', fontsize=10)
    for i, val in enumerate(fi.values):
        ax.text(val * 1.01, i, f'{val:.4f}', va='center', fontsize=8, color=C_NAVY)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f4f5f9')
    plt.close()
    print(f"📊 Dashboard saved: {output_path}")

    # Print summary metrics
    rmse_train = np.sqrt(mean_squared_error(y, y_pred))
    mae_train  = mean_absolute_error(y, y_pred)
    r2_train   = r2_score(y, y_pred)
    print(f"\n📈 Final Model Metrics ({best_name})")
    print(f"   Train RMSE : {rmse_train:.4f}")
    print(f"   Train MAE  : {mae_train:.4f}")
    print(f"   Train R²   : {r2_train:.4f}")
    best_cv_rmse = cv_rmse.get(best_name, cv_rmse[list(cv_rmse.keys())[2]])
    print(f"   CV RMSE    : {best_cv_rmse.mean():.4f} ± {best_cv_rmse.std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/IMDb_Movies_India.csv')
    parser.add_argument('--model',  default='outputs/best_model.pkl')
    parser.add_argument('--output', default='outputs/movie_rating_dashboard.png')
    args = parser.parse_args()
    generate_dashboard(args.data, args.model, args.output)


if __name__ == '__main__':
    main()
