"""
Titanic Survival Prediction — Visualizations
=============================================
Generates the full 11-panel analysis dashboard and saves it to outputs/.

Usage:
    python src/visualize.py --data data/tested.csv --model outputs/best_model.pkl
"""

import argparse
import pickle
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

from preprocess import engineer_features, FEATURES, get_feature_display_names

# ── Palette ──────────────────────────────────────────────────────────────────
C_NAVY, C_RED, C_CYAN, C_GOLD = "#1a237e", "#e53935", "#00acc1", "#f9a825"
C_LIGHT, C_WHITE, C_GRAY      = "#e8eaf6", "#ffffff", "#9e9e9e"
PALETTE = [C_NAVY, C_RED, C_CYAN, C_GOLD]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
})


def style_ax(ax, title):
    ax.set_facecolor(C_WHITE)
    ax.set_title(title, fontsize=13, fontweight='bold', color=C_NAVY, pad=10)
    for spine in ax.spines.values():
        spine.set_color('#dddddd')


def build_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
        'SVM':                 SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    }


def generate_dashboard(df, X, y, best_model, best_name, cv_results, output_path):
    fig = plt.figure(figsize=(20, 24), facecolor='#f5f5f5')
    fig.suptitle('Titanic Survival Prediction — Analysis & Model Results',
                 fontsize=22, fontweight='bold', color=C_NAVY, y=0.98)

    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.94, bottom=0.04)

    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]
    models = build_models()

    # 1 — Survival Distribution
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, '① Survival Distribution')
    counts = y.value_counts()
    bars = ax.bar(['Not Survived', 'Survived'], counts.values,
                  color=[C_RED, C_CYAN], width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}\n({val/len(y)*100:.1f}%)', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Count'); ax.set_ylim(0, max(counts.values) * 1.2)

    # 2 — Survival by Sex
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, '② Survival by Sex')
    df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack().plot(
        kind='bar', ax=ax, color=[C_RED, C_CYAN], edgecolor='white', width=0.6)
    ax.set_xticklabels(['Female', 'Male'], rotation=0, fontsize=11)
    ax.set_ylabel('Proportion'); ax.legend(['Not Survived', 'Survived'], fontsize=9)
    ax.set_xlabel('')

    # 3 — Survival by Pclass
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, '③ Survival by Passenger Class')
    ps = df.groupby('Pclass')['Survived'].mean() * 100
    bars = ax.bar(['1st Class', '2nd Class', '3rd Class'], ps.values,
                  color=[C_CYAN, C_GOLD, C_RED], width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, ps.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Survival Rate (%)'); ax.set_ylim(0, 100)

    # 4 — Age Distribution
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, '④ Age Distribution by Survival')
    ax.hist(df[df['Survived']==0]['Age'].dropna(), bins=25, alpha=0.7,
            color=C_RED, label='Not Survived', edgecolor='white')
    ax.hist(df[df['Survived']==1]['Age'].dropna(), bins=25, alpha=0.7,
            color=C_CYAN, label='Survived', edgecolor='white')
    ax.set_xlabel('Age'); ax.set_ylabel('Count'); ax.legend(fontsize=9)

    # 5 — Fare Boxplot
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, '⑤ Fare vs Survival')
    ax.boxplot([df[df['Survived']==0]['Fare'], df[df['Survived']==1]['Fare']],
               labels=['Not Survived', 'Survived'], patch_artist=True,
               boxprops=dict(facecolor=C_LIGHT, color=C_NAVY),
               medianprops=dict(color=C_RED, linewidth=2),
               whiskerprops=dict(color=C_NAVY), capprops=dict(color=C_NAVY))
    ax.set_ylabel('Fare (£)'); ax.set_ylim(0, 200)

    # 6 — Family Size
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, '⑥ Family Size vs Survival Rate')
    fs = df.groupby('FamilySize')['Survived'].mean() * 100
    ax.plot(fs.index, fs.values, marker='o', color=C_NAVY, linewidth=2.5,
            markersize=8, markerfacecolor=C_CYAN)
    ax.fill_between(fs.index, fs.values, alpha=0.15, color=C_CYAN)
    ax.set_xlabel('Family Size'); ax.set_ylabel('Survival Rate (%)'); ax.set_xticks(fs.index)

    # 7 — Model Comparison
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, '⑦ Model Accuracy Comparison (5-Fold CV)')
    names_short = {'Logistic Regression': 'LR', 'Random Forest': 'RF',
                   'Gradient Boosting': 'GB', 'SVM': 'SVM'}
    keys = list(cv_results.keys())
    means = [cv_results[k].mean() for k in keys]
    stds  = [cv_results[k].std()  for k in keys]
    colors = [C_GOLD if k == best_name else C_NAVY for k in keys]
    bars = ax.barh([names_short[k] for k in keys], means, xerr=stds,
                   color=colors, edgecolor='white', linewidth=1.5, height=0.5,
                   error_kw=dict(ecolor=C_GRAY, capsize=4))
    for bar, val in zip(bars, means):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold', color=C_NAVY)
    ax.set_xlim(0.6, 1.0); ax.set_xlabel('Accuracy')
    ax.legend(handles=[mpatches.Patch(color=C_GOLD, label='Best Model')], fontsize=9)

    # 8 — Confusion Matrix
    ax = fig.add_subplot(gs[2, 1])
    style_ax(ax, f'⑧ Confusion Matrix — {best_name}')
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'],
                linewidths=1, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

    # 9 — ROC Curves
    ax = fig.add_subplot(gs[2, 2])
    style_ax(ax, '⑨ ROC Curves — All Models')
    for (name, model), color in zip(models.items(), PALETTE):
        model.fit(X, y)
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{names_short[name]} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9); ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])

    # 10 — Feature Importance
    ax = fig.add_subplot(gs[3, 0:2])
    style_ax(ax, f'⑩ Feature Importance — {best_name}')
    imps = (best_model.feature_importances_ if hasattr(best_model, 'feature_importances_')
            else np.abs(best_model.coef_[0]))
    feat_imp = pd.Series(imps, index=FEATURES).sort_values(ascending=True)
    disp = get_feature_display_names()
    feat_imp.index = [disp.get(i, i) for i in feat_imp.index]
    ci = [C_CYAN if v >= feat_imp.quantile(0.66) else
          C_GOLD if v >= feat_imp.quantile(0.33) else C_RED for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=ci, edgecolor='white',
            linewidth=1.2, height=0.6)
    ax.set_xlabel('Importance Score')
    for i, val in enumerate(feat_imp.values):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9, color=C_NAVY)

    # 11 — Metrics Card
    ax = fig.add_subplot(gs[3, 2])
    ax.set_facecolor(C_NAVY)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 0.92, '⑪ Best Model Metrics', ha='center', va='top',
            transform=ax.transAxes, fontsize=13, fontweight='bold', color=C_WHITE)
    ax.text(0.5, 0.80, best_name, ha='center', va='top',
            transform=ax.transAxes, fontsize=10, color=C_GOLD)
    for i, (metric, val) in enumerate({
        'Accuracy':  accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall':    recall_score(y, y_pred),
        'F1 Score':  f1_score(y, y_pred),
    }.items()):
        yp = 0.63 - i * 0.15
        ax.text(0.15, yp, metric, ha='left', va='center',
                transform=ax.transAxes, fontsize=11, color=C_LIGHT)
        ax.text(0.85, yp, f'{val:.4f}', ha='right', va='center',
                transform=ax.transAxes, fontsize=14, fontweight='bold', color=C_CYAN)
    ax.text(0.5, 0.05, f'CV Score: {cv_results[best_name].mean():.4f}',
            ha='center', va='bottom', transform=ax.transAxes, fontsize=10, color=C_GOLD)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f5f5f5')
    plt.close()
    print(f"📊 Dashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/tested.csv')
    parser.add_argument('--model',  default='outputs/best_model.pkl')
    parser.add_argument('--output', default='outputs/titanic_analysis.png')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = engineer_features(df)
    X, y = df[FEATURES], df['Survived']

    with open(args.model, 'rb') as f:
        saved = pickle.load(f)
    best_model = saved['model']
    best_name  = saved['name']

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {name: cross_val_score(m, X, y, cv=cv, scoring='accuracy')
                  for name, m in models.items()}

    generate_dashboard(df, X, y, best_model, best_name, cv_results, args.output)


if __name__ == '__main__':
    main()
