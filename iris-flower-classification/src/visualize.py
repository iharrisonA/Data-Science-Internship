"""
Iris Flower Classification — Visualization Dashboard
=====================================================
Generates a comprehensive 12-panel analysis + model results dashboard.

Usage:
    python src/visualize.py --data data/IRIS.csv --model outputs/best_model.pkl
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
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.decomposition   import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics         import confusion_matrix, roc_curve, auc
from sklearn.preprocessing   import label_binarize
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier

from preprocess import load_data, split_and_scale, FEATURES, TARGET

# ── Palette ──────────────────────────────────────────────────────────────────
C_SETOSA     = "#1a237e"   # navy
C_VERSI      = "#00acc1"   # cyan
C_VIRG       = "#f9a825"   # gold
C_NAVY       = "#1a237e"
C_RED        = "#e53935"
C_CYAN       = "#00acc1"
C_GOLD       = "#f9a825"
C_GREEN      = "#43a047"
C_LIGHT      = "#e8eaf6"
C_WHITE      = "#ffffff"
C_GRAY       = "#9e9e9e"
SPECIES_COLORS = [C_SETOSA, C_VERSI, C_VIRG]
MODEL_COLORS   = [C_NAVY, C_CYAN, C_GOLD, C_RED, C_GREEN, "#8e24aa"]

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.color':        '#cccccc',
})

SPECIES_LABELS = ['Setosa', 'Versicolor', 'Virginica']


def style_ax(ax, title, fontsize=13):
    ax.set_facecolor(C_WHITE)
    ax.set_title(title, fontsize=fontsize, fontweight='bold', color=C_NAVY, pad=10)
    for spine in ax.spines.values():
        spine.set_color('#dddddd')


def build_all_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200,
                                                           learning_rate=0.05, random_state=42),
        'SVM':                 SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
    }


def generate_dashboard(data_path, model_path, output_path):
    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler, le = split_and_scale(df)

    with open(model_path, 'rb') as f:
        saved      = pickle.load(f)
    best_model = saved['model']
    best_name  = saved['name']
    all_cv     = saved.get('all_results', {})

    best_model.fit(X_train, y_train)
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    X_all  = scaler.transform(df[FEATURES].values)
    y_all  = le.transform(df[TARGET].values)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 28), facecolor='#f4f5f9')
    fig.suptitle('Iris Flower Classification — Full Analysis Dashboard',
                 fontsize=22, fontweight='bold', color=C_NAVY, y=0.985)
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.50, wspace=0.35,
                           left=0.06, right=0.97, top=0.96, bottom=0.03)

    # ── 1. Species Distribution ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, '① Species Distribution')
    counts = df[TARGET].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=SPECIES_COLORS, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_ylim(0, 65)

    # ── 2. Feature Boxplots by Species ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, '② Petal Length by Species')
    data_bp = [df[df[TARGET] == sp]['petal_length'].values for sp in SPECIES_LABELS]
    bp = ax.boxplot(data_bp, patch_artist=True, labels=SPECIES_LABELS,
                    medianprops=dict(color=C_RED, linewidth=2.5),
                    whiskerprops=dict(color=C_NAVY),
                    capprops=dict(color=C_NAVY))
    for patch, color in zip(bp['boxes'], SPECIES_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel('Petal Length (cm)', fontsize=10)

    # ── 3. Petal Width by Species ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, '③ Petal Width by Species')
    data_pw = [df[df[TARGET] == sp]['petal_width'].values for sp in SPECIES_LABELS]
    bp2 = ax.boxplot(data_pw, patch_artist=True, labels=SPECIES_LABELS,
                     medianprops=dict(color=C_RED, linewidth=2.5),
                     whiskerprops=dict(color=C_NAVY),
                     capprops=dict(color=C_NAVY))
    for patch, color in zip(bp2['boxes'], SPECIES_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel('Petal Width (cm)', fontsize=10)

    # ── 4. Sepal Length vs Sepal Width Scatter ────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, '④ Sepal Length vs Sepal Width')
    for sp, color in zip(SPECIES_LABELS, SPECIES_COLORS):
        sub = df[df[TARGET] == sp]
        ax.scatter(sub['sepal_length'], sub['sepal_width'],
                   color=color, label=sp, alpha=0.75, s=55, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Sepal Length (cm)', fontsize=10)
    ax.set_ylabel('Sepal Width (cm)', fontsize=10)
    ax.legend(fontsize=9)

    # ── 5. Petal Length vs Petal Width Scatter ────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, '⑤ Petal Length vs Petal Width')
    for sp, color in zip(SPECIES_LABELS, SPECIES_COLORS):
        sub = df[df[TARGET] == sp]
        ax.scatter(sub['petal_length'], sub['petal_width'],
                   color=color, label=sp, alpha=0.75, s=55, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Petal Length (cm)', fontsize=10)
    ax.set_ylabel('Petal Width (cm)', fontsize=10)
    ax.legend(fontsize=9)

    # ── 6. PCA 2D Projection ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, '⑥ PCA 2D Projection')
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)
    for i, (sp, color) in enumerate(zip(SPECIES_LABELS, SPECIES_COLORS)):
        mask = y_all == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=color, label=sp, alpha=0.75, s=55, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax.legend(fontsize=9)

    # ── 7. Correlation Heatmap ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, '⑦ Feature Correlation Heatmap')
    corr = df[FEATURES].corr()
    nice = {'sepal_length': 'Sepal Len', 'sepal_width': 'Sepal Wid',
            'petal_length': 'Petal Len', 'petal_width': 'Petal Wid'}
    corr.columns = [nice[c] for c in corr.columns]
    corr.index   = [nice[c] for c in corr.index]
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                linewidths=1, linecolor='white', vmin=-1, vmax=1,
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.tick_params(axis='x', rotation=30)

    # ── 8. Model Accuracy Comparison ─────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    style_ax(ax, '⑧ Model Accuracy (10-Fold CV)')
    model_names  = list(all_cv.keys())
    short_names  = {'Logistic Regression': 'LR', 'Decision Tree': 'DT',
                    'Random Forest': 'RF', 'Gradient Boosting': 'GB',
                    'SVM': 'SVM', 'K-Nearest Neighbors': 'KNN'}
    means = [all_cv[k].mean() for k in model_names]
    stds  = [all_cv[k].std()  for k in model_names]
    bar_colors = [C_GOLD if k == best_name else C_NAVY for k in model_names]
    bars = ax.barh([short_names.get(k, k) for k in model_names], means,
                   xerr=stds, color=bar_colors, edgecolor='white', height=0.55,
                   error_kw=dict(ecolor=C_GRAY, capsize=4))
    for bar, val in zip(bars, means):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold', color=C_NAVY)
    ax.set_xlabel('Accuracy', fontsize=10)
    ax.set_xlim(0.7, 1.05)
    ax.legend(handles=[mpatches.Patch(color=C_GOLD, label='Best Model')], fontsize=9)

    # ── 9. Confusion Matrix ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    style_ax(ax, f'⑨ Confusion Matrix — {best_name}')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=1.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.tick_params(axis='x', rotation=30)

    # ── 10. ROC Curves (One-vs-Rest) ──────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    style_ax(ax, '⑩ ROC Curves (One-vs-Rest)')
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    for i, (sp, color) in enumerate(zip(le.classes_, SPECIES_COLORS)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{sp} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.legend(fontsize=9)

    # ── 11. Learning Curve ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    style_ax(ax, f'⑪ Learning Curve — {best_name}')
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_all, y_all,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    ax.plot(train_sizes, train_mean, color=C_NAVY, lw=2.5, marker='o',
            markersize=6, label='Train Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.12, color=C_NAVY)
    ax.plot(train_sizes, val_mean, color=C_RED, lw=2.5, marker='s',
            markersize=6, label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.12, color=C_RED)
    ax.set_xlabel('Training Set Size', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_ylim(0.7, 1.05)
    ax.legend(fontsize=9)

    # ── 12. Metrics Summary Card ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    ax.set_facecolor(C_NAVY)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall':    recall_score(y_test, y_pred, average='weighted'),
        'F1 Score':  f1_score(y_test, y_pred, average='weighted'),
    }
    ax.text(0.5, 0.93, '⑫ Best Model Metrics', ha='center', va='top',
            transform=ax.transAxes, fontsize=13, fontweight='bold', color=C_WHITE)
    ax.text(0.5, 0.81, best_name, ha='center', va='top',
            transform=ax.transAxes, fontsize=10, color=C_GOLD)
    for i, (metric, val) in enumerate(metrics.items()):
        yp = 0.65 - i * 0.13
        ax.text(0.12, yp, metric, ha='left', va='center',
                transform=ax.transAxes, fontsize=11, color=C_LIGHT)
        ax.text(0.88, yp, f'{val:.4f}', ha='right', va='center',
                transform=ax.transAxes, fontsize=15, fontweight='bold', color=C_CYAN)
    best_cv = all_cv.get(best_name, list(all_cv.values())[0])
    ax.text(0.5, 0.06, f'10-Fold CV: {best_cv.mean():.4f} ± {best_cv.std():.4f}',
            ha='center', va='bottom', transform=ax.transAxes, fontsize=10, color=C_GOLD)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f4f5f9')
    plt.close()
    print(f"📊 Dashboard saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/IRIS.csv')
    parser.add_argument('--model',  default='outputs/best_model.pkl')
    parser.add_argument('--output', default='outputs/iris_dashboard.png')
    args = parser.parse_args()
    generate_dashboard(args.data, args.model, args.output)


if __name__ == '__main__':
    main()
