"""
ML Classification Project
Churn Prediction using Logistic Regression vs Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC CHURN DATASET
# ─────────────────────────────────────────────
np.random.seed(42)

n_samples = 1000

X_raw, y = make_classification(
    n_samples=n_samples,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.70, 0.30],   # 70% not-churned, 30% churned (realistic imbalance)
    flip_y=0.03,
    random_state=42
)

feature_names = [
    'tenure_months', 'monthly_charges', 'total_charges',
    'num_services', 'support_calls', 'contract_length',
    'internet_speed', 'payment_delay', 'satisfaction_score', 'age'
]

df = pd.DataFrame(X_raw, columns=feature_names)
df['churn'] = y

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.describe().round(2))
print(f"\nClass distribution:\n{df['churn'].value_counts()}")
print(f"Churn rate: {df['churn'].mean():.1%}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING & SPLITS
# ─────────────────────────────────────────────
X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)

lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)   # RF doesn't need scaling

# ─────────────────────────────────────────────
# 4. PREDICTIONS & METRICS
# ─────────────────────────────────────────────
def evaluate(model, X_eval, y_eval, name):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_eval, y_pred), 4),
        'Precision': round(precision_score(y_eval, y_pred), 4),
        'Recall':    round(recall_score(y_eval, y_pred), 4),
        'F1':        round(f1_score(y_eval, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_eval, y_prob), 4),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

lr_metrics = evaluate(lr, X_test_scaled, y_test, 'Logistic Regression')
rf_metrics = evaluate(rf, X_test,        y_test, 'Random Forest')

metrics_df = pd.DataFrame([
    {k: v for k, v in lr_metrics.items() if k not in ('y_pred','y_prob')},
    {k: v for k, v in rf_metrics.items() if k not in ('y_pred','y_prob')},
])

print("\n" + "=" * 60)
print("TEST-SET METRICS")
print("=" * 60)
print(metrics_df.to_string(index=False))

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

lr_cv = cross_validate(lr, X_train_scaled, y_train, cv=cv, scoring=cv_scoring)
rf_cv = cross_validate(rf, X_train,        y_train, cv=cv, scoring=cv_scoring)

cv_results = {}
for metric in cv_scoring:
    key = f'test_{metric}'
    cv_results[metric] = {
        'LR mean': round(lr_cv[key].mean(), 4),
        'LR std':  round(lr_cv[key].std(),  4),
        'RF mean': round(rf_cv[key].mean(), 4),
        'RF std':  round(rf_cv[key].std(),  4),
    }

cv_df = pd.DataFrame(cv_results).T
print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)
print(cv_df.to_string())

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
palette = {'LR': '#4A90D9', 'RF': '#E67E22', 'accent': '#2ECC71', 'bg': '#F8F9FA'}
plt.style.use('seaborn-v0_8-whitegrid')

# ── Plot A: Confusion Matrices ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, m, Xs, name in zip(
    axes,
    [lr_metrics, rf_metrics],
    [X_test_scaled, X_test],
    ['Logistic Regression', 'Random Forest']
):
    cm = confusion_matrix(y_test, m['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues', linewidths=0.5,
                xticklabels=['No Churn','Churn'],
                yticklabels=['No Churn','Churn'])
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
fig.suptitle('Confusion Matrices – Test Set', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/ml_project/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")

# ── Plot B: ROC Curves ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for m, label, color in [
    (lr_metrics, 'Logistic Regression', palette['LR']),
    (rf_metrics, 'Random Forest',       palette['RF']),
]:
    fpr, tpr, _ = roc_curve(y_test, m['y_prob'])
    ax.plot(fpr, tpr, label=f"{label}  (AUC={m['ROC-AUC']:.3f})", lw=2.2, color=color)
ax.plot([0,1],[0,1],'--', color='gray', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves – Logistic Regression vs Random Forest', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig('/home/claude/ml_project/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# ── Plot C: Metrics Bar Chart ────────────────────
metric_cols = ['Accuracy','Precision','Recall','F1','ROC-AUC']
x = np.arange(len(metric_cols)); width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
lr_vals = [lr_metrics[m] for m in metric_cols]
rf_vals = [rf_metrics[m] for m in metric_cols]
bars1 = ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', color=palette['LR'], edgecolor='white')
bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest',       color=palette['RF'], edgecolor='white')
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison – All Metrics', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ml_project/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: metrics_comparison.png")

# ── Plot D: Feature Importances (RF) ────────────
importances = rf.feature_importances_
feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors = [palette['RF'] if v > feat_df['importance'].median() else '#BDC3C7' for v in feat_df['importance']]
ax.barh(feat_df['feature'], feat_df['importance'], color=colors, edgecolor='white')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Random Forest – Feature Importances', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/ml_project/feature_importances.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importances.png")

# ── Plot E: CV Scores Box Plot ───────────────────
cv_plot_data = {}
for metric in ['accuracy','f1','roc_auc']:
    cv_plot_data[('LR', metric)] = lr_cv[f'test_{metric}']
    cv_plot_data[('RF', metric)] = rf_cv[f'test_{metric}']

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, metric in zip(axes, ['accuracy','f1','roc_auc']):
    data_to_plot = [lr_cv[f'test_{metric}'], rf_cv[f'test_{metric}']]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.45,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(palette['LR'])
    bp['boxes'][1].set_facecolor(palette['RF'])
    ax.set_xticklabels(['Log. Reg.', 'Rand. Forest'], fontsize=10)
    ax.set_title(metric.replace('_',' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
fig.suptitle('5-Fold Cross-Validation Score Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/ml_project/cv_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: cv_boxplots.png")

# ── Plot F: Class Distribution ───────────────────
fig, ax = plt.subplots(figsize=(5, 4))
vals = y.value_counts()
ax.bar(['No Churn', 'Churn'], vals.values, color=[palette['LR'], palette['RF']], edgecolor='white', width=0.5)
for i, v in enumerate(vals.values):
    ax.text(i, v + 8, str(v), ha='center', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', fontsize=11)
plt.tight_layout()
plt.savefig('/home/claude/ml_project/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: class_distribution.png")

# ─────────────────────────────────────────────
# 7. SAVE METRICS TO CSV
# ─────────────────────────────────────────────
metrics_df.to_csv('/home/claude/ml_project/test_metrics.csv', index=False)
cv_df.to_csv('/home/claude/ml_project/cv_metrics.csv')
print("\nSaved: test_metrics.csv, cv_metrics.csv")
print("\n✅ All outputs generated successfully.")

# Return for notebook use
METRICS = {
    'lr': lr_metrics, 'rf': rf_metrics,
    'metrics_df': metrics_df, 'cv_df': cv_df,
}
