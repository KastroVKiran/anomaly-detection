"""
AIOps Anomaly Detection — Complete ML Training Script
Produces all results, metrics, and saves output to results.txt
"""

import pandas as pd
import numpy as np
import time
import sys
from sklearn.ensemble import (IsolationForest,
                               RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.model_selection import (train_test_split,
                                      cross_val_score,
                                      StratifiedKFold)
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              roc_auc_score,
                              f1_score,
                              precision_score,
                              recall_score)
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# ── Output to both screen and file simultaneously ─────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

output_file = open('results.txt', 'w')
sys.stdout   = Tee(sys.stdout, output_file)

# ── Load data ─────────────────────────────────────────────
print("=" * 60)
print("AIOps ML Model Training and Evaluation")
print("=" * 60)

CSV_PATH = "aiops_dataset_v2.csv"
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\nDataset loaded: {CSV_PATH}")
print(f"Total rows    : {len(df):,}")
print(f"Columns       : {len(df.columns)}")
print(f"Time range    : {df['timestamp'].min()} "
      f"to {df['timestamp'].max()}")
print(f"Duration      : "
      f"{df['timestamp'].max()-df['timestamp'].min()}")

# ── Features and labels ───────────────────────────────────
FEATURES = [
    'cpu_usage_rate',
    'memory_mb',
    'restart_count',
    'restart_delta',
    'node_load1',
    'network_rx_bytes_rate',
    'is_restarting',
    'high_restart',
    'is_stress_pod',
    'cpu_rolling_mean',
]

X = df[FEATURES].fillna(0)
y = df['label']

print(f"\nFeatures used : {len(FEATURES)}")
print(f"Normal rows   : {(y==0).sum():,} "
      f"({(y==0).mean()*100:.1f}%)")
print(f"Anomaly rows  : {(y==1).sum():,} "
      f"({(y==1).mean()*100:.1f}%)")

# ── Train/test split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"\nTrain set     : {len(X_train):,} rows "
      f"({y_train.sum():,} anomalies)")
print(f"Test set      : {len(X_test):,} rows "
      f"({y_test.sum():,} anomalies)")

# Scale for Isolation Forest
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)
X_te_sc   = scaler.transform(X_test)

# ── Cross-validation setup ────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────
# MODEL 1: THRESHOLD BASELINE
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1: Threshold Baseline (no ML)")
print("=" * 60)
print("Rule: anomaly if cpu_usage_rate > 0.5 OR "
      "restart_count > 3")

t0      = time.time()
th_pred = (
    (X_test['cpu_usage_rate'] > 0.5) |
    (X_test['restart_count']  > 3)
).astype(int)
th_train_time = time.time() - t0

t0     = time.time()
_      = (
    (X_test['cpu_usage_rate'] > 0.5) |
    (X_test['restart_count']  > 3)
).astype(int)
th_inf = (time.time() - t0) / len(X_test) * 1000

th_p  = precision_score(y_test, th_pred)
th_r  = recall_score(y_test, th_pred)
th_f1 = f1_score(y_test, th_pred)

print(f"Precision : {th_p:.4f}")
print(f"Recall    : {th_r:.4f}")
print(f"F1 Score  : {th_f1:.4f}")
print(f"Train time: {th_train_time:.4f}s")
print(f"Inf time  : {th_inf:.4f}ms per sample")

# ─────────────────────────────────────────────────────────
# MODEL 2: ISOLATION FOREST
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: Isolation Forest (Unsupervised ML)")
print("=" * 60)
print("Config: n_estimators=200, contamination=0.5, "
      "random_state=42")

t0  = time.time()
iso = IsolationForest(
    n_estimators=200,
    contamination=0.5,
    random_state=42
)
iso.fit(X_scaled)
iso_train_time = time.time() - t0

t0       = time.time()
iso_pred = (iso.predict(X_te_sc) == -1).astype(int)
iso_inf  = (time.time() - t0) / len(X_test) * 1000

iso_p  = precision_score(y_test, iso_pred)
iso_r  = recall_score(y_test, iso_pred)
iso_f1 = f1_score(y_test, iso_pred)

print(f"Precision : {iso_p:.4f}")
print(f"Recall    : {iso_r:.4f}")
print(f"F1 Score  : {iso_f1:.4f}")
print(f"Train time: {iso_train_time:.4f}s")
print(f"Inf time  : {iso_inf:.4f}ms per sample")

# ─────────────────────────────────────────────────────────
# MODEL 3: RANDOM FOREST
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 3: Random Forest (Supervised ML)")
print("=" * 60)
print("Config: n_estimators=200, class_weight=balanced, "
      "random_state=42")

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
rf_train_time = time.time() - t0

t0      = time.time()
rf_pred = rf.predict(X_test)
rf_inf  = (time.time() - t0) / len(X_test) * 1000
rf_prob = rf.predict_proba(X_test)[:, 1]

rf_p   = precision_score(y_test, rf_pred)
rf_r   = recall_score(y_test, rf_pred)
rf_f1  = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"Precision : {rf_p:.4f}")
print(f"Recall    : {rf_r:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_auc:.4f}")
print(f"Train time: {rf_train_time:.4f}s")
print(f"Inf time  : {rf_inf:.4f}ms per sample")

rf_cv_scores = cross_val_score(
    rf, X, y, cv=cv, scoring='f1')
print(f"5-Fold CV F1: {rf_cv_scores.mean():.4f} "
      f"+/- {rf_cv_scores.std():.4f}")
print(f"CV scores   : "
      f"{[round(s,4) for s in rf_cv_scores]}")

print("\nDetailed classification report:")
print(classification_report(
    y_test, rf_pred,
    target_names=['Normal', 'Anomaly'],
    digits=4))

print("Confusion matrix:")
cm_rf = confusion_matrix(y_test, rf_pred)
print(f"  TN={cm_rf[0,0]}  FP={cm_rf[0,1]}")
print(f"  FN={cm_rf[1,0]}  TP={cm_rf[1,1]}")

print("\nFeature Importance (root cause ranking):")
fi = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)
for feat, imp in fi.items():
    bar = "#" * int(imp * 50)
    print(f"  {feat:<30}: {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────
# MODEL 4: GRADIENT BOOSTING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 4: Gradient Boosting (Supervised ML)")
print("=" * 60)
print("Config: n_estimators=200, learning_rate=0.1, "
      "max_depth=4, random_state=42")

t0 = time.time()
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
gb.fit(X_train, y_train)
gb_train_time = time.time() - t0

t0      = time.time()
gb_pred = gb.predict(X_test)
gb_inf  = (time.time() - t0) / len(X_test) * 1000
gb_prob = gb.predict_proba(X_test)[:, 1]

gb_p   = precision_score(y_test, gb_pred)
gb_r   = recall_score(y_test, gb_pred)
gb_f1  = f1_score(y_test, gb_pred)
gb_auc = roc_auc_score(y_test, gb_prob)

print(f"Precision : {gb_p:.4f}")
print(f"Recall    : {gb_r:.4f}")
print(f"F1 Score  : {gb_f1:.4f}")
print(f"ROC-AUC   : {gb_auc:.4f}")
print(f"Train time: {gb_train_time:.4f}s")
print(f"Inf time  : {gb_inf:.4f}ms per sample")

gb_cv_scores = cross_val_score(
    gb, X, y, cv=cv, scoring='f1')
print(f"5-Fold CV F1: {gb_cv_scores.mean():.4f} "
      f"+/- {gb_cv_scores.std():.4f}")

print("\nDetailed classification report:")
print(classification_report(
    y_test, gb_pred,
    target_names=['Normal', 'Anomaly'],
    digits=4))

print("Confusion matrix:")
cm_gb = confusion_matrix(y_test, gb_pred)
print(f"  TN={cm_gb[0,0]}  FP={cm_gb[0,1]}")
print(f"  FN={cm_gb[1,0]}  TP={cm_gb[1,1]}")

# ─────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE TESTS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)

# McNemar test (RF vs GB on test set predictions)
b = int(np.sum((rf_pred == 0) & (gb_pred == 1)))
c = int(np.sum((rf_pred == 1) & (gb_pred == 0)))
print(f"McNemar test — RF vs Gradient Boosting:")
print(f"  b (RF wrong, GB right) = {b}")
print(f"  c (RF right, GB wrong) = {c}")
if (b + c) > 0:
    chi2  = (abs(b - c) - 1) ** 2 / (b + c)
    p_mc  = 1 - stats.chi2.cdf(chi2, 1)
    print(f"  chi2 = {chi2:.4f}")
    print(f"  p    = {p_mc:.4f}")
    if p_mc < 0.05:
        print("  Result: SIGNIFICANT (p < 0.05)")
    else:
        print("  Result: not significant (p >= 0.05)")

# Paired t-test on 5-fold CV F1 scores
t_stat, p_t = stats.ttest_rel(rf_cv_scores, gb_cv_scores)
print(f"\nPaired t-test — RF CV vs GB CV F1 scores:")
print(f"  t = {t_stat:.4f}")
print(f"  p = {p_t:.4f}")
if p_t < 0.05:
    print("  Result: SIGNIFICANT (p < 0.05)")
else:
    print("  Result: not significant (p >= 0.05)")

# ─────────────────────────────────────────────────────────
# PER-ANOMALY-TYPE BREAKDOWN
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PER-ANOMALY-TYPE BREAKDOWN (Gradient Boosting)")
print("=" * 60)

df_test             = X_test.copy()
df_test['label']    = y_test.values
df_test['gb_pred']  = gb_pred
df_test['rf_pred']  = rf_pred
df_test['atype']    = df.loc[X_test.index,
                              'anomaly_type'].values

print(f"\n{'Anomaly type':<16} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8} {'Count':>8}")
print("-" * 55)
for atype in ['cpu_spike', 'memory_spike', 'crash_loop']:
    sub = df_test[
        df_test['atype'].isin([atype, 'normal'])
    ]
    if len(sub) == 0:
        continue
    yt = sub['label']
    yp = sub['gb_pred']
    if len(yt.unique()) < 2:
        continue
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    f  = f1_score(yt, yp, zero_division=0)
    n  = int((yt == 1).sum())
    print(f"{atype:<16} {p:>10.4f} {r:>8.4f} "
          f"{f:>8.4f} {n:>8}")

# ─────────────────────────────────────────────────────────
# HYPERPARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HYPERPARAMETER SENSITIVITY ANALYSIS")
print("=" * 60)

print("\nRandom Forest — varying n_estimators:")
print(f"{'n_estimators':<15} {'F1 mean':>10} {'F1 std':>10}")
print("-" * 38)
for n in [50, 100, 200]:
    m = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        class_weight='balanced'
    )
    s = cross_val_score(m, X, y, cv=3, scoring='f1')
    print(f"{n:<15} {s.mean():>10.4f} {s.std():>10.4f}")

print("\nGradient Boosting — varying learning_rate:")
print(f"{'learning_rate':<15} {'F1 mean':>10} {'F1 std':>10}")
print("-" * 38)
for lr in [0.05, 0.10, 0.20]:
    m = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=lr,
        max_depth=4,
        random_state=42
    )
    s = cross_val_score(m, X, y, cv=3, scoring='f1')
    print(f"{lr:<15} {s.mean():>10.4f} {s.std():>10.4f}")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY TABLE — COPY THIS INTO YOUR PAPER")
print("=" * 60)
print(f"\n{'Model':<22} {'Prec':>7} {'Rec':>7} "
      f"{'F1':>7} {'AUC':>7} {'Train(s)':>9} {'Inf(ms)':>8}")
print("-" * 70)

rows = [
    ("Threshold",        th_p,  th_r,  th_f1,  None,   th_train_time,  th_inf),
    ("Isolation Forest", iso_p, iso_r, iso_f1, None,   iso_train_time, iso_inf),
    ("Random Forest",    rf_p,  rf_r,  rf_f1,  rf_auc, rf_train_time,  rf_inf),
    ("Gradient Boosting",gb_p,  gb_r,  gb_f1,  gb_auc, gb_train_time,  gb_inf),
]
for name, p, r, f, a, tr, ti in rows:
    a_s = f"{a:.4f}" if a else "   N/A"
    print(f"{name:<22} {p:>7.4f} {r:>7.4f} "
          f"{f:>7.4f} {a_s:>7} {tr:>9.2f} {ti:>8.4f}")

print(f"\nRandom Forest 5-fold CV : "
      f"{rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")
print(f"Gradient Boost 5-fold CV: "
      f"{gb_cv_scores.mean():.4f} +/- {gb_cv_scores.std():.4f}")

print("\n" + "=" * 60)
print("DONE — Results saved to results.txt")
print("=" * 60)

output_file.close()
