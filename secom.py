"""SECOM wafer pass/fail detection.

Isolation Forest vs class-weighted LogReg / XGBoost, stratified 5-fold CV,
SHAP ranking on the best model.

Three stages, each reading and writing files so they can run as separate
Airflow tasks (see airflow/dags/secom_dag.py) or all at once via main():
  stage_clean    data/secom.data -> data/processed/{features,labels}.csv
  stage_cv       processed       -> results/oof_scores.csv, metrics.json, pr_curves.png
  stage_explain  processed + metrics.json -> metrics.json (top sensors), shap_top20.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PROC = DATA / "processed"
OUT = ROOT / "results"
SEED = 42
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"


def download(force=False):
    """Fetch the two UCI files if missing. Returns the paths."""
    import urllib.request
    DATA.mkdir(exist_ok=True)
    paths = []
    for name in ("secom.data", "secom_labels.data"):
        p = DATA / name
        if force or not p.exists():
            urllib.request.urlretrieve(UCI + name, p)
        paths.append(p)
    return paths


def load():
    X = pd.read_csv(DATA / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    y = pd.read_csv(DATA / "secom_labels.data", sep=r"\s+", header=None, usecols=[0])[0]
    y = (y == 1).astype(int)  # 1 = fail (positive class), -1 = pass
    X.columns = [f"s{i}" for i in range(X.shape[1])]
    return X, y


def clean(X):
    """Drop sensors that are >50% missing or constant. Imputation happens inside CV folds."""
    missing = X.isna().mean()
    X = X.loc[:, missing <= 0.5]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return X


def recall_at_fpr(y, score, fpr_target=0.10):
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= fpr_target].max())


def models(n_pos, n_neg):
    spw = n_neg / n_pos
    return {
        "isolation_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            IsolationForest(n_estimators=500, contamination="auto", random_state=SEED),
        ),
        "logreg_weighted": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.05, max_iter=2000),
        ),
        "xgboost_weighted": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.5, scale_pos_weight=spw, eval_metric="aucpr",
                          random_state=SEED, n_jobs=4),
        ),
    }


def score(pipe, X):
    est = pipe[-1]
    if isinstance(est, IsolationForest):
        return -pipe.decision_function(X)  # higher = more anomalous
    return pipe.predict_proba(X)[:, 1]


# ---------------- stages ----------------

def stage_clean():
    X, y = load()
    X = clean(X)
    PROC.mkdir(parents=True, exist_ok=True)
    X.to_csv(PROC / "features.csv", index=False)
    y.to_csv(PROC / "labels.csv", index=False, header=["fail"])
    summary = {"rows": int(len(X)), "features": int(X.shape[1]), "fails": int(y.sum()), "fail_rate": round(float(y.mean()), 4)}
    print(f"rows={len(X)} features={X.shape[1]} fails={int(y.sum())} ({y.mean():.1%})")
    return summary


def _load_processed():
    X = pd.read_csv(PROC / "features.csv")
    y = pd.read_csv(PROC / "labels.csv")["fail"]
    return X, y


def stage_cv():
    X, y = _load_processed()
    OUT.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {name: np.zeros(len(y)) for name in models(1, 1)}
    for tr, te in cv.split(X, y):
        for name, build in models(y.iloc[tr].sum(), (1 - y.iloc[tr]).sum()).items():
            pipe = build()
            if name == "isolation_forest":
                pipe.fit(X.iloc[tr][y.iloc[tr] == 0])  # unsupervised: fit on passes only
            else:
                pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[name][te] = score(pipe, X.iloc[te])

    metrics = {"baseline_pr_auc_random": float(y.mean())}
    plt.figure(figsize=(6, 5))
    for name, s in oof.items():
        metrics[name] = {
            "pr_auc": round(float(average_precision_score(y, s)), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4),
            "recall_at_10pct_fpr": round(recall_at_fpr(y, s, 0.10), 4),
        }
        p, r, _ = precision_recall_curve(y, s)
        plt.plot(r, p, label=f"{name} (AP={metrics[name]['pr_auc']:.3f})")
    plt.axhline(y.mean(), ls="--", c="gray", label=f"random (AP={y.mean():.3f})")
    plt.xlabel("Recall (fails caught)"); plt.ylabel("Precision"); plt.title("SECOM fail detection, 5-fold OOF")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "pr_curves.png", dpi=130)

    metrics["best_model"] = max((k for k in oof if k != "isolation_forest"), key=lambda k: metrics[k]["pr_auc"])
    pd.DataFrame(oof).to_csv(OUT / "oof_scores.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def stage_explain():
    import shap
    X, y = _load_processed()
    metrics = json.loads((OUT / "metrics.json").read_text())
    best = metrics["best_model"]
    pipe = models(y.sum(), (1 - y).sum())[best]()
    pipe.fit(X, y)
    Xi = pd.DataFrame(pipe[0].transform(X), columns=X.columns)
    if best == "xgboost_weighted":
        sv = shap.TreeExplainer(pipe[-1]).shap_values(Xi)
    else:
        Xs = pd.DataFrame(pipe[1].transform(Xi), columns=X.columns)
        sv = shap.LinearExplainer(pipe[-1], Xs).shap_values(Xs)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    metrics["top_10_sensors"] = imp.head(10).round(4).to_dict()
    plt.figure()
    shap.summary_plot(sv, Xi, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(OUT / "shap_top20.png", dpi=130)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics["top_10_sensors"]


def main():
    download()
    stage_clean()
    stage_cv()
    stage_explain()
    print((OUT / "metrics.json").read_text())


if __name__ == "__main__":
    main()
