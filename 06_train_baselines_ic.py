#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_train_baselines_ic.py (group-safe)

Baselines for ML reviewers: compare model families on IC-only features.
Models:
  - Logistic Regression (scaled)
  - Random Forest + isotonic calibration (group-safe)
  - HistGradientBoosting + isotonic calibration (group-safe)

Uses OOF (out-of-fold) predictions with 5-fold CV.
If a group column is found and has enough unique groups, uses GroupKFold to avoid leakage.
Calibration is implemented in a group-safe way using a held-out calibration set + cv="prefit"
inside each outer fold.

Inputs:
  out/ml/features_pm_real.csv

Outputs (out/ml/baselines/):
  - oof_predictions_baselines_ic.csv
  - metrics_baselines_ic.json
  - lr_ic.pkl, rf_ic_cal.pkl, hgb_ic_cal.pkl
  - reliability_*.png, roc_*.png, pr_*.png
  - feature_importance_rf_ic_baseline.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold,
    StratifiedShuffleSplit, GroupShuffleSplit
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------
# Config
# ---------------------------
INPUT = "out/ml/features_pm_real.csv"
OUTDIR = Path("out/ml/baselines")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

IC_FEATURES = ["a_over_RH", "e", "i_deg"]
TARGET = "y"

# RF base
RF_PARAMS = dict(
    n_estimators=800,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
)

# HGB base (boosting)
HGB_PARAMS = dict(
    learning_rate=0.05,
    max_iter=600,
    max_depth=6,
    random_state=RANDOM_STATE,
)

CAL_METHOD = "isotonic"
CAL_HOLDOUT_FRAC = 0.20  # fraction used for calibration inside each outer fold


# ---------------------------
# Utilities
# ---------------------------
def detect_groups(df: pd.DataFrame):
    """Return (groups, group_col, n_unique) if a known group column exists, else (None, None, 0)."""
    candidates = ["regime_id", "epoch", "epoch_id", "seed", "group_id", "run_id"]
    for c in candidates:
        if c in df.columns:
            g = np.asarray(df[c])
            n_unique = int(len(np.unique(g)))
            return g, c, n_unique
    return None, None, 0


def make_outer_splitter(y, groups, n_unique_groups):
    """Choose outer CV splitter. GroupKFold only if enough unique groups."""
    if groups is None or n_unique_groups < N_SPLITS:
        return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), "StratifiedKFold"
    return GroupKFold(n_splits=N_SPLITS), "GroupKFold"


def make_calibration_split(y_tr, groups_tr, n_unique_groups_tr):
    """
    Make a single train/calibration split inside an outer fold.
    Group-aware when possible; else stratified.
    """
    n_tr = len(y_tr)
    n_cal = max(1, int(np.floor(CAL_HOLDOUT_FRAC * n_tr)))

    if groups_tr is not None and n_unique_groups_tr >= 2:
        # Group-aware holdout
        splitter = GroupShuffleSplit(n_splits=1, test_size=CAL_HOLDOUT_FRAC, random_state=RANDOM_STATE)
        split_iter = splitter.split(np.zeros(n_tr), y_tr, groups=groups_tr)
        tr_idx, cal_idx = next(split_iter)
        return tr_idx, cal_idx, "GroupShuffleSplit"

    # Stratified holdout
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=CAL_HOLDOUT_FRAC, random_state=RANDOM_STATE)
    tr_idx, cal_idx = next(splitter.split(np.zeros(n_tr), y_tr))
    return tr_idx, cal_idx, "StratifiedShuffleSplit"


def evaluate_and_plot(y_true, p_hat, tag, outdir: Path):
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat).astype(float)

    eps = 1e-12
    p_clip = np.clip(p_hat, eps, 1 - eps)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, p_clip)),
        "ap": float(average_precision_score(y_true, p_clip)),
        "brier": float(brier_score_loss(y_true, p_clip)),
        "logloss": float(log_loss(y_true, p_clip)),
    }

    # Confusion at 0.5
    y_pred = (p_clip >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_0p5"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    metrics["acc_0p5"] = float((tp + tn) / (tp + tn + fp + fn))
    metrics["tpr_0p5"] = float(tp / (tp + fn + 1e-12))
    metrics["fpr_0p5"] = float(fp / (fp + tn + 1e-12))

    # Reliability
    frac_pos, mean_pred = calibration_curve(y_true, p_clip, n_bins=12, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Reliability (OOF) — {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"reliability_{tag}.png", dpi=220)
    plt.close()

    # ROC
    plt.figure()
    RocCurveDisplay.from_predictions(y_true, p_clip)
    plt.title(f"ROC (OOF) — {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"roc_{tag}.png", dpi=220)
    plt.close()

    # PR
    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, p_clip)
    plt.title(f"Precision–Recall (OOF) — {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"pr_{tag}.png", dpi=220)
    plt.close()

    return metrics


def plot_rf_importance(cal_model, feature_names, outdir: Path):
    """
    Average RF feature_importances_ across calibrated estimators if available.
    Works for both cv="prefit" (single calibrator) and regular calibrated_classifiers_ lists.
    """
    importances = []

    if hasattr(cal_model, "calibrated_classifiers_"):
        for cc in cal_model.calibrated_classifiers_:
            rf = getattr(cc, "estimator", None)
            if rf is None:
                rf = getattr(cc, "base_estimator", None)
            if rf is not None and hasattr(rf, "feature_importances_"):
                importances.append(rf.feature_importances_)

    if not importances:
        return

    imp = np.mean(np.vstack(importances), axis=0)
    order = np.argsort(imp)[::-1]

    plt.figure()
    plt.bar(np.array(feature_names)[order], imp[order])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean feature importance")
    plt.title("RF feature importance (IC-only)")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance_rf_ic_baseline.png", dpi=220)
    plt.close()


def fit_base_estimator(base, X, y, tag, sample_weight=None):
    """Fit base estimator with optional sample_weight when supported."""
    if tag == "hgb":
        # HGB supports sample_weight
        if sample_weight is not None:
            base.fit(X, y, sample_weight=sample_weight)
        else:
            base.fit(X, y)
        return base

    # RF: class_weight handles imbalance; still okay to pass sample_weight but not needed.
    base.fit(X, y)
    return base


def prefit_isotonic_calibrate(fitted_base, X_cal, y_cal):
    """
    Calibrate a fitted classifier using isotonic regression on a held-out calibration set.
    Uses CalibratedClassifierCV(cv="prefit") to avoid any internal CV leakage issues.
    """
    cal = CalibratedClassifierCV(fitted_base, method=CAL_METHOD, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def oof_predict_model(df, features, groups, group_col, n_unique_groups, tag):
    X = df[features].values
    y = df[TARGET].values.astype(int)

    splitter, split_name = make_outer_splitter(y, groups, n_unique_groups)
    p_oof = np.zeros(len(df), dtype=float)

    print(f"\n[{tag}] Outer CV splitter: {split_name}")

    # Define base model builders
    if tag == "lr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=6000,
                class_weight="balanced",
                solver="lbfgs",
            ))
        ])
        cal_final = None

    elif tag == "rf":
        model = None  # handled manually
        cal_final = None

    elif tag == "hgb":
        model = None  # handled manually
        cal_final = None

    else:
        raise ValueError("unknown tag")

    # Outer split iterator
    if groups is None or split_name == "StratifiedKFold":
        split_iter = splitter.split(X, y)
    else:
        split_iter = splitter.split(X, y, groups=groups)

    fold = 0
    for tr, va in split_iter:
        fold += 1
        Xtr, ytr = X[tr], y[tr]
        Xva = X[va]

        # LR: straight probas (already probabilistic)
        if tag == "lr":
            model.fit(Xtr, ytr)
            p_oof[va] = model.predict_proba(Xva)[:, 1]
            print(f"[{tag}] fold {fold}/{N_SPLITS} done")
            continue

        # RF/HGB: group-safe calibration within fold
        groups_tr = None
        n_unique_groups_tr = 0
        if groups is not None:
            groups_tr = np.asarray(groups)[tr]
            n_unique_groups_tr = int(len(np.unique(groups_tr)))

        # Inner split: base-fit set + calibration set
        tr_fit_idx, tr_cal_idx, cal_split_name = make_calibration_split(ytr, groups_tr, n_unique_groups_tr)

        X_fit, y_fit = Xtr[tr_fit_idx], ytr[tr_fit_idx]
        X_cal, y_cal = Xtr[tr_cal_idx], ytr[tr_cal_idx]

        # sample_weight for HGB only (since it lacks class_weight)
        if tag == "hgb":
            sw_fit = compute_sample_weight(class_weight="balanced", y=y_fit)
        else:
            sw_fit = None

        if tag == "rf":
            base = RandomForestClassifier(**RF_PARAMS)
        else:
            base = HistGradientBoostingClassifier(**HGB_PARAMS)

        base = fit_base_estimator(base, X_fit, y_fit, tag=tag, sample_weight=sw_fit)
        cal = prefit_isotonic_calibrate(base, X_cal, y_cal)

        p_oof[va] = cal.predict_proba(Xva)[:, 1]

        print(f"[{tag}] fold {fold}/{N_SPLITS} done (cal split: {cal_split_name})")

    # Fit final model on full data (still group-safe via holdout calibration)
    if tag == "lr":
        final = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=6000,
                class_weight="balanced",
                solver="lbfgs",
            ))
        ])
        final.fit(X, y)
        return p_oof, final, split_name

    # Final calibrated models using a single group-safe (or stratified) holdout
    groups_full = None
    n_unique_groups_full = 0
    if groups is not None:
        groups_full = np.asarray(groups)
        n_unique_groups_full = int(len(np.unique(groups_full)))

    fit_idx, cal_idx, cal_split_name = make_calibration_split(y, groups_full, n_unique_groups_full)
    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    if tag == "hgb":
        sw_fit = compute_sample_weight(class_weight="balanced", y=y_fit)
    else:
        sw_fit = None

    if tag == "rf":
        base_final = RandomForestClassifier(**RF_PARAMS)
    else:
        base_final = HistGradientBoostingClassifier(**HGB_PARAMS)

    base_final = fit_base_estimator(base_final, X_fit, y_fit, tag=tag, sample_weight=sw_fit)
    cal_final = prefit_isotonic_calibrate(base_final, X_cal, y_cal)

    print(f"[{tag}] final model trained with calibration holdout: {cal_split_name}")

    return p_oof, cal_final, split_name


def main():
    df = pd.read_csv(INPUT)

    # sanity
    for c in IC_FEATURES + [TARGET]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    groups, group_col, n_unique = detect_groups(df)
    if group_col is None:
        print("No group column detected -> using StratifiedKFold.")
    else:
        if n_unique < N_SPLITS:
            print(f"Detected group column '{group_col}' but only {n_unique} unique groups (<{N_SPLITS}) -> using StratifiedKFold.")
            groups = None
            group_col = None
            n_unique = 0
        else:
            print(f"Detected group column '{group_col}' with {n_unique} unique groups -> using GroupKFold for leakage control.")

    out = pd.DataFrame({"y": df[TARGET].astype(int)})

    metrics = {}
    split_used = {}

    # Logistic
    p_lr, lr_final, split_name = oof_predict_model(df, IC_FEATURES, groups, group_col, n_unique, tag="lr")
    out["p_lr"] = p_lr
    metrics["lr"] = evaluate_and_plot(out["y"], out["p_lr"], "lr_ic", OUTDIR)
    split_used["lr"] = split_name
    joblib.dump({"model": lr_final, "feature_cols": IC_FEATURES}, OUTDIR / "lr_ic.pkl")

    # RF calibrated (group-safe)
    p_rf, rf_final, split_name = oof_predict_model(df, IC_FEATURES, groups, group_col, n_unique, tag="rf")
    out["p_rf"] = p_rf
    metrics["rf"] = evaluate_and_plot(out["y"], out["p_rf"], "rf_ic", OUTDIR)
    split_used["rf"] = split_name
    joblib.dump({"model": rf_final, "feature_cols": IC_FEATURES}, OUTDIR / "rf_ic_cal.pkl")
    plot_rf_importance(rf_final, IC_FEATURES, OUTDIR)

    # HGB calibrated (group-safe)
    p_hgb, hgb_final, split_name = oof_predict_model(df, IC_FEATURES, groups, group_col, n_unique, tag="hgb")
    out["p_hgb"] = p_hgb
    metrics["hgb"] = evaluate_and_plot(out["y"], out["p_hgb"], "hgb_ic", OUTDIR)
    split_used["hgb"] = split_name
    joblib.dump({"model": hgb_final, "feature_cols": IC_FEATURES}, OUTDIR / "hgb_ic_cal.pkl")

    # Save OOF predictions + metrics
    out.to_csv(OUTDIR / "oof_predictions_baselines_ic.csv", index=False)

    payload = {
        "features": IC_FEATURES,
        "splitter": split_used,
        "metrics": metrics,
        "group_col": group_col,
        "note": (
            "OOF probabilities with 5-fold CV. "
            "RF and HGB are isotonic-calibrated in a group-safe way using a held-out calibration split "
            "within each outer fold (cv='prefit'). "
            "LR uses StandardScaler + balanced class_weight."
        )
    }
    with open(OUTDIR / "metrics_baselines_ic.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nDone. Saved:")
    print("  out/ml/baselines/oof_predictions_baselines_ic.csv")
    print("  out/ml/baselines/metrics_baselines_ic.json")
    print("  models: lr_ic.pkl, rf_ic_cal.pkl, hgb_ic_cal.pkl")
    print("  plots: reliability_*.png, roc_*.png, pr_*.png, feature_importance_rf_ic_baseline.png")


if __name__ == "__main__":
    main()

