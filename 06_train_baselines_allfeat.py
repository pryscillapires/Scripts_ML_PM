#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import argparse

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
# Defaults / feature sets
# ---------------------------
INPUT_DEFAULT = "out/ml/features_pm_real.csv"
TARGET = "y"

IC3_FEATURES = ["a_over_RH", "e", "i_deg"]

# ic6 = ic3 + “distâncias” (suspeito até testar leakage)
IC6_FEATURES = IC3_FEATURES + ["min_rP", "min_rM", "max_rB"]

# allfeat = SOMENTE features a priori (ajuste conforme seu CSV)
ALLFEAT_FEATURES = [
    "a_over_RH", "e", "i_deg",
    "srp_level", "Cr", "sense_pro", "regime_id"
]

FEATURE_SETS = {
    "ic3": IC3_FEATURES,
    "ic6": IC6_FEATURES,
    "allfeat": ALLFEAT_FEATURES,
}

RANDOM_STATE = 42
N_SPLITS = 5

RF_PARAMS = dict(
    n_estimators=800,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
)

HGB_PARAMS = dict(
    learning_rate=0.05,
    max_iter=600,
    max_depth=6,
    random_state=RANDOM_STATE,
)

CAL_METHOD = "isotonic"
CAL_HOLDOUT_FRAC = 0.20


# ---------------------------
# Utilities (iguais ao seu)
# ---------------------------
def detect_groups(df: pd.DataFrame):
    candidates = ["regime_id", "epoch", "epoch_id", "seed", "group_id", "run_id"]
    for c in candidates:
        if c in df.columns:
            g = np.asarray(df[c])
            n_unique = int(len(np.unique(g)))
            return g, c, n_unique
    return None, None, 0


def make_outer_splitter(y, groups, n_unique_groups):
    if groups is None or n_unique_groups < N_SPLITS:
        return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), "StratifiedKFold"
    return GroupKFold(n_splits=N_SPLITS), "GroupKFold"


def make_calibration_split(y_tr, groups_tr, n_unique_groups_tr):
    n_tr = len(y_tr)
    if groups_tr is not None and n_unique_groups_tr >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=CAL_HOLDOUT_FRAC, random_state=RANDOM_STATE)
        tr_idx, cal_idx = next(splitter.split(np.zeros(n_tr), y_tr, groups=groups_tr))
        return tr_idx, cal_idx, "GroupShuffleSplit"

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

    y_pred = (p_clip >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_0p5"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    metrics["acc_0p5"] = float((tp + tn) / (tp + tn + fp + fn))
    metrics["tpr_0p5"] = float(tp / (tp + fn + 1e-12))
    metrics["fpr_0p5"] = float(fp / (fp + tn + 1e-12))

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

    plt.figure()
    RocCurveDisplay.from_predictions(y_true, p_clip)
    plt.title(f"ROC (OOF) — {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"roc_{tag}.png", dpi=220)
    plt.close()

    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, p_clip)
    plt.title(f"Precision–Recall (OOF) — {tag}")
    plt.tight_layout()
    plt.savefig(outdir / f"pr_{tag}.png", dpi=220)
    plt.close()

    return metrics


def fit_base_estimator(base, X, y, tag, sample_weight=None):
    if tag == "hgb":
        if sample_weight is not None:
            base.fit(X, y, sample_weight=sample_weight)
        else:
            base.fit(X, y)
        return base
    base.fit(X, y)
    return base


def prefit_isotonic_calibrate(fitted_base, X_cal, y_cal):
    cal = CalibratedClassifierCV(fitted_base, method=CAL_METHOD, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def oof_predict_model(df, features, groups, n_unique_groups, tag):
    X = df[features].values
    y = df[TARGET].values.astype(int)

    splitter, split_name = make_outer_splitter(y, groups, n_unique_groups)
    p_oof = np.zeros(len(df), dtype=float)

    print(f"\n[{tag}] Outer CV splitter: {split_name}")

    if groups is None or split_name == "StratifiedKFold":
        split_iter = splitter.split(X, y)
    else:
        split_iter = splitter.split(X, y, groups=groups)

    fold = 0
    for tr, va in split_iter:
        fold += 1
        Xtr, ytr = X[tr], y[tr]
        Xva = X[va]

        if tag == "lr":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=6000, class_weight="balanced", solver="lbfgs"))
            ])
            model.fit(Xtr, ytr)
            p_oof[va] = model.predict_proba(Xva)[:, 1]
            print(f"[{tag}] fold {fold}/{N_SPLITS} done")
            continue

        groups_tr = None
        n_unique_groups_tr = 0
        if groups is not None:
            groups_tr = np.asarray(groups)[tr]
            n_unique_groups_tr = int(len(np.unique(groups_tr)))

        tr_fit_idx, tr_cal_idx, cal_split_name = make_calibration_split(ytr, groups_tr, n_unique_groups_tr)
        X_fit, y_fit = Xtr[tr_fit_idx], ytr[tr_fit_idx]
        X_cal, y_cal = Xtr[tr_cal_idx], ytr[tr_cal_idx]

        sw_fit = compute_sample_weight(class_weight="balanced", y=y_fit) if tag == "hgb" else None

        base = RandomForestClassifier(**RF_PARAMS) if tag == "rf" else HistGradientBoostingClassifier(**HGB_PARAMS)
        base = fit_base_estimator(base, X_fit, y_fit, tag=tag, sample_weight=sw_fit)
        cal = prefit_isotonic_calibrate(base, X_cal, y_cal)

        p_oof[va] = cal.predict_proba(Xva)[:, 1]
        print(f"[{tag}] fold {fold}/{N_SPLITS} done (cal split: {cal_split_name})")

    # Final model on full data (with one calibration holdout)
    if tag == "lr":
        final = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=6000, class_weight="balanced", solver="lbfgs"))
        ])
        final.fit(X, y)
        return p_oof, final, split_name

    groups_full = np.asarray(groups) if groups is not None else None
    n_unique_groups_full = int(len(np.unique(groups_full))) if groups_full is not None else 0

    fit_idx, cal_idx, cal_split_name = make_calibration_split(y, groups_full, n_unique_groups_full)
    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]

    sw_fit = compute_sample_weight(class_weight="balanced", y=y_fit) if tag == "hgb" else None
    base_final = RandomForestClassifier(**RF_PARAMS) if tag == "rf" else HistGradientBoostingClassifier(**HGB_PARAMS)
    base_final = fit_base_estimator(base_final, X_fit, y_fit, tag=tag, sample_weight=sw_fit)
    cal_final = prefit_isotonic_calibrate(base_final, X_cal, y_cal)

    print(f"[{tag}] final model trained with calibration holdout: {cal_split_name}")
    return p_oof, cal_final, split_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DEFAULT)
    ap.add_argument("--set", choices=list(FEATURE_SETS.keys()), default="allfeat")
    ap.add_argument("--permute_y", action="store_true")
    args = ap.parse_args()

    features = FEATURE_SETS[args.set]
    outdir = Path(f"out/ml/baselines_{args.set}")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # sanity
    missing = [c for c in features + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for set='{args.set}': {missing}")

    if args.permute_y:
        rng = np.random.default_rng(RANDOM_STATE)
        df[TARGET] = rng.permutation(df[TARGET].values)

    groups, group_col, n_unique = detect_groups(df)
    if group_col is None or n_unique < N_SPLITS:
        groups = None
        group_col = None
        n_unique = 0

    out = pd.DataFrame({"y": df[TARGET].astype(int)})

    metrics = {}
    split_used = {}

    # LR
    p_lr, lr_final, split_name = oof_predict_model(df, features, groups, n_unique, tag="lr")
    out["p_lr"] = p_lr
    metrics["lr"] = evaluate_and_plot(out["y"], out["p_lr"], f"lr_{args.set}", outdir)
    split_used["lr"] = split_name
    joblib.dump({"model": lr_final, "feature_cols": features}, outdir / f"lr_{args.set}.pkl")

    # RF
    p_rf, rf_final, split_name = oof_predict_model(df, features, groups, n_unique, tag="rf")
    out["p_rf"] = p_rf
    metrics["rf"] = evaluate_and_plot(out["y"], out["p_rf"], f"rf_{args.set}", outdir)
    split_used["rf"] = split_name
    joblib.dump({"model": rf_final, "feature_cols": features}, outdir / f"rf_{args.set}_cal.pkl")

    # HGB
    p_hgb, hgb_final, split_name = oof_predict_model(df, features, groups, n_unique, tag="hgb")
    out["p_hgb"] = p_hgb
    metrics["hgb"] = evaluate_and_plot(out["y"], out["p_hgb"], f"hgb_{args.set}", outdir)
    split_used["hgb"] = split_name
    joblib.dump({"model": hgb_final, "feature_cols": features}, outdir / f"hgb_{args.set}_cal.pkl")

    out.to_csv(outdir / f"oof_predictions_baselines_{args.set}.csv", index=False)

    payload = {
        "features": features,
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
    with open(outdir / f"metrics_baselines_{args.set}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nDone. Saved to: {outdir}")


if __name__ == "__main__":
    main()

