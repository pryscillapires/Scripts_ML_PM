#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_train_calibrate_v2.py

Leakage-safe ML surrogate for Patroclus–Menoetius stability with selectable mode.

Modes:
- pro   : train/eval only prograde dataset
- retro : train/eval only retrograde dataset
- both  : combine pro+retro and include sense_pro as a feature

Always:
- Features are IC-only (a_over_RH, e, i_eff_deg [+ sense_pro if mode=both])
- Drops leaky post-integration columns (status, t_end, min_r*, max_r*, n_steps*, etc.)
- GroupKFold on coarse tiles in (a,e,i_eff) to avoid neighbor leakage
- Calibration uses held-out GROUPS from the training fold

Outputs per mode in outdir/<mode>/:
- oof_predictions.csv
- metrics.json
- models.pkl
- F_reliability.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix
)

import joblib


LEAKY_COLS_SUBSTR = ["t_end", "t_over", "min_r", "max_r", "n_steps", "status"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    i = df["i_deg"].astype(float).to_numpy()
    df["i_eff_deg"] = np.where(i <= 90.0, i, 180.0 - i)
    df["y_stable"] = (df["status"] == "stable").astype(int)
    return df


def drop_leaky_columns(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    cols = list(df.columns)
    drop = []
    for c in cols:
        if c in keep_cols:
            continue
        lc = c.lower()
        if any(sub in lc for sub in LEAKY_COLS_SUBSTR):
            drop.append(c)
    return df.drop(columns=drop, errors="ignore")


def make_tile_groups(df: pd.DataFrame, a_col: str, e_col: str, i_col: str,
                     na=12, ne=10, ni=6) -> np.ndarray:
    a = df[a_col].to_numpy(float)
    e = df[e_col].to_numpy(float)
    inc = df[i_col].to_numpy(float)

    a_edges = np.linspace(np.nanmin(a), np.nanmax(a), na + 1)
    e_edges = np.linspace(np.nanmin(e), np.nanmax(e), ne + 1)
    i_edges = np.linspace(np.nanmin(inc), np.nanmax(inc), ni + 1)

    ia = np.clip(np.digitize(a, a_edges) - 1, 0, na - 1)
    ie = np.clip(np.digitize(e, e_edges) - 1, 0, ne - 1)
    ii = np.clip(np.digitize(inc, i_edges) - 1, 0, ni - 1)

    gid = ia + na * ie + (na * ne) * ii
    return gid.astype(int)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins=15) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    m = np.isfinite(p)
    y_true, p = y_true[m], p[m]
    if p.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mb = idx == b
        if not np.any(mb):
            continue
        conf = p[mb].mean()
        acc = y_true[mb].mean()
        ece += (mb.mean()) * abs(acc - conf)
    return float(ece)


def build_models(seed: int):
    models = {}

    models["lr"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed
        ))
    ])

    models["rf"] = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed
    )

    models["hgb"] = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=600,
        random_state=seed
    )

    return models


def fit_calibrated(base_model, X_train, y_train, groups_train,
                   calib_method="sigmoid", calib_frac=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=calib_frac, random_state=seed)
    tr_idx, cal_idx = next(gss.split(X_train, y_train, groups=groups_train))

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    base = clone(base_model)
    base.fit(X_tr, y_tr)


    cal = CalibratedClassifierCV(base, method=calib_method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def load_mode_df(pro_path: str, retro_path: str, mode: str) -> pd.DataFrame:
    mode = mode.lower()
    df_pro = pd.read_csv(pro_path)
    df_pro["tag"] = "prograde"
    df_ret = pd.read_csv(retro_path)
    df_ret["tag"] = "retrograde"

    if mode == "pro":
        return df_pro
    if mode == "retro":
        return df_ret
    if mode == "both":
        return pd.concat([df_pro, df_ret], ignore_index=True)
    raise ValueError("mode must be one of: pro, retro, both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pro", default="out/integrated_pm_pro.csv")
    ap.add_argument("--retro", default="out/integrated_pm_retro.csv")
    ap.add_argument("--mode", choices=["pro", "retro", "both"], default="pro")
    ap.add_argument("--outdir", default="out/ml/surrogate_v2")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tile-na", type=int, default=12)
    ap.add_argument("--tile-ne", type=int, default=10)
    ap.add_argument("--tile-ni", type=int, default=6)

    ap.add_argument("--calib-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--calib-frac", type=float, default=0.25)
    ap.add_argument("--plot-bins", type=int, default=15)
    args = ap.parse_args()

    mode = args.mode.lower()
    outdir = Path(args.outdir) / mode
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_mode_df(args.pro, args.retro, mode)
    df = add_derived(df)

    # Feature set
    feat = ["a_over_RH", "e", "i_eff_deg"]
    if mode == "both":
        if "sense_pro" not in df.columns:
            raise RuntimeError("mode=both requires column 'sense_pro' in the CSVs.")
        feat.append("sense_pro")

    # Drop leaky columns (safety net)
    df = drop_leaky_columns(df, keep_cols=feat + ["y_stable", "id", "tag", "epoch_id", "regime_id", "i_deg"])

    # Build arrays
    X = df[feat].to_numpy(float)
    y = df["y_stable"].to_numpy(int)

    # Groups: tiles in IC space
    groups = make_tile_groups(
        df, a_col="a_over_RH", e_col="e", i_col="i_eff_deg",
        na=args.tile_na, ne=args.tile_ne, ni=args.tile_ni
    )

    gkf = GroupKFold(n_splits=args.folds)
    models = build_models(args.seed)

    oof = pd.DataFrame({
        "id": df["id"].astype(int).to_numpy(),
        "tag": df["tag"].astype(str).to_numpy(),
        "y": y
    })
    for name in models:
        oof[f"p_{name}"] = np.nan

    fitted_per_fold = {name: [] for name in models}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        Xte = X[te]

        for name, base in models.items():
            cal = fit_calibrated(
                base, Xtr, ytr, gtr,
                calib_method=args.calib_method,
                calib_frac=args.calib_frac,
                seed=args.seed + 1000*fold
            )
            oof.loc[te, f"p_{name}"] = proba(cal, Xte)
            fitted_per_fold[name].append(cal)

    # Metrics
    metrics = {
        "mode": mode,
        "features": feat,
        "split": {"folds": args.folds, "tile": [args.tile_na, args.tile_ne, args.tile_ni]},
        "calibration": {"method": args.calib_method, "frac": args.calib_frac},
        "models": {}
    }

    for name in models:
        p = oof[f"p_{name}"].to_numpy(float)
        m = np.isfinite(p)
        yy = oof.loc[m, "y"].to_numpy(int)
        pp = np.clip(p[m], 1e-6, 1 - 1e-6)

        out = {}
        out["brier"] = float(brier_score_loss(yy, pp))
        out["logloss"] = float(log_loss(yy, pp))
        out["ece"] = expected_calibration_error(yy, pp, n_bins=args.plot_bins)

        if len(np.unique(yy)) == 2:
            out["roc_auc"] = float(roc_auc_score(yy, pp))
            out["ap"] = float(average_precision_score(yy, pp))
        else:
            out["roc_auc"] = float("nan")
            out["ap"] = float("nan")

        yhat = (pp >= 0.5).astype(int)
        cm = confusion_matrix(yy, yhat, labels=[0, 1])
        out["confusion_0p5"] = cm.tolist()
        out["acc_0p5"] = float((yhat == yy).mean())

        metrics["models"][name] = out

    # Save
    oof.to_csv(outdir / "oof_predictions.csv", index=False)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(
        {"models_per_fold": fitted_per_fold, "features": feat, "metrics": metrics},
        outdir / "models.pkl"
    )

    # Reliability plot
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    for name in models:
        p = oof[f"p_{name}"].to_numpy(float)
        m = np.isfinite(p)
        yy = oof.loc[m, "y"].to_numpy(int)
        pp = p[m]
        frac_pos, mean_pred = calibration_curve(yy, pp, n_bins=args.plot_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "F_reliability.png", bbox_inches="tight")
    plt.close(fig)

    print("[OK] Mode:", mode)
    print("[OK] Wrote:")
    print(" ", outdir / "oof_predictions.csv")
    print(" ", outdir / "metrics.json")
    print(" ", outdir / "models.pkl")
    print(" ", outdir / "F_reliability.png")


if __name__ == "__main__":
    main()

