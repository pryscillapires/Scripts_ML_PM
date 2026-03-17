#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_pr_policy_curves.py (paper-safe final)

OOF-only PR + policy curves for Patroclus–Menoetius stability surrogate.

Inputs:
- <base>/pro/oof_predictions.csv
- <base>/retro/oof_predictions.csv

Outputs (in --outdir):
- F_PR_pro_<model>.png
- F_PR_retro_<model>.png
- F_policy_pro_<model>.png
- F_policy_retro_<model>.png
- summary_pr_policy_<model>.json

Notes:
- Thresholds are selected ONLY from OOF predictions, under transparent FPR targets.
- If unstable_selected_share is ~0 for all shown top-k points, the secondary axis is hidden.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, precision_recall_curve


EPS = 1e-12


def _confusion_at_threshold(y, p, thr):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= thr).astype(int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))  # false stable (false positive)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, fp, tn, fn


def _rates_from_conf(tp, fp, tn, fn):
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec  = tp / (tp + fn) if (tp + fn) else np.nan
    fpr  = fp / (fp + tn) if (fp + tn) else np.nan
    tnr  = tn / (tn + fp) if (tn + fp) else np.nan
    acc  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else np.nan
    return float(prec), float(rec), float(fpr), float(tnr), float(acc)


def choose_threshold_by_fpr(y, p, fpr_target):
    """
    Choose a probability threshold thr such that FPR(thr) <= fpr_target.
    Among feasible thresholds, maximize recall (stable), then precision.
    Uses ONLY the provided arrays (intended to be OOF predictions).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    m = np.isfinite(p)
    y, p = y[m], p[m]
    if p.size == 0:
        return np.nan, None

    p = np.clip(p, EPS, 1.0 - EPS)

    thr_candidates = np.unique(p)
    thr_candidates = np.unique(np.concatenate([thr_candidates, [0.0, 1.0]]))
    thr_candidates.sort()

    best = None
    best_thr = None

    for thr in thr_candidates:
        tp, fp, tn, fn = _confusion_at_threshold(y, p, thr)
        prec, rec, fpr, tnr, acc = _rates_from_conf(tp, fp, tn, fn)
        if not (np.isfinite(fpr) and np.isfinite(rec) and np.isfinite(prec)):
            continue
        if fpr <= fpr_target:
            cand = (rec, prec)  # maximize recall, then precision
            if (best is None) or (cand > best):
                best = cand
                best_thr = float(thr)

    if best_thr is None:
        return np.nan, None

    tp, fp, tn, fn = _confusion_at_threshold(y, p, best_thr)
    prec, rec, fpr, tnr, acc = _rates_from_conf(tp, fp, tn, fn)
    details = {
        "threshold": best_thr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec,
        "recall_stable": rec,
        "fpr_false_stable": fpr,
        "tnr_unstable": tnr,
        "accuracy": acc
    }
    return best_thr, details


def policy_topk(y, p, fracs=(0.01, 0.02, 0.05, 0.10, 0.20, 0.30)):
    """
    Rank by p descending; select top-k fraction; report:
    - precision@k (stable yield)
    - recall@k (stable coverage)
    - unstable_selected_share = (#unstable selected)/(#unstable total)
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    m = np.isfinite(p)
    y, p = y[m], p[m]
    n = y.size
    if n == 0:
        return {"frac": [], "k": [], "precision_at_k": [], "recall_at_k": [], "unstable_selected_share": []}

    p = np.clip(p, EPS, 1.0 - EPS)

    n_stable = int(np.sum(y == 1))
    n_unstable = int(np.sum(y == 0))

    order = np.argsort(-p)  # descending
    y_sorted = y[order]

    out = {"frac": [], "k": [], "precision_at_k": [], "recall_at_k": [], "unstable_selected_share": []}

    for frac in fracs:
        k = max(1, int(round(float(frac) * n)))
        sel = y_sorted[:k]
        tp = int(np.sum(sel == 1))
        fp = int(np.sum(sel == 0))

        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec  = tp / n_stable if n_stable else np.nan
        share_unstable_selected = fp / n_unstable if n_unstable else np.nan

        out["frac"].append(float(frac))
        out["k"].append(int(k))
        out["precision_at_k"].append(float(prec))
        out["recall_at_k"].append(float(rec))
        out["unstable_selected_share"].append(float(share_unstable_selected))

    return out


def _auto_zoom_pr(ax, mode, prevalence, ap):
    # For retro: full view.
    if mode != "pro":
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        return

    # For pro: zoom to show meaningful variation near high precision.
    y0 = max(0.80, min(0.98, prevalence - 0.05))
    ax.set_ylim(y0, 1.005)

    if prevalence > 0.85 and ap > 0.95:
        ax.set_xlim(0.50, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)


def plot_pr(y, p, title, outpath, prevalence_line=True, mark_points=None, mode="pro"):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    m = np.isfinite(p)
    y, p = y[m], p[m]
    p = np.clip(p, EPS, 1.0 - EPS)

    ap = float(average_precision_score(y, p))
    prec, rec, _thr = precision_recall_curve(y, p)
    prevalence = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
    ax.plot(rec, prec, label=f"OOF PR (AP={ap:.4f})")

    if prevalence_line:
        ax.hlines(prevalence, 0.0, 1.0, linestyles="--", linewidth=1.2,
                  label=f"Prevalence baseline={prevalence:.3f}")

    offsets = [(8, 6), (8, -6), (8, -18), (8, -30), (8, -42)]
    used = 0

    if mark_points:
        for mp in mark_points:
            thr0 = float(mp["thr"])
            tp, fp, tn, fn = _confusion_at_threshold(y, p, thr0)
            prc, rcl, fpr, tnr, acc = _rates_from_conf(tp, fp, tn, fn)
            if np.isfinite(prc) and np.isfinite(rcl):
                ax.plot([rcl], [prc], marker="o", markersize=5)
                dx, dy = offsets[min(used, len(offsets) - 1)]
                used += 1
                ax.annotate(
                    mp["label"],
                    xy=(rcl, prc),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left",
                    va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.8),
                )

    ax.set_xlabel("Recall (stable)")
    ax.set_ylabel("Precision (stable)")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    _auto_zoom_pr(ax, mode=mode, prevalence=prevalence, ap=ap)

    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    return {"ap": ap, "prevalence": prevalence}


def plot_policy(policy_dict, title, outpath):
    x = np.array(policy_dict["frac"], float)
    prec = np.array(policy_dict["precision_at_k"], float)
    rec  = np.array(policy_dict["recall_at_k"], float)
    u_sh = np.array(policy_dict["unstable_selected_share"], float)

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
    ax.plot(x, prec, marker="o", label="precision@k (stable yield)")
    ax.plot(x, rec,  marker="o", label="recall@k (stable coverage)")

    ax.set_xlabel("Selected fraction k/N")
    ax.set_ylabel("Value")
    ax.set_xlim(0.0, max(x) * 1.02 if x.size else 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    finite = np.isfinite(u_sh)
    umax = float(np.nanmax(u_sh[finite])) if np.any(finite) else 0.0

    # Hide secondary axis if effectively zero everywhere (avoid 1e-9 confusing scale)
    if umax <= 1e-6:
        ax.text(
            0.02, 0.08,
            "unstable_selected_share ≈ 0 for all shown k",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )
        ax.legend(frameon=True, loc="lower right")
    else:
        ax2 = ax.twinx()
        ax2.plot(x, u_sh, marker="s", linestyle="--", label="unstable_selected_share", alpha=0.9)
        ax2.set_ylabel("Share of unstable selected")
        ax2.set_ylim(0.0, umax * 1.10 + 1e-9)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=True, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def load_oof(oof_path, model):
    df = pd.read_csv(oof_path)
    col = f"p_{model}"
    if col not in df.columns:
        raise RuntimeError(f"Missing column '{col}' in {oof_path}. Available: {list(df.columns)}")
    if "y" not in df.columns:
        raise RuntimeError(f"Missing column 'y' in {oof_path}.")
    y = df["y"].to_numpy(int)
    p = df[col].to_numpy(float)
    return df, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/ml/surrogate_v2",
                    help="Base directory containing <mode>/oof_predictions.csv (mode=pro,retro).")
    ap.add_argument("--outdir", default="out/ml/figs_pr_policy")
    ap.add_argument("--model", choices=["hgb", "rf", "lr"], default="hgb")
    ap.add_argument("--topk", default="0.01,0.02,0.05,0.10,0.20,0.30",
                    help="Comma-separated fractions for top-k policy.")
    ap.add_argument("--fpr-targets", default="0.001,0.005,0.01",
                    help="Comma-separated FPR targets for threshold selection.")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fracs = tuple(float(x) for x in args.topk.split(",") if x.strip())
    # IMPORTANT: argparse converts --fpr-targets -> args.fpr_targets
    fpr_targets = tuple(float(x) for x in args.fpr_targets.split(",") if x.strip())

    summary = {"model": args.model, "modes": {}}

    for mode in ["pro", "retro"]:
        oof_path = base / mode / "oof_predictions.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"Not found: {oof_path}")

        _, y, p = load_oof(oof_path, args.model)

        op_points = [{"thr": 0.5, "label": "thr=0.5"}]
        chosen = {}

        for tgt in fpr_targets:
            thr, det = choose_threshold_by_fpr(y, p, fpr_target=tgt)
            key = f"fpr<= {tgt:g}"
            chosen[key] = det if det is not None else None
            if det is not None and np.isfinite(det["threshold"]):
                op_points.append({"thr": det["threshold"], "label": f"FPR≤{tgt:g}"})

        pr_stats = plot_pr(
            y, p,
            title=f"{mode.upper()} | OOF Precision–Recall ({args.model})",
            outpath=outdir / f"F_PR_{mode}_{args.model}.png",
            mark_points=op_points,
            mode=mode
        )

        pol = policy_topk(y, p, fracs=fracs)
        plot_policy(
            pol,
            title=f"{mode.upper()} | OOF Top-k policy ({args.model})",
            outpath=outdir / f"F_policy_{mode}_{args.model}.png"
        )

        # Reference at thr=0.5
        tp, fp, tn, fn = _confusion_at_threshold(np.asarray(y), np.clip(np.asarray(p), EPS, 1 - EPS), 0.5)
        prec, rec, fpr, tnr, acc = _rates_from_conf(tp, fp, tn, fn)

        mode_sum = {
            "prevalence_stable": pr_stats["prevalence"],
            "AP": pr_stats["ap"],
            "thresholds_by_fpr": chosen,
            "policy_topk": pol,
            "at_thr_0p5": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": prec, "recall_stable": rec,
                "fpr_false_stable": fpr, "tnr_unstable": tnr,
                "accuracy": acc
            },
            "n": int(np.size(y)),
            "n_stable": int(np.sum(np.asarray(y) == 1)),
            "n_unstable": int(np.sum(np.asarray(y) == 0)),
        }
        summary["modes"][mode] = mode_sum

        print(f"[{mode}] prevalence={pr_stats['prevalence']:.4f}  AP={pr_stats['ap']:.4f}")
        print(f"[{mode}] thr=0.5  prec={prec:.4f}  recall={rec:.4f}  "
              f"FPR(false-stable)={fpr:.4f}  TNR={tnr:.4f}")
        for k, det in chosen.items():
            if det is None:
                print(f"[{mode}] {k}: no feasible threshold")
            else:
                print(f"[{mode}] {k}: thr={det['threshold']:.6f}  prec={det['precision']:.4f}  "
                      f"recall={det['recall_stable']:.4f}  FPR={det['fpr_false_stable']:.4f}")

    out_json = outdir / f"summary_pr_policy_{args.model}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Wrote:")
    for fn in [
        f"F_PR_pro_{args.model}.png",
        f"F_PR_retro_{args.model}.png",
        f"F_policy_pro_{args.model}.png",
        f"F_policy_retro_{args.model}.png",
        f"summary_pr_policy_{args.model}.json",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()

