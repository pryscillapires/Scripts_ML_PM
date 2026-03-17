#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_summarize_phase_gap.py

Summarize integrated CSV chunks for:
- phase-randomized prograde / retrograde runs
- inclination gap run

Outputs:
- one merged CSV per run
- a JSON summary with stable fractions and deltas vs baseline
- simple plots (matplotlib): portrait (a/RH,e) colored by status and stable fraction vs a/RH.

Assumptions:
- Each outdir contains partial_chunk*.csv files produced by 02_integrate_pm_real_v3.py
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_partials(outdir: Path) -> pd.DataFrame:
    files = sorted(outdir.glob("partial_chunk*.csv"))
    if not files:
        raise FileNotFoundError(f"No partial_chunk*.csv found in {outdir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def stable_mask(df: pd.DataFrame) -> np.ndarray:
    return (df["status"].astype(str) == "stable").to_numpy()


def stable_fraction(df: pd.DataFrame) -> float:
    m = stable_mask(df)
    return float(m.mean()) if len(m) else float("nan")


def plot_portrait_ae(df: pd.DataFrame, outpng: Path, title: str):
    # simple encoding: stable blue, escape orange, collision red, other gray
    status = df["status"].astype(str).values
    a = df["a_over_RH"].values
    e = df["e"].values

    # map to categories
    cat = np.full(len(df), "other", dtype=object)
    cat[status == "stable"] = "stable"
    cat[np.char.find(status, "escape") >= 0] = "escape"
    cat[np.char.find(status, "collision") >= 0] = "collision"

    plt.figure()
    # plot in layers for readability
    for key, marker in [("other", "."), ("escape", "."), ("collision", "."), ("stable", ".")]:
        sel = (cat == key)
        if sel.any():
            plt.scatter(a[sel], e[sel], s=6, marker=marker, alpha=0.7, label=key)

    plt.xlabel(r"$a/R_H$")
    plt.ylabel(r"$e$")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def plot_stable_vs_a(df: pd.DataFrame, outpng: Path, title: str, nbins: int = 12):
    a = df["a_over_RH"].values
    y = stable_mask(df).astype(float)

    bins = np.linspace(np.nanmin(a), np.nanmax(a), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    frac = np.full(nbins, np.nan)
    err = np.full(nbins, np.nan)

    for i in range(nbins):
        sel = (a >= bins[i]) & (a < bins[i+1])
        n = int(sel.sum())
        if n >= 30:
            p = float(y[sel].mean())
            frac[i] = p
            # binomial standard error (for display only; paper can use Wilson later)
            err[i] = np.sqrt(p*(1-p)/n)

    plt.figure()
    ok = np.isfinite(frac)
    plt.errorbar(centers[ok], frac[ok], yerr=err[ok], fmt="o", capsize=3)
    plt.xlabel(r"$a/R_H$")
    plt.ylabel("Stable fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_phase_pro", default="out/integrated_phase_pro")
    ap.add_argument("--out_phase_retro", default="out/integrated_phase_retro")
    ap.add_argument("--out_gap", default="out/integrated_gap")
    ap.add_argument("--outdir", default="out/phase_gap_summary")
    ap.add_argument("--baseline_pro", type=float, default=0.9456)
    ap.add_argument("--baseline_retro", type=float, default=0.1106)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_pro = load_partials(Path(args.out_phase_pro))
    df_retro = load_partials(Path(args.out_phase_retro))
    df_gap = load_partials(Path(args.out_gap))

    # Save merged
    df_pro.to_csv(outdir / "integrated_phase_pro_merged.csv", index=False)
    df_retro.to_csv(outdir / "integrated_phase_retro_merged.csv", index=False)
    df_gap.to_csv(outdir / "integrated_gap_merged.csv", index=False)

    f_pro = stable_fraction(df_pro)
    f_retro = stable_fraction(df_retro)
    f_gap = stable_fraction(df_gap)

    summary = {
        "phase_pro": {"N": int(len(df_pro)), "f_stable": f_pro, "delta_vs_baseline": f_pro - args.baseline_pro},
        "phase_retro": {"N": int(len(df_retro)), "f_stable": f_retro, "delta_vs_baseline": f_retro - args.baseline_retro},
        "gap": {"N": int(len(df_gap)), "f_stable": f_gap},
        "baselines": {"pro": args.baseline_pro, "retro": args.baseline_retro},
    }

    with open(outdir / "summary_phase_gap.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plot_portrait_ae(df_pro, outdir / "F_phase_portrait_ae_pro.png", "Phase-randomized truth (prograde): (a/R_H, e)")
    plot_portrait_ae(df_retro, outdir / "F_phase_portrait_ae_retro.png", "Phase-randomized truth (retrograde): (a/R_H, e)")
    plot_portrait_ae(df_gap, outdir / "F_gap_portrait_ae.png", "Inclination-gap diagnostic: (a/R_H, e)")

    plot_stable_vs_a(df_pro, outdir / "F_phase_stable_vs_a_pro.png", "Phase-randomized stable fraction vs a/R_H (prograde)")
    plot_stable_vs_a(df_retro, outdir / "F_phase_stable_vs_a_retro.png", "Phase-randomized stable fraction vs a/R_H (retrograde)")

    print("[OK] Saved summary + merged CSVs + plots to:", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
