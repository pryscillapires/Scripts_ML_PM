#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03a_portraits.py

"Portrait" scatter plots (no hex, no binning):
- a–e portrait: stable vs escape, prograde vs retrograde panels
- a–i_eff portrait: stable vs escape, prograde vs retrograde panels

Axes in English. CMDA-clean.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_cfg(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    pbin = float(cfg["pm_binary"]["P_bin_s"])
    RH_km = float(cfg["constants"]["R_H_pm_sun"]) / 1000.0
    return cfg, pbin, RH_km


def add_derived(df: pd.DataFrame, pbin: float, RH_km: float) -> pd.DataFrame:
    df = df.copy()
    df["is_stable"] = (df["status"] == "stable").astype(int)
    df["is_escape"] = (df["status"] == "escape_local").astype(int)
    df["t_over_Pbin"] = df["t_end"].astype(float) / float(pbin)

    i = df["i_deg"].astype(float).to_numpy()
    df["i_eff_deg"] = np.where(i <= 90.0, i, 180.0 - i)

    df["a_km"] = df["a_over_RH"].astype(float) * RH_km
    return df


def _panel(ax, df, title, xcol, xlabel, ycol, ylabel,
           xlim=None, ylim=None, s=8, alpha=0.65):
    # Masks
    m_st = df["is_stable"].to_numpy(int) == 1
    m_es = df["is_escape"].to_numpy(int) == 1
    m_other = ~(m_st | m_es)

    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)

    # Plot stable first (background), then escapes, then "other"
    if np.any(m_st):
        ax.scatter(x[m_st], y[m_st], s=s, alpha=alpha, label=f"Stable (N={m_st.sum()})")
    if np.any(m_es):
        ax.scatter(x[m_es], y[m_es], s=s, alpha=alpha, marker="x", label=f"Escape (N={m_es.sum()})")
    if np.any(m_other):
        ax.scatter(x[m_other], y[m_other], s=s, alpha=alpha, marker=".", label=f"Other (N={m_other.sum()})")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def plot_portrait_ae(df_pro, df_ret, outpath: Path,
                     x_unit="km", RH_km=48305.0,
                     aRH_min=0.4, aRH_max=1.2,
                     e_min=0.0, e_max=0.5,
                     s=8, alpha=0.65):

    if x_unit == "km":
        xcol = "a_km"
        xlabel = r"$a$ (km)"
        xlim = (aRH_min * RH_km, aRH_max * RH_km)
    else:
        xcol = "a_over_RH"
        xlabel = r"$a/R_H$"
        xlim = (aRH_min, aRH_max)

    ycol = "e"
    ylabel = r"$e$"
    ylim = (e_min, e_max)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160, sharex=True, sharey=True)

    _panel(axes[0], df_pro, "Prograde", xcol, xlabel, ycol, ylabel, xlim=xlim, ylim=ylim, s=s, alpha=alpha)
    _panel(axes[1], df_ret, "Retrograde", xcol, xlabel, ycol, ylabel, xlim=xlim, ylim=ylim, s=s, alpha=alpha)

    # Single legend (right)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_portrait_ai(df_pro, df_ret, outpath: Path,
                     x_unit="km", RH_km=48305.0,
                     aRH_min=0.4, aRH_max=1.2,
                     i_min=0.0, i_max=40.0,
                     s=8, alpha=0.65):

    if x_unit == "km":
        xcol = "a_km"
        xlabel = r"$a$ (km)"
        xlim = (aRH_min * RH_km, aRH_max * RH_km)
    else:
        xcol = "a_over_RH"
        xlabel = r"$a/R_H$"
        xlim = (aRH_min, aRH_max)

    ycol = "i_eff_deg"
    ylabel = r"$i_{\rm eff}$ (deg)"
    ylim = (i_min, i_max)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160, sharex=True, sharey=True)

    _panel(axes[0], df_pro, "Prograde", xcol, xlabel, ycol, ylabel, xlim=xlim, ylim=ylim, s=s, alpha=alpha)
    _panel(axes[1], df_ret, "Retrograde", xcol, xlabel, ycol, ylabel, xlim=xlim, ylim=ylim, s=s, alpha=alpha)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pm_physical.json")
    ap.add_argument("--pro", default="out/integrated_pm_pro.csv")
    ap.add_argument("--retro", default="out/integrated_pm_retro.csv")
    ap.add_argument("--outdir", default="out/figs_truth_km")
    ap.add_argument("--x-unit", choices=["aRH", "km"], default="km")
    ap.add_argument("--aRH-min", type=float, default=0.4)
    ap.add_argument("--aRH-max", type=float, default=1.2)
    ap.add_argument("--e-min", type=float, default=0.0)
    ap.add_argument("--e-max", type=float, default=0.5)
    ap.add_argument("--i-max", type=float, default=40.0)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.65)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, pbin, RH_km = load_cfg(args.config)

    df_pro = add_derived(pd.read_csv(args.pro), pbin, RH_km)
    df_ret = add_derived(pd.read_csv(args.retro), pbin, RH_km)

    plot_portrait_ae(
        df_pro, df_ret, outdir / "F_truth_portrait_ae.png",
        x_unit=args.x_unit, RH_km=RH_km,
        aRH_min=args.aRH_min, aRH_max=args.aRH_max,
        e_min=args.e_min, e_max=args.e_max,
        s=args.s, alpha=args.alpha
    )

    plot_portrait_ai(
        df_pro, df_ret, outdir / "F_truth_portrait_aiEff.png",
        x_unit=args.x_unit, RH_km=RH_km,
        aRH_min=args.aRH_min, aRH_max=args.aRH_max,
        i_max=args.i_max,
        s=args.s, alpha=args.alpha
    )

    print("[OK] Wrote:")
    print(" ", outdir / "F_truth_portrait_ae.png")
    print(" ", outdir / "F_truth_portrait_aiEff.png")


if __name__ == "__main__":
    main()

