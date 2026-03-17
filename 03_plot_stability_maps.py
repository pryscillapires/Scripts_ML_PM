#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_plot_portraits_truth.py

"Pluto-style" stability portraits using INITIAL CONDITIONS (ICs):
- a/R_H vs e : stable vs escape (scatter)
- a/R_H vs i_eff : stable vs escape (scatter)
- a/R_H vs e : stable fraction (hexbin-style ratio on a fixed grid)

No suptitles. All labels in English.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pbin(cfg_path: str) -> float:
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return float(cfg["pm_binary"]["P_bin_s"])


def add_derived(df: pd.DataFrame, pbin: float) -> pd.DataFrame:
    df = df.copy()
    df["is_stable"] = (df["status"] == "stable")
    df["is_escape"] = (df["status"] == "escape_local")
    df["t_over_Pbin"] = df["t_end"] / pbin

    # i_eff in [0, 90], so prograde/retrograde share the same axis scale
    i = df["i_deg"].astype(float).to_numpy()
    df["i_eff_deg"] = np.where(i <= 90.0, i, 180.0 - i)
    return df


def _panel_label(ax, txt):
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        fontsize=10
    )


def plot_scatter_portrait(df_pro, df_ret, outpath, ycol, ylabel, ylims=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=180, sharex=True, sharey=True)

    for ax, (tag, df) in zip(axes, [("Prograde", df_pro), ("Retrograde", df_ret)]):

        st = df[df["is_stable"]]
        esc = df[df["is_escape"]]

        # Stable: dense cloud (small markers, transparent)
        ax.scatter(
            st["a_over_RH"], st[ycol],
            s=2, alpha=0.20, linewidths=0,
            rasterized=True,
            label=f"Stable (N={len(st)})"
        )

        # Escapes: highlight
        ax.scatter(
            esc["a_over_RH"], esc[ycol],
            s=10, alpha=0.70, linewidths=0,
            rasterized=True,
            label=f"Escape (N={len(esc)})"
        )

        _panel_label(ax, tag)
        ax.grid(True, alpha=0.25)

        ax.set_xlabel(r"$a/R_H$")
        ax.set_ylabel(ylabel)

    if ylims is not None:
        axes[0].set_ylim(*ylims)

    # one legend for the whole figure (inside right panel)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=True, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def stable_fraction_grid(df, a_edges, e_edges):
    a = df["a_over_RH"].to_numpy()
    e = df["e"].to_numpy()
    st = df["is_stable"].to_numpy().astype(float)

    ia = np.digitize(a, a_edges) - 1
    ie = np.digitize(e, e_edges) - 1

    na = len(a_edges) - 1
    ne = len(e_edges) - 1

    cnt = np.zeros((ne, na), dtype=float)
    stc = np.zeros((ne, na), dtype=float)

    ok = (ia >= 0) & (ia < na) & (ie >= 0) & (ie < ne)
    ia = ia[ok]; ie = ie[ok]; st = st[ok]

    # accumulate
    for j, i, s in zip(ie, ia, st):
        cnt[j, i] += 1.0
        stc[j, i] += s

    frac = stc / np.where(cnt > 0, cnt, np.nan)
    return frac, cnt


def plot_stable_fraction_ae(df_pro, df_ret, outpath,
                            a_min=0.4, a_max=1.2, na=60,
                            e_min=0.0, e_max=0.5, ne=50,
                            min_count=20):

    a_edges = np.linspace(a_min, a_max, na + 1)
    e_edges = np.linspace(e_min, e_max, ne + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=180, sharex=True, sharey=True)

    last_im = None
    for ax, (tag, df) in zip(axes, [("Prograde", df_pro), ("Retrograde", df_ret)]):

        frac, cnt = stable_fraction_grid(df, a_edges, e_edges)
        frac = np.where(cnt >= min_count, frac, np.nan)

        im = ax.pcolormesh(a_edges, e_edges, frac, shading="auto", vmin=0.0, vmax=1.0)
        last_im = im

        _panel_label(ax, tag)
        ax.set_xlabel(r"$a/R_H$")
        ax.set_ylabel(r"$e$")
        ax.grid(True, alpha=0.15)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Stable fraction")

    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pm_physical.json")
    ap.add_argument("--pro", default="out/integrated_pm_pro.csv")
    ap.add_argument("--retro", default="out/integrated_pm_retro.csv")
    ap.add_argument("--outdir", default="out/figs_truth")
    ap.add_argument("--min-count", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pbin = load_pbin(args.config)

    df_pro = add_derived(pd.read_csv(args.pro), pbin)
    df_ret = add_derived(pd.read_csv(args.retro), pbin)

    # a-e portrait (ICs)
    plot_scatter_portrait(
        df_pro, df_ret,
        outdir / "F_truth_portrait_ae.png",
        ycol="e",
        ylabel=r"$e$",
        ylims=(0.0, 0.5),
    )

    # a-i_eff portrait (ICs)
    plot_scatter_portrait(
        df_pro, df_ret,
        outdir / "F_truth_portrait_ai.png",
        ycol="i_eff_deg",
        ylabel=r"$i_{\rm eff}$ [deg]",
        ylims=(0.0, 40.0),
    )

    # stable fraction map in a-e (dense + robust)
    plot_stable_fraction_ae(
        df_pro, df_ret,
        outdir / "F_truth_portrait_ae_hex.png",
        min_count=args.min_count
    )

    print("[OK] Wrote:")
    for fn in [
        "F_truth_portrait_ae.png",
        "F_truth_portrait_ai.png",
        "F_truth_portrait_ae_hex.png",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()

