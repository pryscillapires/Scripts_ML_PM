#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_plot_figures.py (fixed)

Truth plots from direct N-body integration:
- Escape-time ECDF (escapes only)
- Stable fraction vs a (a/R_H or km)
- 2x3 maps (Prograde/Retrograde rows; i_eff bins as columns):
    * median t_esc/P_bin (escapes only)
    * stable fraction (all outcomes)

No suptitle. Labels in English.
Robust 2D binning (no pandas categorical shrink -> no pcolormesh shape errors).
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
    df["t_over_Pbin"] = df["t_end"] / pbin

    i = df["i_deg"].astype(float).to_numpy()
    df["i_eff_deg"] = np.where(i <= 90.0, i, 180.0 - i)

    df["a_km"] = df["a_over_RH"].astype(float) * RH_km
    return df


def ecdf(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot_escape_ecdf(df_pro, df_ret, outpath: Path):
    esc_pro = df_pro.loc[df_pro["is_escape"] == 1, "t_over_Pbin"].to_numpy()
    esc_ret = df_ret.loc[df_ret["is_escape"] == 1, "t_over_Pbin"].to_numpy()

    x1, y1 = ecdf(esc_pro)
    x2, y2 = ecdf(esc_ret)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    if x1.size:
        ax.step(x1, y1, where="post", label=f"Prograde (N={x1.size})")
    if x2.size:
        ax.step(x2, y2, where="post", label=f"Retrograde (N={x2.size})")

    ax.set_xlabel(r"$t_{\rm esc}/P_{\rm bin}$")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_stable_vs_a(df_pro, df_ret, outpath: Path, x_unit="aRH",
                     aRH_min=0.4, aRH_max=1.2, nbins=40, RH_km=48305.0):
    if x_unit == "km":
        xcol = "a_km"
        xmin, xmax = aRH_min*RH_km, aRH_max*RH_km
        xlabel = r"$a$ (km)"
    else:
        xcol = "a_over_RH"
        xmin, xmax = aRH_min, aRH_max
        xlabel = r"$a/R_H$"

    bins = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def stable_fraction_vs_bin(df):
        x = df[xcol].to_numpy(float)
        st = df["is_stable"].to_numpy(float)
        which = np.digitize(x, bins) - 1
        frac = np.full(nbins, np.nan)
        for k in range(nbins):
            m = which == k
            if np.any(m):
                frac[k] = st[m].mean()
        return frac

    f_pro = stable_fraction_vs_bin(df_pro)
    f_ret = stable_fraction_vs_bin(df_ret)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.plot(centers, f_pro, label="Prograde")
    ax.plot(centers, f_ret, label="Retrograde")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Stable fraction")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def bin2d_median(a, e, val, a_edges, e_edges, min_count):
    na = len(a_edges) - 1
    ne = len(e_edges) - 1
    ia = np.digitize(a, a_edges) - 1
    ie = np.digitize(e, e_edges) - 1
    m = (ia >= 0) & (ia < na) & (ie >= 0) & (ie < ne) & np.isfinite(val)
    ia, ie, val = ia[m], ie[m], val[m]

    grid = np.full((ne, na), np.nan)
    cnt  = np.zeros((ne, na), dtype=int)

    if val.size == 0:
        return grid, cnt

    tmp = pd.DataFrame({"ie": ie, "ia": ia, "val": val})
    g = tmp.groupby(["ie", "ia"])
    c = g.size()
    med = g["val"].median()

    for (ie_i, ia_i), c_ij in c.items():
        cnt[int(ie_i), int(ia_i)] = int(c_ij)
    for (ie_i, ia_i), m_ij in med.items():
        if cnt[int(ie_i), int(ia_i)] >= min_count:
            grid[int(ie_i), int(ia_i)] = float(m_ij)

    return grid, cnt


def bin2d_stable_fraction(a, e, is_stable, a_edges, e_edges, min_count):
    na = len(a_edges) - 1
    ne = len(e_edges) - 1
    ia = np.digitize(a, a_edges) - 1
    ie = np.digitize(e, e_edges) - 1
    m = (ia >= 0) & (ia < na) & (ie >= 0) & (ie < ne)
    ia, ie, st = ia[m], ie[m], is_stable[m]

    frac = np.full((ne, na), np.nan)
    cnt  = np.zeros((ne, na), dtype=int)
    stc  = np.zeros((ne, na), dtype=int)

    if st.size == 0:
        return frac, cnt

    tmp = pd.DataFrame({"ie": ie, "ia": ia, "st": st.astype(int)})
    g = tmp.groupby(["ie", "ia"])
    c = g.size()
    s = g["st"].sum()

    for (ie_i, ia_i), c_ij in c.items():
        cnt[int(ie_i), int(ia_i)] = int(c_ij)
    for (ie_i, ia_i), s_ij in s.items():
        stc[int(ie_i), int(ia_i)] = int(s_ij)

    with np.errstate(invalid="ignore", divide="ignore"):
        frac = stc / cnt.astype(float)
    frac[cnt < min_count] = np.nan
    return frac, cnt


def plot_2x3_maps(df_pro, df_ret, outdir: Path,
                 x_unit="aRH", RH_km=48305.0,
                 aRH_min=0.4, aRH_max=1.2, na=28,
                 e_min=0.0, e_max=0.5, ne=25,
                 i_bins=((0, 10), (10, 20), (20, 40)),
                 min_count_stable=30,
                 min_count_escape=10):

    if x_unit == "km":
        a_edges = np.linspace(aRH_min*RH_km, aRH_max*RH_km, na + 1)
        xcol = "a_km"
        xlabel = r"$a$ (km)"
    else:
        a_edges = np.linspace(aRH_min, aRH_max, na + 1)
        xcol = "a_over_RH"
        xlabel = r"$a/R_H$"

    e_edges = np.linspace(e_min, e_max, ne + 1)

    def panel_label(ax, txt):
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                fontsize=9)

    # Use tmax_mult from data (robust)
    tmax_mult = float(max(df_pro["t_over_Pbin"].max(), df_ret["t_over_Pbin"].max()))
    vmin, vmax = 0.0, tmax_mult

    # ---------- Escape median maps ----------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=160, sharex=True, sharey=True)
    last_im = None

    for row_idx, (tag, df) in enumerate([("Prograde", df_pro), ("Retrograde", df_ret)]):
        a = df[xcol].to_numpy(float)
        e = df["e"].to_numpy(float)
        i_eff = df["i_eff_deg"].to_numpy(float)
        esc = df["is_escape"].to_numpy(int) == 1
        val = df["t_over_Pbin"].to_numpy(float)

        for col_idx, (i0, i1) in enumerate(i_bins):
            ax = axes[row_idx, col_idx]
            in_i = (i_eff >= i0) & (i_eff < i1) & esc

            grid, _ = bin2d_median(a[in_i], e[in_i], val[in_i], a_edges, e_edges, min_count_escape)

            im = ax.pcolormesh(a_edges, e_edges, grid, shading="auto", vmin=vmin, vmax=vmax)
            last_im = im

            panel_label(ax, f"{tag} | $i_{{\\rm eff}} \\in [{i0},{i1})^\\circ$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$e$")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label(r"Median $t_{\rm esc}/P_{\rm bin}$ (escapes only)")
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(outdir / "F_truth_escape_median_2x3.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- Stable fraction maps ----------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=160, sharex=True, sharey=True)
    last_im = None

    for row_idx, (tag, df) in enumerate([("Prograde", df_pro), ("Retrograde", df_ret)]):
        a = df[xcol].to_numpy(float)
        e = df["e"].to_numpy(float)
        i_eff = df["i_eff_deg"].to_numpy(float)
        st = df["is_stable"].to_numpy(int)

        for col_idx, (i0, i1) in enumerate(i_bins):
            ax = axes[row_idx, col_idx]
            in_i = (i_eff >= i0) & (i_eff < i1)

            grid, _ = bin2d_stable_fraction(a[in_i], e[in_i], st[in_i], a_edges, e_edges, min_count_stable)

            im = ax.pcolormesh(a_edges, e_edges, grid, shading="auto", vmin=0.0, vmax=1.0)
            last_im = im

            panel_label(ax, f"{tag} | $i_{{\\rm eff}} \\in [{i0},{i1})^\\circ$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$e$")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Stable fraction")
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(outdir / "F_truth_stable_frac_2x3.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pm_physical.json")
    ap.add_argument("--pro", default="out/integrated_pm_pro.csv")
    ap.add_argument("--retro", default="out/integrated_pm_retro.csv")
    ap.add_argument("--outdir", default="out/figs_truth")
    ap.add_argument("--min-count-stable", type=int, default=30)
    ap.add_argument("--min-count-escape", type=int, default=10)
    ap.add_argument("--x-unit", choices=["aRH", "km"], default="aRH",
                    help="x-axis for maps: a/R_H or km (a around the PM barycenter).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, pbin, RH_km = load_cfg(args.config)

    df_pro = add_derived(pd.read_csv(args.pro), pbin, RH_km)
    df_ret = add_derived(pd.read_csv(args.retro), pbin, RH_km)

    plot_escape_ecdf(df_pro, df_ret, outdir / "F_truth_escape_ecdf.png")
    plot_stable_vs_a(df_pro, df_ret, outdir / "F_truth_stable_vs_a.png",
                     x_unit=args.x_unit, RH_km=RH_km)

    plot_2x3_maps(
        df_pro, df_ret, outdir,
        x_unit=args.x_unit, RH_km=RH_km,
        min_count_stable=args.min_count_stable,
        min_count_escape=args.min_count_escape
    )

    print("[OK] Wrote:")
    for fn in ["F_truth_escape_ecdf.png", "F_truth_stable_vs_a.png",
               "F_truth_escape_median_2x3.png", "F_truth_stable_frac_2x3.png"]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()

