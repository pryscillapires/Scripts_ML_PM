#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03b_survival_curves.py

Survival/decay plots from direct N-body integration:
- Kaplan–Meier survival S(t) vs t/P_bin (treating 'stable' as censored at t_max)
- Simple "still alive" fraction: N(t_end >= t)/N (empirical decay)
- Prints median/quantiles of escape times (escapes only)

Outputs:
  F_truth_survival_KM.png
  F_truth_survival_alive_fraction.png
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


def km_survival(times: np.ndarray, events: np.ndarray):
    """
    Kaplan–Meier estimator.
    times: non-negative, shape (N,)
    events: 1 if event (escape), 0 if censored (stable), shape (N,)
    Returns step arrays (t, S), including t=0, S=1.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    m = np.isfinite(times) & (times >= 0)
    times = times[m]
    events = events[m]

    if times.size == 0:
        return np.array([0.0]), np.array([1.0])

    # Unique event times only (where events==1)
    ev_times = np.unique(times[events == 1])
    if ev_times.size == 0:
        # no events -> survival stays 1 to max observed time
        return np.array([0.0, times.max()]), np.array([1.0, 1.0])

    # At-risk at time t: those with time >= t
    # Events at time t: those with event==1 and time == t
    S = 1.0
    t_steps = [0.0]
    S_steps = [1.0]

    for t in ev_times:
        n_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        if n_risk > 0:
            S *= (1.0 - d / n_risk)
        t_steps.append(float(t))
        S_steps.append(float(S))

    return np.array(t_steps), np.array(S_steps)


def alive_fraction_curve(times: np.ndarray, t_grid: np.ndarray):
    """
    Empirical fraction alive: N(t_end >= t)/N.
    """
    times = np.asarray(times, dtype=float)
    m = np.isfinite(times) & (times >= 0)
    times = times[m]
    if times.size == 0:
        return np.full_like(t_grid, np.nan, dtype=float)
    return np.array([np.mean(times >= t) for t in t_grid], dtype=float)


def summarize_escape_times(df: pd.DataFrame, label: str):
    esc = df.loc[df["is_escape"] == 1, "t_over_Pbin"].to_numpy(float)
    if esc.size == 0:
        print(f"[{label}] No escapes.")
        return
    q50 = np.quantile(esc, 0.50)
    q16 = np.quantile(esc, 0.16)
    q84 = np.quantile(esc, 0.84)
    print(f"[{label}] escapes N={esc.size} | median={q50:.3g} P_bin | (16,84)%=({q16:.3g},{q84:.3g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pm_physical.json")
    ap.add_argument("--pro", default="out/integrated_pm_pro.csv")
    ap.add_argument("--retro", default="out/integrated_pm_retro.csv")
    ap.add_argument("--outdir", default="out/figs_truth")
    ap.add_argument("--time-unit", choices=["pbin", "s"], default="pbin",
                    help="x-axis unit for survival curves: t/P_bin (pbin) or seconds (s).")
    ap.add_argument("--tgrid", type=int, default=250,
                    help="Number of points in the alive-fraction grid.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, pbin, RH_km = load_cfg(args.config)

    df_pro = add_derived(pd.read_csv(args.pro), pbin, RH_km)
    df_ret = add_derived(pd.read_csv(args.retro), pbin, RH_km)

    # Choose time column
    if args.time_unit == "s":
        tcol = "t_end"
        xlabel = r"$t$ (s)"
    else:
        tcol = "t_over_Pbin"
        xlabel = r"$t/P_{\rm bin}$"

    # Event definition: escape is event, stable is censored.
    t_pro = df_pro[tcol].to_numpy(float)
    e_pro = df_pro["is_escape"].to_numpy(int)

    t_ret = df_ret[tcol].to_numpy(float)
    e_ret = df_ret["is_escape"].to_numpy(int)

    # Summaries (escapes only)
    summarize_escape_times(df_pro if args.time_unit == "pbin" else df_pro.assign(t_over_Pbin=df_pro["t_end"]/pbin),
                           "Prograde")
    summarize_escape_times(df_ret if args.time_unit == "pbin" else df_ret.assign(t_over_Pbin=df_ret["t_end"]/pbin),
                           "Retrograde")

    # ---------- Kaplan–Meier survival ----------
    tp, Sp = km_survival(t_pro, e_pro)
    tr, Sr = km_survival(t_ret, e_ret)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.step(tp, Sp, where="post", label="Prograde (KM)")
    ax.step(tr, Sr, where="post", label="Retrograde (KM)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N(t_end ≥ t)/N")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "F_truth_survival_KM.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- Alive fraction curve ----------
    tmax = float(np.nanmax([np.nanmax(t_pro), np.nanmax(t_ret)]))
    t_grid = np.linspace(0.0, tmax, args.tgrid)

    ap_frac = alive_fraction_curve(t_pro, t_grid)
    ar_frac = alive_fraction_curve(t_ret, t_grid)

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.plot(t_grid, ap_frac, label="Prograde")
    ax.plot(t_grid, ar_frac, label="Retrograde")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N(t_end ≥ t)/N")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "F_truth_survival_alive_fraction.png", bbox_inches="tight")
    plt.close(fig)

    print("[OK] Wrote:")
    print(" ", outdir / "F_truth_survival_KM.png")
    print(" ", outdir / "F_truth_survival_alive_fraction.png")


if __name__ == "__main__":
    main()

