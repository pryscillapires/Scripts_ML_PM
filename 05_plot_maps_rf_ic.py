#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

DATA = "out/results_all.csv"
MODEL_PKL = "out/ml/rf_ic_cal.pkl"
OUTDIR = Path("out/ml")
OUTDIR.mkdir(parents=True, exist_ok=True)

NBINS_A = 45
NBINS_E = 45

df = pd.read_csv(DATA)
df["y"] = (df["status"] == "stable").astype(int)

bundle = joblib.load(MODEL_PKL)
model = bundle["model"]
feat = bundle["feature_cols"]

X = df[feat].values
p = model.predict_proba(X)[:,1]
df["p_rf_ic"] = p

# bins
a = df["a_over_RH"].values
e = df["e"].values

a_edges = np.linspace(a.min(), a.max(), NBINS_A+1)
e_edges = np.linspace(e.min(), e.max(), NBINS_E+1)

# helper to aggregate per bin
def bin_mean(x, y, val):
    out = np.full((NBINS_E, NBINS_A), np.nan)   # rows=e, cols=a
    cnt = np.zeros((NBINS_E, NBINS_A), dtype=int)
    ia = np.digitize(x, a_edges) - 1
    ie = np.digitize(y, e_edges) - 1
    ok = (ia>=0)&(ia<NBINS_A)&(ie>=0)&(ie<NBINS_E)
    ia, ie = ia[ok], ie[ok]
    vv = val[ok]
    for j,i,v in zip(ie, ia, vv):
        if np.isnan(out[j,i]): out[j,i]=0.0
        out[j,i]+=v
        cnt[j,i]+=1
    out = out / np.where(cnt==0, np.nan, cnt)
    return out, cnt

Pmean, C = bin_mean(a, e, df["p_rf_ic"].values)
Ymean, _ = bin_mean(a, e, df["y"].values)

extent = [a_edges[0], a_edges[-1], e_edges[0], e_edges[-1]]

# --- plot: predicted vs observed (side-by-side)
plt.figure(figsize=(11,4.5))

plt.subplot(1,2,1)
im1 = plt.imshow(Pmean, origin="lower", extent=extent, aspect="auto")
plt.xlabel("a / R_H")
plt.ylabel("e")
plt.title("RF (IC-only): mean predicted P(stable)")
plt.colorbar(im1, fraction=0.046, pad=0.04)

plt.subplot(1,2,2)
im2 = plt.imshow(Ymean, origin="lower", extent=extent, aspect="auto")
plt.xlabel("a / R_H")
plt.ylabel("e")
plt.title("Observed stability fraction")
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(OUTDIR / "map_pred_vs_obs_rf_ic.png", dpi=220)
plt.close()

# --- optional: mask low-count bins + save count map
plt.figure()
im3 = plt.imshow(C, origin="lower", extent=extent, aspect="auto")
plt.xlabel("a / R_H")
plt.ylabel("e")
plt.title("Counts per bin")
plt.colorbar(im3, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUTDIR / "map_counts_bins.png", dpi=220)
plt.close()

print("Saved:")
print("  out/ml/map_pred_vs_obs_rf_ic.png")
print("  out/ml/map_counts_bins.png")

