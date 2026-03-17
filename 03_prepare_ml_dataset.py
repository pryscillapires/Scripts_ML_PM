#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_prepare_ml_dataset.py

Prepare ML-ready dataset from N-body results
(Patroclus–Menoetius REAL system).

Output:
  out/ml/features_pm_real.csv
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
INPUT  = "out/results_all.csv"
OUTPUT = "out/ml/features_pm_real.csv"

Path("out/ml").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# Label
# ------------------------------------------------------------
df["y"] = (df["status"] == "stable").astype(int)

# ------------------------------------------------------------
# Feature selection
# ------------------------------------------------------------
FEATURES = [
    "a_over_RH",
    "e",
    "i_deg",
    "min_rP",
    "min_rM",
    "max_rB",
]

TARGET = "y"

cols = FEATURES + [TARGET]

df_ml = df[cols].copy()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
df_ml.to_csv(OUTPUT, index=False)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
print("ML dataset prepared")
print(f"  Input : {INPUT}")
print(f"  Output: {OUTPUT}")
print(f"  N rows: {len(df_ml)}")
print("\nLabel balance:")
print(df_ml["y"].value_counts(normalize=True))

