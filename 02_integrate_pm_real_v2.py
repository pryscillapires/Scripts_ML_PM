#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_integrate_pm_real_v3.py

REAL heliocentric Patroclus–Menoetius system
(Sun + Jupiter + Saturn + Patroclus + Menoetius + test particles)

- Supports ICs stored either in:
  (a) inertial frame ("inertial")  -> default when 01_v2 used --out_frame inertial
  (b) binary-plane frame ("binaryplane") -> rotates to inertial before adding

Outputs one CSV per chunk with per-orbit classification + distance diagnostics.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rebound


# ------------------------------------------------------------
# Binary plane basis (same definition as 01_v2)
# ------------------------------------------------------------
def binary_plane_basis_from_cfg(cfg, epoch):
    stP = cfg["states"]["patroclus"][epoch]
    stM = cfg["states"]["menoetius"][epoch]

    r_rel = np.array([stM["x"] - stP["x"], stM["y"] - stP["y"], stM["z"] - stP["z"]], dtype=float)
    v_rel = np.array([stM["vx"] - stP["vx"], stM["vy"] - stP["vy"], stM["vz"] - stP["vz"]], dtype=float)

    nr = np.linalg.norm(r_rel)
    if nr == 0:
        raise RuntimeError("Zero relative separation (Patroclus vs Menoetius) at epoch.")
    ex = r_rel / nr

    h = np.cross(r_rel, v_rel)
    nh = np.linalg.norm(h)
    if nh == 0:
        raise RuntimeError("Zero binary angular momentum at epoch (degenerate plane).")
    ez = h / nh

    ey = np.cross(ez, ex)
    ney = np.linalg.norm(ey)
    if ney == 0:
        raise RuntimeError("Degenerate binary basis (ey ~ 0).")
    ey = ey / ney

    return ex, ey, ez


def rotate_from_binaryplane_to_inertial(vec_bin, ex, ey, ez):
    x, y, z = vec_bin
    return x * ex + y * ey + z * ez


# ------------------------------------------------------------
# Initialize heliocentric system
# IMPORTANT: ordering is:
#   0 Sun, 1 Jupiter, 2 Saturn, 3 Patroclus, 4 Menoetius
# ------------------------------------------------------------
def init_sim(cfg, epoch):
    sim = rebound.Simulation()
    sim.G = float(cfg["constants"]["G"])
    sim.integrator = "ias15"

    # Sun at origin (as provided)
    sim.add(m=float(cfg["constants"]["M_sun"]), x=0, y=0, z=0, vx=0, vy=0, vz=0)

    for body in ["jupiter", "saturn", "patroclus", "menoetius"]:
        sim.add(**cfg["states"][body][epoch])

    return sim


# ------------------------------------------------------------
# Add test particle: PM-barycentric relative -> heliocentric inertial
# ------------------------------------------------------------
def add_particle(sim, cfg, row, epoch, ics_frame_mode="auto", basis=None):
    # Massive bodies
    pP = sim.particles[3]
    pM = sim.particles[4]

    mP = float(cfg["bodies"]["patroclus"]["mass_kg"])
    mM = float(cfg["bodies"]["menoetius"]["mass_kg"])
    Mtot = mP + mM

    # PM barycenter (inertial, heliocentric coordinates)
    r_bary = (mP*np.array([pP.x,  pP.y,  pP.z])  + mM*np.array([pM.x,  pM.y,  pM.z]))  / Mtot
    v_bary = (mP*np.array([pP.vx, pP.vy, pP.vz]) + mM*np.array([pM.vx, pM.vy, pM.vz])) / Mtot

    # Decide IC frame
    if ics_frame_mode == "auto":
        ics_frame = str(row.get("ics_frame", "inertial")).strip().lower()
    else:
        ics_frame = ics_frame_mode.strip().lower()

    # Read relative state from CSV
    r_rel = np.array([row["x"], row["y"], row["z"]], dtype=float)
    v_rel = np.array([row["vx"], row["vy"], row["vz"]], dtype=float)

    # If ICs are in binaryplane, rotate them to inertial
    if ics_frame == "binaryplane":
        if basis is None:
            basis = binary_plane_basis_from_cfg(cfg, epoch)
        ex, ey, ez = basis
        r_rel = rotate_from_binaryplane_to_inertial(r_rel, ex, ey, ez)
        v_rel = rotate_from_binaryplane_to_inertial(v_rel, ex, ey, ez)

    elif ics_frame != "inertial":
        raise ValueError(f"Unknown ics_frame='{ics_frame}'. Expected 'inertial' or 'binaryplane'.")

    # Add particle in heliocentric inertial coords
    sim.add(
        m=0.0,
        x=float(r_bary[0] + r_rel[0]),
        y=float(r_bary[1] + r_rel[1]),
        z=float(r_bary[2] + r_rel[2]),
        vx=float(v_bary[0] + v_rel[0]),
        vy=float(v_bary[1] + v_rel[1]),
        vz=float(v_bary[2] + v_rel[2]),
    )

    return len(sim.particles) - 1


# ------------------------------------------------------------
# Integrate and classify
# ------------------------------------------------------------
def integrate_orbit(sim, cfg, idx, tmax, n_steps=500, k_escape=3.0):
    mP = float(cfg["bodies"]["patroclus"]["mass_kg"])
    mM = float(cfg["bodies"]["menoetius"]["mass_kg"])
    Mtot = mP + mM

    R_P = float(cfg["bodies"]["patroclus"]["radius_m"])
    R_M = float(cfg["bodies"]["menoetius"]["radius_m"])
    RH  = float(cfg["constants"]["R_H_pm_sun"])

    status = "stable"
    min_rP = np.inf
    min_rM = np.inf
    max_rB = 0.0

    dt = float(tmax) / float(n_steps)
    steps = 0

    try:
        while sim.t < tmax and steps < n_steps:
            steps += 1
            t_next = min(sim.t + dt, tmax)
            sim.integrate(t_next)

            p  = sim.particles[idx]
            pP = sim.particles[3]
            pM = sim.particles[4]

            # distances to primaries
            dP = np.linalg.norm([p.x - pP.x, p.y - pP.y, p.z - pP.z])
            dM = np.linalg.norm([p.x - pM.x, p.y - pM.y, p.z - pM.z])

            # distance to PM barycenter
            rB_vec = np.array([
                p.x - (mP*pP.x + mM*pM.x)/Mtot,
                p.y - (mP*pP.y + mM*pM.y)/Mtot,
                p.z - (mP*pP.z + mM*pM.z)/Mtot
            ], dtype=float)
            rB = float(np.linalg.norm(rB_vec))

            min_rP = min(min_rP, float(dP))
            min_rM = min(min_rM, float(dM))
            max_rB = max(max_rB, rB)

            # events
            if dP <= R_P:
                status = "collision_P"
                break
            if dM <= R_M:
                status = "collision_M"
                break
            if rB >= float(k_escape) * RH:
                status = "escape_local"
                break

    except Exception:
        status = "integration_error"

    return status, float(sim.t), float(min_rP), float(min_rM), float(max_rB), int(steps)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pm_physical.json")
    ap.add_argument("--ics", default="out/ics/ics_pm.csv")
    ap.add_argument("--ics-frame", choices=["auto", "inertial", "binaryplane"], default="auto",
                    help="How to interpret (x,y,z,vx,vy,vz) stored in the IC CSV. "
                         "auto uses column 'ics_frame' if present, else assumes inertial.")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--tmax-mult", type=float, default=500.0)
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--k-escape", type=float, default=3.0)
    ap.add_argument("--outdir", default="out/integrated")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    epoch = cfg["epochs_utc"][0]
    Pbin = float(cfg["pm_binary"]["P_bin_s"])
    tmax = float(args.tmax_mult) * Pbin

    df = pd.read_csv(args.ics)
    # chunking
    df = df[df["id"] % args.nchunks == args.chunk].copy()

    # precompute basis once (only used if needed)
    basis = None
    if args.ics_frame in ("binaryplane", "auto"):
        # safe to compute even if you won't use; cheap
        basis = binary_plane_basis_from_cfg(cfg, epoch)

    out_rows = []
    for _, row in df.iterrows():
        sim = init_sim(cfg, epoch)
        idx = add_particle(sim, cfg, row, epoch, ics_frame_mode=args.ics_frame, basis=basis)

        status, t_end, min_rP, min_rM, max_rB, steps = integrate_orbit(
            sim, cfg, idx, tmax=tmax, n_steps=args.n_steps, k_escape=args.k_escape
        )

        out_rows.append({
            "id": int(row["id"]),
            "epoch": str(row.get("epoch", epoch)),
            "epoch_id": str(row.get("epoch_id", epoch)),
            "regime": row.get("regime", "circumbin"),
            "regime_id": int(row.get("regime_id", 0)),
            "sense": int(row.get("sense", 0)),
            "sense_pro": int(row.get("sense_pro", 0)),

            "a_over_RH": float(row["a_over_RH"]),
            "e": float(row["e"]),
            "i_deg": float(row["i_deg"]),

            "status": status,
            "t_end": float(t_end),
            "n_steps_done": int(steps),
            "min_rP": float(min_rP),
            "min_rM": float(min_rM),
            "max_rB": float(max_rB),
        })

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    outfile = Path(args.outdir) / f"partial_chunk{args.chunk:03d}.csv"
    pd.DataFrame(out_rows).to_csv(outfile, index=False)
    print(f"[OK] Saved {len(out_rows)} rows to {outfile}")


if __name__ == "__main__":
    main()

