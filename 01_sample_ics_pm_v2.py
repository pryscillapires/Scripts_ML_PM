#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_sample_ics_pm.py (v2)

Generate ICs for circumbinary test particles around Patroclus–Menoetius.

Key idea:
- Define the reference plane as the *instantaneous binary plane* at the chosen epoch
  (ez || h_PM = r_PM x v_PM).
- Sample (a/RH, e, i) in that plane, with physically meaningful prograde/retrograde.
- Output states as *PM-barycentric relative* (x,y,z,vx,vy,vz) either:
    (a) in the binary-plane frame ("binaryplane"), or
    (b) rotated to the inertial frame ("inertial").

Downstream:
- If you output frame = "binaryplane": run 02_v2 with --ics-frame binaryplane
- If you output frame = "inertial": run 02_v2 with default --ics-frame inertial
"""

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import qmc


# ------------------------------------------------------------
# Binary plane basis from cfg states at epoch (inertial frame)
# ------------------------------------------------------------
def binary_plane_basis_from_cfg(cfg, epoch):
    stP = cfg["states"]["patroclus"][epoch]
    stM = cfg["states"]["menoetius"][epoch]

    r_rel = np.array([stM["x"] - stP["x"], stM["y"] - stP["y"], stM["z"] - stP["z"]], dtype=float)
    v_rel = np.array([stM["vx"] - stP["vx"], stM["vy"] - stP["vy"], stM["vz"] - stP["vz"]], dtype=float)

    nr = np.linalg.norm(r_rel)
    if nr == 0:
        raise RuntimeError("Zero relative separation (patroclus vs menoetius) at epoch.")
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
    return x*ex + y*ey + z*ez


# ------------------------------------------------------------
# Keplerian → Cartesian in a given reference plane (XY)
# Here, XY is the binary-plane frame.
# ------------------------------------------------------------
def keplerian_to_cartesian(a, e, inc, Omega, omega, f, mu):
    """
    Returns r, v in the *reference frame where the reference plane is XY*.
    If you interpret that frame as the binary-plane frame, then r,v are in binary-plane coords.
    """
    p = a * (1.0 - e**2)
    r_pf = p / (1.0 + e * np.cos(f))

    x_pf = r_pf * np.cos(f)
    y_pf = r_pf * np.sin(f)

    vx_pf = -np.sqrt(mu / p) * np.sin(f)
    vy_pf =  np.sqrt(mu / p) * (e + np.cos(f))

    cosO, sinO = np.cos(Omega), np.sin(Omega)
    cosi, sini = np.cos(inc), np.sin(inc)
    cosw, sinw = np.cos(omega), np.sin(omega)

    # Rotation from perifocal (pf) to reference frame
    R11 = cosO*cosw - sinO*sinw*cosi
    R12 = -cosO*sinw - sinO*cosw*cosi
    R21 = sinO*cosw + cosO*sinw*cosi
    R22 = -sinO*sinw + cosO*cosw*cosi
    R31 = sinw*sini
    R32 = cosw*sini

    r = np.array([R11*x_pf + R12*y_pf,
                  R21*x_pf + R22*y_pf,
                  R31*x_pf + R32*y_pf], dtype=float)

    v = np.array([R11*vx_pf + R12*vy_pf,
                  R21*vx_pf + R22*vy_pf,
                  R31*vx_pf + R32*vy_pf], dtype=float)

    return r, v


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pm_json", type=str, default="pm_physical.json")
    ap.add_argument("--outfile", type=str, default="out/ics/ics_pm.csv")

    # Prograde / Retrograde
    ap.add_argument("--sense", choices=["prograde", "retrograde"], default="prograde",
                    help="Defines how inclination is sampled w.r.t. the binary plane.")
    ap.add_argument("--i_max_deg", type=float, default=40.0,
                    help="Max inclination from the reference plane (prograde: [0,i_max], retrograde: [180-i_max,180]).")

    # Domain
    ap.add_argument("--aRH_min", type=float, default=0.4)
    ap.add_argument("--aRH_max", type=float, default=1.2)
    ap.add_argument("--e_min", type=float, default=0.0)
    ap.add_argument("--e_max", type=float, default=0.5)

    # Phase control (keep defaults fixed for CMDA main paper)
    ap.add_argument("--Omega", type=float, default=0.0, help="Ascending node in radians (binary-plane frame).")
    ap.add_argument("--omega", type=float, default=0.0, help="Argument of periapsis in radians (binary-plane frame).")
    ap.add_argument("--f_mode", choices=["fixed", "uniform"], default="fixed",
                    help="fixed: f=0. uniform: f ~ U[0,2pi).")
    ap.add_argument("--f_value", type=float, default=0.0, help="True anomaly when f_mode=fixed, in radians.")

    # Output frame: either keep in binary-plane coords or rotate to inertial
    ap.add_argument("--out_frame", choices=["binaryplane", "inertial"], default="inertial",
                    help="Frame used for (x,y,z,vx,vy,vz) stored in CSV.")

    args = ap.parse_args()

    with open(args.pm_json, "r") as f:
        cfg = json.load(f)

    epoch = cfg["epochs_utc"][0]
    RH = float(cfg["constants"]["R_H_pm_sun"])     # meters
    mu_PM = float(cfg["pm_binary"]["GM_total"])    # m^3/s^2

    # Build binary-plane basis at epoch (in inertial frame)
    ex, ey, ez = binary_plane_basis_from_cfg(cfg, epoch)

    # Latin Hypercube in (a/RH, e, i_draw)
    sampler = qmc.LatinHypercube(d=3, seed=args.seed)
    U = sampler.random(n=args.n_samples)

    a_over_RH = args.aRH_min + (args.aRH_max - args.aRH_min) * U[:, 0]
    e = args.e_min + (args.e_max - args.e_min) * U[:, 1]
    i_draw = 0.0 + args.i_max_deg * U[:, 2]  # degrees in [0, i_max]

    # Map to actual inclination
    if args.sense == "prograde":
        i_deg = i_draw
        sense_pro = 1
        sense = +1
    else:
        i_deg = 180.0 - i_draw  # -> [180-i_max, 180]
        sense_pro = 0
        sense = -1

    # True anomaly
    rng = np.random.default_rng(args.seed)
    if args.f_mode == "fixed":
        f_arr = np.full(args.n_samples, float(args.f_value), dtype=float)
    else:
        f_arr = rng.uniform(0.0, 2.0*np.pi, size=args.n_samples)

    rows = []
    for k in range(args.n_samples):
        a = a_over_RH[k] * RH
        inc = np.deg2rad(i_deg[k])

        # In the binary-plane frame:
        r_bin, v_bin = keplerian_to_cartesian(
            a=a,
            e=e[k],
            inc=inc,
            Omega=float(args.Omega),
            omega=float(args.omega),
            f=float(f_arr[k]),
            mu=mu_PM
        )

        # Choose output frame for stored (x,y,z,vx,vy,vz)
        if args.out_frame == "binaryplane":
            r_out, v_out = r_bin, v_bin
        else:
            r_out = rotate_from_binaryplane_to_inertial(r_bin, ex, ey, ez)
            v_out = rotate_from_binaryplane_to_inertial(v_bin, ex, ey, ez)

        rows.append({
            "id": k,
            "epoch": epoch,
            "epoch_id": epoch,        # for future grouping
            "regime": "circumbin",
            "regime_id": 0,           # keep int for ML (single epoch/run)
            "sense": sense,           # +1 prograde, -1 retrograde
            "sense_pro": sense_pro,   # 1 prograde, 0 retrograde

            "a_over_RH": float(a_over_RH[k]),
            "e": float(e[k]),
            "i_deg": float(i_deg[k]),

            "Omega": float(args.Omega),
            "omega": float(args.omega),
            "f": float(f_arr[k]),
            "ics_frame": args.out_frame,

            # PM-barycentric relative state (units: m, m/s)
            "x": float(r_out[0]), "y": float(r_out[1]), "z": float(r_out[2]),
            "vx": float(v_out[0]), "vy": float(v_out[1]), "vz": float(v_out[2]),
        })

    df = pd.DataFrame(rows)
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outfile, index=False)

    print("[OK] IC sampling completed")
    print(f"  N = {args.n_samples}")
    print(f"  epoch = {epoch}")
    print(f"  sense = {args.sense}  (i_max_deg={args.i_max_deg})")
    print(f"  out_frame = {args.out_frame}")
    print(f"  saved to {args.outfile}")


if __name__ == "__main__":
    main()

