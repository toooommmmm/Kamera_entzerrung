"""Compares zielmodell.py (standalone) with geometrie.GeometrieModell.winkel_fuer_ziel.

Run after changing zielmodell.py or the model:  python test_zielmodell.py [MODEL.json]
Exit code 0 = all targets agree.
"""

import os
import sys
import time

import numpy as np

import geometrie
from zielmodell import Zielmodell

HERE = os.path.dirname(os.path.abspath(__file__))
TOL_THETA = 0.01   # [deg]
TOL_SIGMA = 0.01   # [deg]


def targets(z_model):
    """Targets in experiment coordinates (x, z, label)."""
    t = []
    lo, hi = z_model.theta_min, z_model.theta_max
    # grid over the working area
    for x in np.arange(1.0, 13.01, 1.0):
        for z in np.arange(0.0, 4.01, 0.5):
            t.append((float(x), float(z), "grid"))
    # points on jets near the ends of the range and in between
    for th in (lo + 0.05, lo + 1.0, 0.5 * (lo + hi), hi - 1.0, hi - 0.05):
        xd, _ = z_model.nozzle(th)
        xl = z_model.landing(th)
        for k in (0.2, 0.5, 0.8, 0.97):
            x = xd + k * (xl - xd)
            t.append((x, z_model.z(x, th), f"on jet {th:.2f} deg"))
        t.append((xl, 0.0, f"ground at landing {th:.2f} deg"))
        t.append((xl + 0.03, 0.0, f"ground +3 cm {th:.2f} deg"))
        t.append((xl + 0.30, 0.0, f"ground +30 cm {th:.2f} deg"))
    # invalid targets
    t += [(5.0, 10.0, "too high"), (3.0, -1.0, "below ground"), (z_model.xp - 1.0, 1.0, "behind nozzle"),
          (40.0, 0.0, "too far"), (z_model.xp + 0.05, z_model.zp, "at pivot")]
    return t


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="replace")
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "auswertung", "fit_modell_geometrie.json")
    ref = geometrie.lade_geometrie(path)
    new = Zielmodell(path)

    errors, n_ok, n_invalid = [], 0, 0
    max_dth, max_dsig = 0.0, 0.0
    for x, z, label in targets(new):
        a = ref.winkel_fuer_ziel(x, z, bezug="boden", schuss=True)
        b = new.angle_for_target(x, z, frame="experiment", shot=True)
        ok_a = a["grund"] == "ok"
        if ok_a != b.ok:
            errors.append(f"x={x:.3f} z={z:.3f} [{label}]: reference {a['grund']!r}, new {b.reason!r} ({b.message})")
            continue
        if not ok_a:
            n_invalid += 1
            continue
        n_ok += 1
        dth, dsig = abs(a["theta"] - b.theta), abs(a["sigma_theta"] - b.sigma_theta)
        max_dth, max_dsig = max(max_dth, dth), max(max_dsig, dsig)
        if dth > TOL_THETA or dsig > TOL_SIGMA:
            errors.append(f"x={x:.3f} z={z:.3f} [{label}]: theta {a['theta']:.4f} vs {b.theta:.4f}, "
                          f"sigma {a['sigma_theta']:.4f} vs {b.sigma_theta:.4f}")

    # invalid targets: show reason codes
    print("Reason codes for invalid targets:")
    for x, z, label in targets(new)[-5:]:
        print(f"  {label:<14} -> {new.angle_for_target(x, z, frame='experiment').reason}")

    # timing
    sample = [(x, z) for x, z, _ in targets(new)][:60]
    t0 = time.perf_counter()
    for x, z in sample:
        ref.winkel_fuer_ziel(x, z, bezug="boden")
    t_ref = (time.perf_counter() - t0) / len(sample)
    t0 = time.perf_counter()
    for _ in range(10):
        for x, z in sample:
            new.angle_for_target(x, z, frame="experiment")
    t_new = (time.perf_counter() - t0) / (10 * len(sample))

    print(f"\nTargets: {n_ok} reachable, {n_invalid} unreachable, {len(errors)} mismatches")
    print(f"Max difference: theta {max_dth:.2e} deg, sigma_theta {max_dsig:.2e} deg")
    print(f"Time per call: GeometrieModell {t_ref * 1000:.1f} ms, Zielmodell {t_new * 1000:.2f} ms")
    for e in errors:
        print("MISMATCH", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
