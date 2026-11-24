"""Checks zielmodell.py (standalone, used in the ROS package) against einfach.EinfachModell.

Run after changing zielmodell.py or refitting the model:  python test_zielmodell.py [MODEL.json]
Exit code 0 = all checks pass.

1. Forward model: z, slope, sigma of a new shot and landing point agree on a theta x distance grid.
2. Monotonicity: dz/dtheta > 0 everywhere in the valid range (the inverse uses bisection).
3. Inverse: targets on a jet of angle theta give back theta, sigma_theta = sigma_shot / |dz/dtheta|.
4. Unreachable targets are rejected (reason codes printed).
"""

import os
import sys
import time

import numpy as np

import einfach
from zielmodell import DTHETA, Zielmodell

HERE = os.path.dirname(os.path.abspath(__file__))
TOL_Z = 1e-9        # [m]
TOL_LANDING = 1e-6  # [m]
TOL_THETA = 0.01    # [deg]
TOL_SIGMA = 0.01    # [deg]


def jet_targets(zm):
    """Targets on jets near the ends of the range and in between: (x, z, theta, label), experiment coordinates."""
    lo, hi = zm.theta_min, zm.theta_max
    t = []
    for th in (lo + 0.05, lo + 1.0, 0.5 * (lo + hi), hi - 1.0, hi - 0.05):
        xd, _ = zm.nozzle(th)
        xl = zm.landing(th)
        for k in (0.2, 0.5, 0.8, 0.97):
            x = xd + k * (xl - xd)
            t.append((x, zm.z(x, th), th, f"on jet {th:.2f} deg"))
        t.append((xl, 0.0, th, f"ground at landing {th:.2f} deg"))
    return t


def invalid_targets(zm):
    return [(5.0, 10.0, "too high"), (3.0, -1.0, "below ground"), (zm.xp - 1.0, 1.0, "behind nozzle"),
            (40.0, 0.0, "too far"), (zm.xp + 0.05, zm.zp, "at pivot")]


def main():
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="replace")
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "auswertung", "fit_modell_einfach.json")
    ref = einfach.lade_einfach(path)
    new = Zielmodell(path)
    errors = []

    # 1. forward model
    max_dz = max_dl = 0.0
    n_grid = 0
    for th in np.linspace(new.theta_min, new.theta_max, 15):
        xl_ref, _ = ref.aufprall(th)
        xl_new = new.landing(th)
        max_dl = max(max_dl, abs(xl_ref - xl_new))
        if not abs(xl_ref - xl_new) <= TOL_LANDING:
            errors.append(f"landing theta={th:.2f}: reference {xl_ref:.6f}, new {xl_new:.6f}")
        for x in np.linspace(ref.x_start(th) + 0.05, xl_ref, 12):
            n_grid += 1
            pairs = [("z", ref.z(x, th)[0], new.z(x, th)),
                     ("slope", ref.steigung(x, th)[0], new.slope(x, th)),
                     ("sigma_shot", ref.sigma_schuss(x, th)[0], new.sigma_shot(x, th)),
                     ("sigma_curve", ref.sigma_z(x, th)[0], new.sigma_curve(x, th))]
            for name, a, b in pairs:
                max_dz = max(max_dz, abs(a - b))
                if abs(a - b) > TOL_Z:
                    errors.append(f"{name} theta={th:.2f} x={x:.3f}: reference {a:.9f}, new {b:.9f}")

    # 2. monotonicity of z in theta (the inverse relies on it)
    min_dzdth = np.inf
    for th in np.linspace(new.theta_min, new.theta_max, 60):
        xl, _ = ref.aufprall(th)
        xs = np.linspace(ref.x_start(th) + 0.05, xl, 60)
        min_dzdth = min(min_dzdth, float(np.min((ref.z(xs, th + 0.01) - ref.z(xs, th - 0.01)) / 0.02)))
    if not min_dzdth > 0:
        errors.append(f"dz/dtheta not positive everywhere (min {min_dzdth:.4f} m/deg): inverse not unique")

    # 3. inverse
    max_dth = max_dsig = 0.0
    n_ok = 0
    for x, z, th, label in jet_targets(new):
        b = new.angle_for_target(x, z, frame="experiment", shot=True)
        if not b.ok:
            errors.append(f"x={x:.3f} z={z:.3f} [{label}]: not reached ({b.reason}: {b.message})")
            continue
        n_ok += 1
        dzdth = (ref.z(x, b.theta + DTHETA)[0] - ref.z(x, b.theta - DTHETA)[0]) / (2 * DTHETA)
        sig = ref.sigma_schuss(x, b.theta)[0] / abs(dzdth)
        dth, dsig = abs(b.theta - th), abs(b.sigma_theta - sig)
        max_dth, max_dsig = max(max_dth, dth), max(max_dsig, dsig)
        if dth > TOL_THETA or dsig > TOL_SIGMA:
            errors.append(f"x={x:.3f} z={z:.3f} [{label}]: theta {th:.4f} -> {b.theta:.4f}, "
                          f"sigma {sig:.4f} vs {b.sigma_theta:.4f}")

    # 4. unreachable targets
    print("Reason codes for invalid targets:")
    for x, z, label in invalid_targets(new):
        b = new.angle_for_target(x, z, frame="experiment")
        print(f"  {label:<14} -> {b.reason}")
        if b.ok:
            errors.append(f"x={x:.3f} z={z:.3f} [{label}]: accepted (theta {b.theta:.2f}), expected rejection")

    # timing
    sample = [(x, z) for x, z, _, _ in jet_targets(new)]
    t0 = time.perf_counter()
    for _ in range(10):
        for x, z in sample:
            new.angle_for_target(x, z, frame="experiment")
    t_new = (time.perf_counter() - t0) / (10 * len(sample))

    print(f"\nForward: {n_grid} grid points, max difference {max_dz:.1e} m, landing {max_dl:.1e} m")
    print(f"Monotonicity: smallest dz/dtheta = {min_dzdth * 100:.2f} cm/deg")
    print(f"Inverse: {n_ok} targets reached, max difference theta {max_dth:.1e} deg, sigma_theta {max_dsig:.1e} deg")
    print(f"Time per call: Zielmodell {t_new * 1000:.2f} ms")
    print(f"{len(errors)} mismatches")
    for e in errors:
        print("MISMATCH", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
