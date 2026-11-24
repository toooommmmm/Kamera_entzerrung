"""Checks zielmodell.py (standalone, used in the ROS package) against einfach.EinfachModell.

Run after changing zielmodell.py or refitting the model:  python test_zielmodell.py [MODEL.json]
Exit code 0 = all checks pass.

1. Forward model: z, slope, sigma of a new shot and landing point agree on a theta x distance grid.
2. Monotonicity: dz/dtheta > 0 everywhere in the valid range (the inverse uses bisection).
3. Inverse: targets on a jet of angle theta give back theta, sigma_theta = sigma_shot / |dz/dtheta|.
4. Unreachable targets are rejected (reason codes printed).
5. Ballistic extension (only if the JSON has a "ballistik" block):
   forward model against ballistik.BallistikModell (+ continuity correction), continuity at theta_min,
   dz/dtheta > 0 over theta_lo .. theta_max, round trip target -> theta -> target for theta below theta_min
   (also below the ground of the test setup), and the same JSON without the block behaves exactly as before.
"""

import json
import os
import sys
import tempfile
import time

import numpy as np

import einfach
import zielmodell
from zielmodell import DTHETA, Zielmodell

HERE = os.path.dirname(os.path.abspath(__file__))
TOL_Z = 1e-9        # [m]
TOL_LANDING = 1e-6  # [m]
TOL_THETA = 0.01    # [deg]
TOL_SIGMA = 0.01    # [deg]
TOL_Z_BAL = 2e-3    # [m] ballistic branch: correction is tabulated in zielmodell, exact in the reference
TOL_SLOPE_BAL = 0.02
TOL_SIGMA_BAL = 2e-3  # [m]
TOL_JUMP = 0.01     # [m] continuity at theta_min


def check_ballistic(path, new, errors):
    """Section 5. -> list of summary lines."""
    import ballistik
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    b = d["ballistik"]
    ref_e0 = einfach.lade_einfach(path)
    bal = ballistik.BallistikModell(b["v0"], b["kp"], np.array(b["kovarianz"]), b["duese"], g=b["g"],
                                    s_eps=b["s_eps_rad"], s_alpha=b["s_alpha"], dt=b["dt"])
    c_mf = b["modellfehler"]["m_pro_grad_pro_m"]
    tmin, tlo = new.theta_min, new.theta_lo
    xd_min = ref_e0.x_start(tmin)
    dx_max = ref_e0.aufprall(tmin)[0] - xd_min
    corr_on = new._corr is not None

    def corr(x, th):
        if not corr_on:
            return 0.0
        w = max(0.0, 1.0 - (tmin - th) / zielmodell.FADE_DEG)
        u = min(max(x - bal.x_start(th), 0.0), dx_max)
        return w * float(ref_e0.z(xd_min + u, tmin)[0] - bal.z(xd_min + u, tmin)[0])

    # 5a forward
    max_dz = max_ds = max_dsig = 0.0
    n = 0
    for th in np.linspace(tlo, tmin - 0.01, 12):
        for x in bal.x_start(th) + np.linspace(0.05, 8.0, 15):
            n += 1
            dx = x - bal.x_start(th)
            z_ref = bal.z(x, th)[0] + corr(x, th)
            s_ref = bal.steigung(x, th)[0] + (corr(x + 1e-4, th) - corr(x - 1e-4, th)) / 2e-4
            sig_ref = np.sqrt(bal.sigma_schuss(x, th)[0] ** 2 + (c_mf * (tmin - th) * dx) ** 2)
            dz, ds, dsig = abs(z_ref - new.z(x, th)), abs(s_ref - new.slope(x, th)), abs(sig_ref - new.sigma_shot(x, th))
            max_dz, max_ds, max_dsig = max(max_dz, dz), max(max_ds, ds), max(max_dsig, dsig)
            if dz > TOL_Z_BAL or ds > TOL_SLOPE_BAL or dsig > TOL_SIGMA_BAL:
                errors.append(f"ballistic theta={th:.2f} x={x:.2f}: z {z_ref:.4f}/{new.z(x, th):.4f}, slope "
                              f"{s_ref:.4f}/{new.slope(x, th):.4f}, sigma_shot {sig_ref:.4f}/{new.sigma_shot(x, th):.4f}")

    # 5b continuity at theta_min (along the jet up to the E0 landing point)
    jump_raw = max(abs(float(ref_e0.z(xd_min + u, tmin)[0] - bal.z(xd_min + u, tmin)[0]))
                   for u in np.linspace(0.05, dx_max, 40))
    jump = max(abs(new.z(xd_min + u, tmin - 1e-7) - new.z(xd_min + u, tmin)) for u in np.linspace(0.05, dx_max, 40))
    if jump > TOL_JUMP:
        errors.append(f"jump at theta_min = {jump * 100:.1f} cm")

    # 5c monotonicity over the whole range
    min_dzdth, where = np.inf, None
    for th in np.linspace(tlo + 0.02, new.theta_max - 0.02, 120):
        x_end = new.landing(th) if th >= tmin else bal.x_start(th) + 8.0
        if not np.isfinite(x_end):
            continue
        for x in np.linspace(new.nozzle(th)[0] + 0.05, x_end, 30):
            v = (new.z(x, th + 0.01) - new.z(x, th - 0.01)) / 0.02
            if v < min_dzdth:
                min_dzdth, where = v, (th, x)
    if not min_dzdth > 0:
        errors.append(f"dz/dtheta not positive over theta_lo..theta_max (min {min_dzdth:.4f} m/deg at {where})")

    # 5d round trip in the ballistic branch, incl. targets below the ground of the test setup
    max_dth, n_ok, n_below = 0.0, 0, 0
    for th in (tlo + 0.05, -15.0, -10.0, tmin - FADE_MID, tmin - 0.05):
        for dx in (0.5, 2.0, 4.0, 6.0):
            x = new.nozzle(th)[0] + dx
            z = new.z(x, th)
            r = new.angle_for_target(x, z, frame="experiment")
            if not r.ok or r.branch != "ballistic":
                errors.append(f"round trip theta={th:.2f} dx={dx}: {r.reason} {r.branch} ({r.message})")
                continue
            n_ok += 1
            n_below += z < 0
            max_dth = max(max_dth, abs(r.theta - th))
            if abs(r.theta - th) > TOL_THETA:
                errors.append(f"round trip theta={th:.3f} dx={dx}: got {r.theta:.4f}")

    # 5e old JSON without the block: identical results
    d_old = {k: v for k, v in d.items() if k != "ballistik"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(d_old, f)
        old_path = f.name
    old = Zielmodell(old_path)
    os.unlink(old_path)
    n_same, n_diff = 0, 0
    for x, z, _, _ in jet_targets(old):
        a, c = old.angle_for_target(x, z, frame="experiment"), new.angle_for_target(x, z, frame="experiment")
        if (a.reason, round(a.theta, 9) if a.ok else None) == (c.reason, round(c.theta, 9) if c.ok else None):
            n_same += 1
        else:
            n_diff += 1
            errors.append(f"E0 target x={x:.2f} z={z:.2f}: without block {a.reason} {a.theta}, with {c.reason} {c.theta}")
    xb = old.nozzle(-10.0)[0] + 3.0
    a = old.angle_for_target(xb, new.z(xb, -10.0), frame="experiment")
    if a.reason != "too_low" or old.theta_lo != old.theta_min or old.bal is not None:
        errors.append(f"JSON without block: target below theta_min gives {a.reason}, expected too_low")

    t0 = time.perf_counter()
    for th in (-18.0, -12.0, -6.0):
        x = new.nozzle(th)[0] + 4.0
        new.angle_for_target(x, new.z(x, th), frame="experiment")
    t_bal = (time.perf_counter() - t0) / 3
    return [f"Ballistic branch (theta {tlo:g} .. {tmin:g} deg, v0 = {b['v0']:.2f} m/s, k = {b['kp'][0]:.4f} 1/m):",
            f"  forward: {n} points, max difference z {max_dz:.1e} m, slope {max_ds:.1e}, sigma_shot {max_dsig:.1e} m",
            f"  jump at theta_min: {jump_raw * 100:.1f} cm without correction, {jump * 100:.2f} cm with "
            f"(correction {'on' if corr_on else 'off'})",
            f"  monotonicity theta_lo .. theta_max: smallest dz/dtheta = {min_dzdth * 100:.2f} cm/deg",
            f"  round trip: {n_ok} targets ({n_below} below the test ground), max difference theta {max_dth:.1e} deg",
            f"  JSON without block: {n_same} E0 targets identical, {n_diff} different; below theta_min -> {a.reason}",
            f"  time per ballistic call: {t_bal * 1000:.1f} ms"]


FADE_MID = 0.5 * zielmodell.FADE_DEG


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
    if new.bal is not None:
        print()
        print("\n".join(check_ballistic(path, new, errors)))
    print(f"{len(errors)} mismatches")
    for e in errors:
        print("MISMATCH", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
