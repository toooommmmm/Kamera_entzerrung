"""Standalone aiming model: target position -> absolute nozzle angle.

Reads the fitted jet model E (fit_modell_einfach.json, written by einfach.py in Kamera_entzerrung)
and inverts it. Standard library only, so it can be copied into the ROS package as is.
After a refit only the JSON has to be copied again.

Model E (all lengths in m, theta = absolute nozzle angle, t = theta / skala_grad):
    nozzle exit  x_d = x_p + L cos(theta) - h sin(theta),  z_d = z_p + L sin(theta) + h cos(theta)
    jet          dz = (tan(theta) + d(t)) dx + a(t) dx^2 + b(t) dx^3,  dx = x - x_d, dz = z - z_d
                 d, a, b polynomials in t (d optional, grad_d = -1: none)
    shot scatter variance  s_eps^2 dx^2 + s_alpha^2 dx^4

Optional ballistic extension below the measured range (JSON block "ballistik", written by strahl.py fit):
    for theta_min_ballistic <= theta < theta_min (e.g. -20 .. -3 deg) a point mass with quadratic drag,
    fitted on the flat shots, same nozzle exit:
        x(0) = x_d, z(0) = z_d, v(0) = v0 (cos theta, sin theta)
        x'' = -k |v| vx,  z'' = -g - k |v| vz        (RK4, fixed dt, z linear between samples)
    plus a correction c(dx) = z_E0(theta_min) - z_ballistic(theta_min), faded linearly to 0 over FADE_DEG,
    so that z is continuous at theta_min. Uncertainty: curve (v0, k covariance) + shot scatter + model error
    m_per_deg_per_m * (theta_min - theta) * dx. The ground check (landing) is skipped in this branch: in flight
    the target may lie below the ground of the test setup. Without the block: behaviour as before.

Usage:
    m = Zielmodell("fit_modell_einfach.json")
    r = m.angle_for_target(x, z)      # x, z [m] relative to the servo pivot
    if r.ok:
        command(r.theta)              # absolute nozzle angle [deg], r.sigma_theta = 1-sigma [deg]
    else:
        log(r.reason, r.message)

Conventions (same as einfach.py):
    x  horizontal distance in the vertical plane of the jet [m], z height [m]
    theta  absolute nozzle angle against the horizon [deg] (inclinometer, not servo command)
    frame="pivot":      x, z relative to the servo pivot (for the drone)
    frame="experiment": x, z in the coordinates of the test setup (posts, ground at z = 0)
Valid only for theta inside the measured range and between nozzle and landing point (ballistic branch:
theta down to theta_min_ballistic, no landing check).

Quick check:  python zielmodell.py fit_modell_einfach.json X Z
"""

import json
import math
import sys
from dataclasses import dataclass

LANDING_TOLERANCE = 0.05   # [m] targets at ground level lie exactly on the landing point
LANDING_STEP = 0.05        # [m] step for the zero-crossing search
LANDING_MAX = 40.0         # [m] search range behind the nozzle
DTHETA = 0.01              # [deg] step for the numerical derivative dz/dtheta
FADE_DEG = 5.0             # [deg] ballistic branch: continuity correction fades out over this range below theta_min
JUMP_TOL = 0.03            # [m] correction only if E0 and ballistic differ by more than this at theta_min
CORR_STEP = 0.02           # [m] grid of the correction table along the jet
Z_TOL = 0.01               # [m] a solution must hit the target height within this
T_MAX = 8.0                # [s] ballistic integration limit


@dataclass
class Result:
    ok: bool
    theta: float = math.nan         # absolute nozzle angle [deg]
    sigma_theta: float = math.nan   # 1-sigma uncertainty of theta [deg]
    sigma_z: float = math.nan       # 1-sigma height uncertainty at the target [m]
    dz_dtheta: float = math.nan     # sensitivity at the target [m/deg]
    reason: str = ""                # ok, invalid_input, too_close, too_far, too_high, too_low, beyond_landing
    message: str = ""               # readable explanation for logs
    x_exp: float = math.nan         # target in experiment coordinates [m]
    z_exp: float = math.nan
    branch: str = ""                # "E0" (measured range) or "ballistic" (extrapolated below theta_min)


class Zielmodell:
    def __init__(self, json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("art") != "einfach":
            raise ValueError(f"{json_path}: model type {d.get('art')!r}, this code only implements 'einfach' "
                             f"(model E, fit_modell_einfach.json)")
        nd, n = int(d["grad_d"]) + 1, int(d["grad_theta"]) + 1
        if nd < 0 or n < 1:
            raise ValueError(f"{json_path}: grad_d = {nd - 1}, grad_theta = {n - 1}, expected >= -1 and >= 0")
        p = [float(v) for v in d["param"]]
        n_expected = nd + 2 * n
        if len(p) != n_expected:
            raise ValueError(f"{json_path}: {len(p)} parameters, expected {n_expected} for the given degrees")
        self.param = p
        self.d_coef = p[:nd]               # d_i: start direction correction [rad], empty if grad_d = -1
        self.a_coef = p[nd:nd + n]         # a_i: coefficient of dx^2
        self.b_coef = p[nd + n:]           # b_i: coefficient of dx^3
        self.cov = [[float(v) for v in row] for row in d["kovarianz"]]
        if len(self.cov) != len(p) or any(len(row) != len(p) for row in self.cov):
            raise ValueError(f"{json_path}: covariance is not {len(p)}x{len(p)}")
        self.xp, self.zp, self.L, self.h = (float(v) for v in d["duese"])
        self.scale = float(d["skala_grad"])
        self.theta_min, self.theta_max = (float(v) for v in d["theta_bereich_grad"])
        self.sigma_eps = float(d.get("s_eps_rad", 0.0))     # angle scatter per shot [rad]
        self.sigma_alpha = float(d.get("s_alpha", 0.0))     # curvature scatter per shot [1/m]
        self.bal = None
        self.theta_lo = self.theta_min                      # lowest angle the inverse may return
        if d.get("ballistik"):
            self._load_ballistic(d["ballistik"], json_path)

    def _load_ballistic(self, b, json_path):
        kp = [float(v) for v in b["kp"]]
        if len(kp) != 1 or b.get("ks") is not None:
            raise ValueError(f"{json_path}: ballistic block with variable drag, only constant v0, k is implemented")
        noz = [float(v) for v in b["duese"]]
        if any(abs(a - c) > 1e-9 for a, c in zip(noz, (self.xp, self.zp, self.L, self.h))):
            raise ValueError(f"{json_path}: ballistic block uses a different nozzle than model E")
        cov = [[float(v) for v in row] for row in b["kovarianz"]]
        if len(cov) != 2 or any(len(row) != 2 for row in cov):
            raise ValueError(f"{json_path}: ballistic covariance is not 2x2")
        self.bal = {"v0": float(b["v0"]), "k": kp[0], "cov": cov, "g": float(b["g"]), "dt": float(b["dt"]),
                    "s_eps": float(b.get("s_eps_rad", 0.0)), "s_alpha": float(b.get("s_alpha", 0.0)),
                    "model_err": float(b.get("modellfehler", {}).get("m_pro_grad_pro_m", 0.0))}
        self.theta_lo = min(float(b["theta_min_grad"]), self.theta_min)
        # continuity correction along the jet at theta_min, tabulated up to the E0 landing point
        xd, _ = self.nozzle(self.theta_min)
        xl = self._xl_min = self.landing(self.theta_min)
        n = max(1, int(math.ceil((xl - xd) / CORR_STEP))) if math.isfinite(xl) else 1
        step = (xl - xd) / n if math.isfinite(xl) else CORR_STEP
        table = [self._e0_z(xd + i * step, self.theta_min) - self._ballistic(xd + i * step, self.theta_min)[0]
                 for i in range(n + 1)]
        self._corr = table if max(abs(c) for c in table) > JUMP_TOL else None
        self._corr_step = step

    # =======================================================================
    # Model-specific part (model E). Swap only this block if the model
    # family changes; everything below uses nozzle/z/slope/gradient/
    # shot_scatter only.
    # =======================================================================

    @staticmethod
    def _poly(coef, t):
        return sum(c * t ** i for i, c in enumerate(coef))

    def nozzle(self, theta):
        """Nozzle exit (x_d, z_d) [m] in experiment coordinates."""
        r = math.radians(theta)
        return (self.xp + self.L * math.cos(r) - self.h * math.sin(r),
                self.zp + self.L * math.sin(r) + self.h * math.cos(r))

    def _terms(self, X, theta):
        t = theta / self.scale
        xd, zd = self.nozzle(theta)
        k1 = math.tan(math.radians(theta)) + self._poly(self.d_coef, t)   # slope at the nozzle
        return t, zd, X - xd, k1, self._poly(self.a_coef, t), self._poly(self.b_coef, t)

    def _e0_z(self, X, theta):
        """Jet height [m] at horizontal position X (experiment coordinates)."""
        t, zd, dx, k1, a, b = self._terms(X, theta)
        return zd + k1 * dx + a * dx ** 2 + b * dx ** 3

    def _e0_slope(self, X, theta):
        t, zd, dx, k1, a, b = self._terms(X, theta)
        return k1 + 2 * a * dx + 3 * b * dx ** 2

    def _e0_gradient(self, X, theta):
        """dz/dparam (same order as param)."""
        t, zd, dx, k1, a, b = self._terms(X, theta)
        g = [dx * t ** i for i in range(len(self.d_coef))]
        g += [dx ** 2 * t ** i for i in range(len(self.a_coef))]
        g += [dx ** 3 * t ** i for i in range(len(self.b_coef))]
        return g

    def _e0_shot_scatter(self, X, theta):
        """Variance [m^2] of a new shot around the mean curve (angle scatter + curvature scatter)."""
        t, zd, dx, k1, a, b = self._terms(X, theta)
        return (self.sigma_eps * dx) ** 2 + (self.sigma_alpha * dx ** 2) ** 2

    # =======================================================================
    # Ballistic extension (theta < theta_min)
    # =======================================================================

    def _ballistic(self, X, theta, v0=None, k=None):
        """(z, slope) of the ballistic jet at X: RK4 from the nozzle exit until x >= X, linear in between."""
        b = self.bal
        v0 = b["v0"] if v0 is None else v0
        k = b["k"] if k is None else k
        g, dt = b["g"], b["dt"]
        r = math.radians(theta)
        x, z = self.nozzle(theta)
        vx, vz = v0 * math.cos(r), v0 * math.sin(r)
        if X < x:   # behind the exit: straight line in nozzle direction
            return z + vz / vx * (X - x), vz / vx

        def f(vx, vz):
            v = math.hypot(vx, vz)
            return -k * v * vx, -g - k * v * vz

        for _ in range(int(T_MAX / dt)):
            a1x, a1z = f(vx, vz)
            b2x, b2z = vx + 0.5 * dt * a1x, vz + 0.5 * dt * a1z
            a2x, a2z = f(b2x, b2z)
            b3x, b3z = vx + 0.5 * dt * a2x, vz + 0.5 * dt * a2z
            a3x, a3z = f(b3x, b3z)
            b4x, b4z = vx + dt * a3x, vz + dt * a3z
            a4x, a4z = f(b4x, b4z)
            xn = x + dt / 6.0 * (vx + 2 * b2x + 2 * b3x + b4x)
            zn = z + dt / 6.0 * (vz + 2 * b2z + 2 * b3z + b4z)
            vxn = vx + dt / 6.0 * (a1x + 2 * a2x + 2 * a3x + a4x)
            vzn = vz + dt / 6.0 * (a1z + 2 * a2z + 2 * a3z + a4z)
            if xn >= X:
                w = (X - x) / (xn - x)
                return z + w * (zn - z), vz / vx + w * (vzn / vxn - vz / vx)
            x, z, vx, vz = xn, zn, vxn, vzn
        return math.nan, math.nan

    def _correction(self, X, theta):
        """(c, dc/dX) of the continuity correction, faded to 0 at theta_min - FADE_DEG."""
        if self._corr is None:
            return 0.0, 0.0
        w = 1.0 - (self.theta_min - theta) / FADE_DEG
        if w <= 0:
            return 0.0, 0.0
        u = (X - self.nozzle(theta)[0]) / self._corr_step   # along the jet, measured from this angle's nozzle
        n = len(self._corr) - 1
        if u <= 0:
            return w * self._corr[0], 0.0
        if u >= n:
            return w * self._corr[n], 0.0
        i = int(u)
        i = min(i, n - 1)
        a, c = self._corr[i], self._corr[i + 1]
        return w * (a + (u - i) * (c - a)), w * (c - a) / self._corr_step

    def _bal_z(self, X, theta):
        return self._ballistic(X, theta)[0] + self._correction(X, theta)[0]

    def _bal_slope(self, X, theta):
        return self._ballistic(X, theta)[1] + self._correction(X, theta)[1]

    def _bal_sigma_curve(self, X, theta):
        b = self.bal
        hv, hk = 1e-3 * b["v0"], max(1e-3 * b["k"], 1e-6)
        gv = (self._ballistic(X, theta, v0=b["v0"] + hv)[0] - self._ballistic(X, theta, v0=b["v0"] - hv)[0]) / (2 * hv)
        gk = (self._ballistic(X, theta, k=b["k"] + hk)[0] - self._ballistic(X, theta, k=b["k"] - hk)[0]) / (2 * hk)
        c = b["cov"]
        return math.sqrt(max(gv * gv * c[0][0] + 2 * gv * gk * c[0][1] + gk * gk * c[1][1], 0.0))

    def _bal_shot_scatter(self, X, theta):
        b = self.bal
        dx = X - self.nozzle(theta)[0]
        model = b["model_err"] * (self.theta_min - theta) * dx
        return (b["s_eps"] * dx) ** 2 + (b["s_alpha"] * dx ** 2) ** 2 + model ** 2

    # =======================================================================
    # Dispatch: E0 inside the measured range, ballistic below it
    # =======================================================================

    def is_ballistic(self, theta):
        return self.bal is not None and theta < self.theta_min

    def z(self, X, theta):
        """Jet height [m] at horizontal position X (experiment coordinates)."""
        return self._bal_z(X, theta) if self.is_ballistic(theta) else self._e0_z(X, theta)

    def slope(self, X, theta):
        return self._bal_slope(X, theta) if self.is_ballistic(theta) else self._e0_slope(X, theta)

    def gradient(self, X, theta):
        """dz/dparam of model E (same order as param). Only defined in the measured range."""
        return self._e0_gradient(X, theta)

    def shot_scatter(self, X, theta):
        """Variance [m^2] of a new shot around the mean curve (ballistic branch: incl. model error)."""
        return self._bal_shot_scatter(X, theta) if self.is_ballistic(theta) else self._e0_shot_scatter(X, theta)

    # =======================================================================
    # Generic part
    # =======================================================================

    def landing(self, theta):
        """Landing point x [m] (first crossing of z = 0 behind the nozzle), nan if none."""
        xd, zd = self.nozzle(theta)
        if zd <= 0:
            return math.nan
        lo = 0.0
        while lo < LANDING_MAX:
            hi = lo + LANDING_STEP
            if self.z(xd + hi, theta) < 0:
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if self.z(xd + mid, theta) > 0 else (lo, mid)
                return xd + 0.5 * (lo + hi)
            lo = hi
        return math.nan

    def sigma_curve(self, X, theta):
        """1-sigma uncertainty of the mean curve (fixed parameters) [m]."""
        if self.is_ballistic(theta):
            return self._bal_sigma_curve(X, theta)
        g = self.gradient(X, theta)
        var = sum(g[i] * self.cov[i][j] * g[j] for i in range(len(g)) for j in range(len(g)))
        return math.sqrt(max(var, 0.0))

    def sigma_shot(self, X, theta):
        """1-sigma deviation of a new shot (mean curve + shot-to-shot scatter) [m]."""
        return math.sqrt(self.sigma_curve(X, theta) ** 2 + self.shot_scatter(X, theta))

    def angle_for_target(self, x, z, frame="pivot", shot=True):
        """Target (x, z) [m] -> Result with absolute nozzle angle theta [deg] and sigma_theta [deg].

        shot=True: sigma for a new shot (incl. shot-to-shot scatter), False: mean curve only."""
        if frame == "pivot":
            X, Z = self.xp + x, self.zp + z
        elif frame == "experiment":
            X, Z = float(x), float(z)
        else:
            raise ValueError(f"unknown frame {frame!r}, use 'pivot' or 'experiment'")
        res = Result(ok=False, x_exp=X, z_exp=Z)

        def fail(reason, message):
            res.reason, res.message = reason, message
            return res

        if not (math.isfinite(X) and math.isfinite(Z)):
            return fail("invalid_input", f"target not finite (x={x}, z={z})")
        lo, hi = self.theta_min, self.theta_max
        if X <= min(self.nozzle(self.theta_lo)[0], self.nozzle(hi)[0], self.nozzle(0.5 * (lo + hi))[0]):
            return fail("too_close", f"target at x={X:.2f} m is not in front of the nozzle")

        if self.bal is not None and Z < self._boundary_z(X):
            return self._angle_ballistic(X, Z, res, fail, shot)

        f_lo, f_hi = self.z(X, lo) - Z, self.z(X, hi) - Z
        if f_hi < 0:
            xl = self.landing(hi)
            if not math.isfinite(xl) or X > xl + LANDING_TOLERANCE:
                return fail("too_far", f"target at x={X:.2f} m is beyond the longest measured jet "
                                       f"(lands at {xl:.2f} m at theta={hi:g} deg)")
            return fail("too_high", f"target is {-f_hi:.2f} m above the steepest measured jet (theta={hi:g} deg)")
        if f_lo > 0 and self.bal is not None:   # above the theta_min jet, but behind its E0 landing point
            return fail("beyond_landing", f"target at x={X:.2f} m, z={Z:.2f} m is behind the landing point of the "
                                          f"measured jets and above the ballistic extrapolation")
        if f_lo > 0:
            return fail("too_low", f"target is {f_lo:.2f} m below the flattest measured jet (theta={lo:g} deg)")

        # z(X, theta) increases with theta in the valid range -> bisection
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.z(X, mid) - Z < 0 else (lo, mid)
        theta = 0.5 * (lo + hi)

        if X <= self.nozzle(theta)[0]:
            return fail("too_close", f"target at x={X:.2f} m is not in front of the nozzle at theta={theta:.2f} deg")
        xl = self.landing(theta)
        if not math.isfinite(xl) or X > xl + LANDING_TOLERANCE:
            return fail("beyond_landing", f"at theta={theta:.2f} deg the jet lands at {xl:.2f} m, "
                                          f"before the target at x={X:.2f} m")

        dz_dtheta = (self.z(X, theta + DTHETA) - self.z(X, theta - DTHETA)) / (2 * DTHETA)
        sz = self.sigma_shot(X, theta) if shot else self.sigma_curve(X, theta)
        res.ok, res.theta, res.sigma_z, res.dz_dtheta = True, theta, sz, dz_dtheta
        res.sigma_theta = sz / abs(dz_dtheta)
        res.reason, res.branch = "ok", "E0"
        res.message = f"theta = {theta:.2f} +- {res.sigma_theta:.2f} deg"
        return res

    def _boundary_z(self, X):
        """Height of the theta_min jet at X: E0 up to its landing point, beyond that the corrected ballistic
        jet (the E0 polynomial is not valid behind its landing point). Targets below go to the ballistic branch."""
        xl = self._xl_min
        if math.isfinite(xl) and X <= xl:
            return self._e0_z(X, self.theta_min)
        return self._ballistic(X, self.theta_min)[0] + (self._corr[-1] if self._corr is not None else 0.0)

    def _angle_ballistic(self, X, Z, res, fail, shot):
        """Bisection in [theta_lo, theta_min) with the ballistic branch. No ground (landing) check."""
        lo, hi = self.theta_lo, self.theta_min
        f_lo = self.z(X, lo) - Z
        if not math.isfinite(f_lo) or f_lo > 0:
            return fail("too_low", f"target is {f_lo:.2f} m below the flattest jet (theta={lo:g} deg, "
                                   f"ballistic extrapolation)")
        while hi - lo > 1e-6:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.z(X, mid) - Z < 0 else (lo, mid)
        theta = 0.5 * (lo + hi)
        if X <= self.nozzle(theta)[0]:
            return fail("too_close", f"target at x={X:.2f} m is not in front of the nozzle at theta={theta:.2f} deg")
        if abs(self.z(X, theta) - Z) > Z_TOL:
            return fail("too_low", f"no ballistic solution for x={X:.2f} m, z={Z:.2f} m")
        t1, t2 = max(theta - DTHETA, self.theta_lo), min(theta + DTHETA, self.theta_min - 1e-9)
        dz_dtheta = (self.z(X, t2) - self.z(X, t1)) / (t2 - t1)
        sz = self.sigma_shot(X, theta) if shot else self.sigma_curve(X, theta)
        res.ok, res.theta, res.sigma_z, res.dz_dtheta = True, theta, sz, dz_dtheta
        res.sigma_theta = sz / abs(dz_dtheta)
        res.reason, res.branch = "ok", "ballistic"
        res.message = (f"theta = {theta:.2f} +- {res.sigma_theta:.2f} deg (ballistic extrapolation below the "
                       f"measured {self.theta_min:g} deg)")
        return res


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python zielmodell.py MODEL.json X Z   (target relative to the servo pivot, metres)")
        sys.exit(2)
    m = Zielmodell(sys.argv[1])
    r = m.angle_for_target(float(sys.argv[2]), float(sys.argv[3]))
    print(f"target pivot x={sys.argv[2]} z={sys.argv[3]} -> experiment x={r.x_exp:.2f} z={r.z_exp:.2f}")
    print(f"{r.reason}: {r.message}")
    if r.ok:
        print(f"sigma_z new shot {r.sigma_z * 100:.0f} cm, dz/dtheta = {r.dz_dtheta * 100:.1f} cm/deg")
