"""Standalone aiming model: target position -> absolute nozzle angle.

Reads the fitted jet model E (fit_modell_einfach.json, written by einfach.py in Kamera_entzerrung)
and inverts it. Standard library only, so it can be copied into the ROS package as is.
After a refit only the JSON has to be copied again.

Model E (all lengths in m, theta = absolute nozzle angle, t = theta / skala_grad):
    nozzle exit  x_d = x_p + L cos(theta) - h sin(theta),  z_d = z_p + L sin(theta) + h cos(theta)
    jet          dz = (tan(theta) + d(t)) dx + a(t) dx^2 + b(t) dx^3,  dx = x - x_d, dz = z - z_d
                 d, a, b polynomials in t (d optional, grad_d = -1: none)
    shot scatter variance  s_eps^2 dx^2 + s_alpha^2 dx^4

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
Valid only for theta inside the measured range and between nozzle and landing point.

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

    def z(self, X, theta):
        """Jet height [m] at horizontal position X (experiment coordinates)."""
        t, zd, dx, k1, a, b = self._terms(X, theta)
        return zd + k1 * dx + a * dx ** 2 + b * dx ** 3

    def slope(self, X, theta):
        t, zd, dx, k1, a, b = self._terms(X, theta)
        return k1 + 2 * a * dx + 3 * b * dx ** 2

    def gradient(self, X, theta):
        """dz/dparam (same order as param)."""
        t, zd, dx, k1, a, b = self._terms(X, theta)
        g = [dx * t ** i for i in range(len(self.d_coef))]
        g += [dx ** 2 * t ** i for i in range(len(self.a_coef))]
        g += [dx ** 3 * t ** i for i in range(len(self.b_coef))]
        return g

    def shot_scatter(self, X, theta):
        """Variance [m^2] of a new shot around the mean curve (angle scatter + curvature scatter)."""
        t, zd, dx, k1, a, b = self._terms(X, theta)
        return (self.sigma_eps * dx) ** 2 + (self.sigma_alpha * dx ** 2) ** 2

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
        if X <= min(self.nozzle(lo)[0], self.nozzle(hi)[0], self.nozzle(0.5 * (lo + hi))[0]):
            return fail("too_close", f"target at x={X:.2f} m is not in front of the nozzle")

        f_lo, f_hi = self.z(X, lo) - Z, self.z(X, hi) - Z
        if f_hi < 0:
            xl = self.landing(hi)
            if not math.isfinite(xl) or X > xl + LANDING_TOLERANCE:
                return fail("too_far", f"target at x={X:.2f} m is beyond the longest measured jet "
                                       f"(lands at {xl:.2f} m at theta={hi:g} deg)")
            return fail("too_high", f"target is {-f_hi:.2f} m above the steepest measured jet (theta={hi:g} deg)")
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
        res.reason = "ok"
        res.message = f"theta = {theta:.2f} +- {res.sigma_theta:.2f} deg"
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
