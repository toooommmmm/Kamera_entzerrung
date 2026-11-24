# Handover: extended aiming range below −3° (ballistic branch in `zielmodell.py`)

Status 24.11.2026. Implemented and tested in `~/Documents/Kamera_entzerrung` and in
`CORD/cord_ws/src/cord_nozzle_aim` on the laptop. This file contains everything needed to apply the same change in
another checkout of `cord_nozzle_aim`: the complete new `zielmodell.py`, the new JSON block with the fitted values,
the small changes to `aim_test.py` and `README.md`, and the checks with their expected results.
Background and the full derivation: `Kamera_entzerrung/modell_E_einfach.md`, section 10.

## 1. What changes and why

- Model E0 (`fit_modell_einfach.json`) is only valid for the measured nozzle angles θ = −3° … 25°. Targets that
  need a flatter jet were rejected as `too_low`. In flight that happens often: the drone is higher than the test
  nozzle (1.31 m), so the target lies below the nozzle.
- New: for θ = −20° … −3° the jet comes from a **ballistic model**, a point mass with quadratic air drag:

  ```
  x(0) = x_d(θ), z(0) = z_d(θ)     (nozzle exit, same geometry and pivot as E0)
  v(0) = v0·(cos θ, sin θ)
  ẍ = −k·|v|·vx,   z̈ = −g − k·|v|·vz
  ```

  **v0 = 19.43 m/s and k = 0.135 1/m are constants**, fitted on the flat test shots only (clips 0, 5, 10, 15,
  θ = −3° … 11°).
- Validation (hold-out): fitted on clips 5/10/15 (2°–11°), it predicted the unseen clip 0 shots (−2°/−3°) with
  a landing error of 22 cm RMS and an angle error of 0.21° RMS (E0 polynomial extrapolated: 271 cm, 4.7°). It is
  not validated further than 4.7° below the data, so the model carries a growing model-error term (section 3).
- **E0 is unchanged.** Every target that was reachable before gets the identical θ. A JSON without the new block
  behaves exactly as before.

Reach (horizontal distance in front of the pivot, 1 m grid up to 10 m):

| below the pivot | 0.5 m | 1.0 m | 1.5 m | 2.0 m | 2.5 m | 3.0 m | 4.0 m |
|---|---|---|---|---|---|---|---|
| new | 2–10 m | 3–10 m | 4–7 m | 5–8 m | 6–8 m | 7–9 m | 8–10 m |
| before | 5–10 m | 7–10 m | – | – | – | – | – |

## 2. Files to change in `cord_nozzle_aim`

| File | Change |
|---|---|
| `cord_nozzle_aim/zielmodell.py` | replace completely with the code in section 6 (standard library only, as before) |
| `config/fit_modell_einfach.json` | add the `"ballistik"` block from section 5 (top-level key, next to `art`, `param`, ...). Better: copy the whole file from `Kamera_entzerrung/auswertung/fit_modell_einfach.json` |
| `cord_nozzle_aim/aim_test.py` | one warning extended (section 7) |
| `README.md` | messages table and known limitations (section 7) |
| `dist_angle_mapping.py` | **no change** (`compute_phi` only calls `angle_for_target`). φ at θ = −20° with a level drone: (−20 + 5) / 0.89 = −16.9°, inside `phi_min_deg = −45` |

Then `colcon build --packages-select cord_nozzle_aim`.

## 3. How the new `zielmodell.py` works

Interface unchanged: `Zielmodell(json_path)`, `angle_for_target(x, z, frame="pivot", shot=True) -> Result`.
`Result` has one new field, `branch` (`"E0"` or `"ballistic"`, empty on failure). Reason strings are unchanged.

1. **Loading.** If the JSON has a `"ballistik"` block, it is loaded (`_load_ballistic`). The code checks that the
   block uses the same nozzle (`duese`) as E0, that the drag is constant (`kp` has one value, no `ks`), and that
   the covariance is 2×2. `theta_lo = theta_min_grad` (−20) becomes the lowest angle the inverse may return.
   Without the block, `bal = None` and `theta_lo = theta_min`, so the behaviour is the old one.
2. **Ballistic jet** (`_ballistic(X, theta)`): RK4 with fixed dt = 1 ms (from the JSON), integrated from the
   nozzle exit only until x ≥ X, then linear interpolation between the last two samples (z and slope vz/vx). This
   is the same scheme as `ballistik.py`, and the test agrees with it to 3.5 µm.
3. **Continuity at θ_min = −3°.** Along the −3° jet, E0 and the ballistic jet agree within 1.5 cm up to 5 m from
   the nozzle, but differ by 11 cm at the E0 landing point. The ballistic branch therefore adds
   c(Δx) = z_E0(X, −3°) − z_ballistic(X, −3°), tabulated every 2 cm along the jet up to the E0 landing point,
   constant behind it, and faded linearly to 0 between −3° and −8° (`FADE_DEG = 5`). It is only used if the jump
   is above 3 cm (`JUMP_TOL`). Result: no jump at −3°, and z(X, θ) still increases with θ everywhere from −20° to
   25° (min 1.5 cm/°), which the bisection needs.
4. **Choosing the branch** (`angle_for_target`): first the target height Z is compared with the −3° jet at X
   (`_boundary_z`: E0 up to its landing point, behind it the corrected ballistic jet, because the E0 polynomial is
   not valid behind its landing point).
   - Z on or above it: the **old E0 path**, unchanged (bisection in [−3°, 25°], landing check). One small
     difference: a target above the −3° jet but behind the E0 landing point now gives `beyond_landing` instead of
     `too_low`.
   - Z below it: `_angle_ballistic`, bisection in [−20°, −3°) to 1e-6°. **No ground / landing check** in this
     branch, because in flight the target may lie below the ground of the test setup. Only "in front of the
     nozzle" and the θ bounds remain. A solution must hit Z within 1 cm (`Z_TOL`). `too_low` now means below the
     −20° jet. The message says "ballistic extrapolation below the measured -3 deg".
5. **Uncertainty in the ballistic branch** (σ_theta = σ_z / |dz/dθ|):
   ```
   σ_z² = σ_curve²  +  (s_ε·Δx)² + (s_α·Δx²)²  +  σ_model²
   σ_curve: numerical gradient of z w.r.t. (v0, k) with the 2×2 covariance from the JSON
   σ_model = m_pro_grad_pro_m · (θ_min − θ) · Δx      (0.27 cm per degree below −3° per metre from the nozzle)
   ```
   Example: θ = −15°, 4 m from the nozzle: σ_model = 13 cm. The coefficient is calibrated from the hold-out test
   (RMS of residual / (angle gap · Δx) over the validation points more than 1 m from the nozzle).
6. **Speed:** 3–20 ms per ballistic target in pure Python (E0 targets: under 1 ms as before).

## 4. Known limits

- Validated only 4.7° below the fit data. −20° is 17° below. Shots at −5° … −20° on a test day would validate it.
  Log them: `aim_test` now prints a NOTE when the nozzle angle is in the ballistic range.
- Closer, steep targets need flatter than −20° (e.g. 4 m ahead and 2.5 m below the pivot: about −30°) and stay
  `too_low`.
- Far targets below the test ground at small angles (e.g. 8 m ahead, 1.5 m below the pivot) stay
  `beyond_landing`: they need θ ≈ −3° … 0°, where E0 only knows the jet down to the test ground. Possible next
  step: continue those jets with the ballistic model behind the E0 landing point.
- The drag model is a deliberately simple physical model. A drag that varies with θ or grows with flight distance
  was tested and gave no improvement (`modell_E_einfach.md`, section 10).

## 5. JSON block (add to `config/fit_modell_einfach.json`)

Written by `python strahl.py fit` in Kamera_entzerrung (`ballistik.fit_fuer_ziel`). `duese` must equal the E0
`duese` of the same file, otherwise `zielmodell.py` refuses to load. After every refit the block is rewritten, so
copying the whole file is the safer way.

```json
{
  "ballistik": {
    "v0": 19.43009865845326,
    "kp": [
      0.13499472491142855
    ],
    "kovarianz": [
      [
        0.24222501196905286,
        0.004165351782821792
      ],
      [
        0.004165351782821791,
        7.963644287436439e-05
      ]
    ],
    "g": 9.81,
    "dt": 0.001,
    "skala_grad": 30.0,
    "theta_min_grad": -20.0,
    "s_eps_rad": 0.0,
    "s_alpha": 0.0027548975788159787,
    "duese": [
      -0.9616855639995406,
      1.3110946557880092,
      0.81,
      0.0
    ],
    "formel": "Start x_d, z_d wie Modell E; v(0) = v0 (cos θ, sin θ); k(θ) = max(0, Σ kp_i t^i), t = θ/skala_grad; ẍ = −k|v|vx, z̈ = −g − k|v|vz; RK4 dt, z(x) linear interpoliert",
    "modellfehler": {
      "m_pro_grad_pro_m": 0.0026904623112721264,
      "test": "Fit 5/10/15, Vorhersage 0",
      "aufprall_rms_m": 0.2214330071176744
    },
    "clips": [
      "0",
      "0_a",
      "0_b",
      "10_a",
      "10_b",
      "15_a",
      "15_b",
      "5",
      "5_a",
      "5_b"
    ],
    "theta_fit_bereich_grad": [
      -3.0,
      11.0
    ]
  }
}
```

Fields used by `zielmodell.py`: `v0` [m/s], `kp[0]` = k [1/m], `kovarianz` (2×2, order v0, k), `g`, `dt` [s],
`theta_min_grad`, `s_eps_rad`, `s_alpha` [1/m], `duese`, `modellfehler.m_pro_grad_pro_m` [m/(deg·m)].
Optional `ks` (drag growing with distance) is refused. The other fields are documentation.

## 6. Complete new `cord_nozzle_aim/zielmodell.py`

Identical in `Kamera_entzerrung/zielmodell.py` and `cord_nozzle_aim/cord_nozzle_aim/zielmodell.py`.

```python
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
```

## 7. Small changes

**`aim_test.py`** (in the result printout, around the "WARNING: nozzle angle outside the measured range" lines).
Replace

```python
        if not model.theta_min <= r["theta_real"] <= model.theta_max:
            lines.append(f"  WARNING: nozzle angle outside the measured range of the jet model "
                         f"[{model.theta_min:g}, {model.theta_max:g}] deg, prediction is extrapolated")
```

with

```python
        if model.bal is not None and model.theta_lo <= r["theta_real"] < model.theta_min:
            lines.append(f"  NOTE: nozzle angle below the measured range ({model.theta_min:g} deg), jet from the "
                         f"ballistic extrapolation (valid down to {model.theta_lo:g} deg, wider error band)")
        elif not model.theta_min <= r["theta_real"] <= model.theta_max:
            lines.append(f"  WARNING: nozzle angle outside the measured range of the jet model "
                         f"[{model.theta_min:g}, {model.theta_max:g}] deg, prediction is extrapolated")
```

**`README.md`**
- Section "3. Jet model", after the bullet about `WARNING: nozzle angle outside the measured range`, add:
  "`NOTE: nozzle angle below the measured range ... ballistic extrapolation`: between −20° and −3° the jet comes
  from the ballistic extension (see Known limitations). Its error band is wider and grows with the distance below
  −3°. Shots in this range are exactly the data that would validate it, so log them."
- Messages table:
  - `too_low`: "Target is below the flattest jet the model allows (−20°, ballistic extrapolation; −3° with a JSON
    without the `ballistik` block). Hover lower or move further away. Roughly reachable (horizontal distance from
    the pivot): 0.5 m below at 2–10 m, 1.5 m below at 4–7 m, 2.5 m below at 6–8 m, 4 m below at 8–10 m."
  - `too_far` / `beyond_landing`: add "Also for far, low targets (e.g. 8 m ahead, 1.5 m below the pivot): they need
    about −3° to 0°, where the measured model only knows the jet down to the test-site ground."
- Known limitations, "Jet model range": E0 valid for −3° … 25°; below −3° (to −20°) the ballistic extension if the
  JSON has a `ballistik` block (constant v0 = 19.4 m/s, k = 0.135 1/m, fitted on the flat shots −3° … 11°;
  hold-out 22 cm at landing, not tested further than 4.7° below the data). The result says
  `ballistic extrapolation` and `Result.branch = "ballistic"`. No ground check in this branch.

## 8. Checks and expected results

1. **Package tests:** `cd cord_ws/src/cord_nozzle_aim && python3 -m pytest test/test_hold_target.py` → 4 passed.
2. **Offline, with the new JSON** (from the package directory):
   ```bash
   python3 -c "
   from cord_nozzle_aim.zielmodell import Zielmodell
   m = Zielmodell('config/fit_modell_einfach.json')
   for x, z in ((5, -1.5), (3, -0.5), (8, -4), (4, -2.5), (5, 0)):
       r = m.angle_for_target(x, z); print(x, z, r.reason, r.branch, r.message)"
   ```
   Expected (pivot frame):
   | target | reason | branch | θ |
   |---|---|---|---|
   | 5 m, −1.5 m | ok | ballistic | −12.70° (φ = −8.65° with level drone) |
   | 3 m, −0.5 m | ok | ballistic | −7.99° ± 0.62° |
   | 8 m, −4 m | ok | ballistic | −16.11° ± 1.62° |
   | 4 m, −2.5 m | too_low | – | (0.83 m below the −20° jet) |
   | 5 m, 0 m | ok | E0 | 4.10° (same as before) |
3. **Old behaviour:** remove the `ballistik` block from a copy of the JSON. All targets that E0 could reach must give
   the same θ as with the block, and (5, −1.5) must give `too_low`.
4. **Full check** (needs Kamera_entzerrung with `einfach.py`, `ballistik.py`): `python3 test_zielmodell.py` →
   `0 mismatches`, including section 5: forward model vs. `ballistik.BallistikModell` (max 3.5 µm), jump at −3°
   0.00 cm with correction (11.4 cm without), min dz/dθ 1.50 cm/° over −20 … 25°, round trip 20 targets
   (6 below the test ground), JSON without block identical on 25 E0 targets.
5. **ROS node:** `ros2 launch cord_nozzle_aim nozzle_aim.launch.py` with fake inputs (IMU on
   `/leader/mavros/mavros/imu/data` with MAVROS pitch +12°, i.e. quaternion y = sin 6°, w = cos 6° = level with
   `imu_pitch_offset_deg: -12.0`; target on `/red_dot_position` in camera coordinates
   `(-0.0925, 1.5 + 0.0903, 5 - 0.3427)` = 5 m ahead, 1.5 m below the pivot). Expected: `/leader/nozzle_degrees`
   = −8.65. The same with 4 m / 2.5 m below gives the warning `No aim solution (too_low)`.

## 9. Fit side (Kamera_entzerrung), for reference

Only needed if the model is refitted on the other machine. Files: `ballistik.py` (new), `strahl.py`, `einfach.py`,
`test_zielmodell.py`, `modell_E_einfach.md`, all modified on the laptop and not yet committed. Commit and pull them
rather than re-implementing.

- `ballistik.py`: `BallistikModell` (same interface as `EinfachModell`; v0 constant, drag `kp` polynomial in θ,
  optional `ks` for drag growing with distance), `fit_ballistik` (scipy `least_squares`, same data and weights as
  E0, two passes, covariance scaled by χ²_red), `streuung` (damped version of `einfach._streuung`; the undamped one
  alternates between two solutions on the ballistic residuals), `validierung` / `plot_validierung` (hold-out
  test), `fit_fuer_ziel` (final fit on clips 0/5/10/15 with the E0 pivot + model-error coefficient from the
  hold-out fit 5/10/15 → 0, returns the JSON block).
- `strahl.py fit` writes the block into `fit_modell_einfach.json` (`--ohne-ballistik` to skip).
  `strahl.py ballistik-test [--train 5,10,15 --val 0 --endsatz 0,5,10,15] [--grad-k N] [--mit-weg]` runs the
  hold-out tests (`auswertung/ballistik_test*.txt/.png`).
- `einfach.speichere(m, path, zusatz)` writes extra JSON blocks.
- Run with the system `python3`: the project `.venv` has no scipy.
