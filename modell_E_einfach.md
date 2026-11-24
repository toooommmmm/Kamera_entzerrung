# Model E0 ("einfach"): water-jet trajectory model

Status 24.11.2026. This is the chosen trajectory model. `python strahl.py fit` uses it by default.
Code: `einfach.py` (model and fit), `strahl.py` (command line, nozzle dimensions).
The previous model write-up (D2 chosen, A3 as benchmark) is `CORD/BA_Tom_Gian/strahlmodell_D2_A3.md`.

## 1. What the model does

Given the nozzle angle θ, the model predicts the height z of the jet at every horizontal position x, from the
nozzle exit to the landing point. It also gives an uncertainty for this prediction (σ_shot), i.e. how far a
single new shot is expected to deviate from the mean curve.

Coordinates are those of the test setup: x horizontal along the jet (relative to the reference posts),
z height above ground, θ the absolute nozzle angle (0° = horizontal, positive = upwards).

## 2. Why a new model

D2 works, but it is more complicated than needed:

- an angle offset δ(θ) inside tan(θ + δ), which makes the fit nonlinear and iterative,
- sec²θ / sec³θ factors from the drag-free ballistic formula,
- per-shot random effects estimated by REML,
- a nozzle circle whose 4 constants (x_p, z_p, L, h) were fitted from the clicked nozzle tips.

A3 (a free cubic polynomial in x with coefficients quadratic in θ) is simple, but it is not tied to the
nozzle. Its start direction is off from the nozzle direction by up to 3.8°.

Model E is A3 tied to the nozzle: the polynomial is written around the nozzle exit, so it starts there and
in the direction the nozzle points. It predicts as well as D2 and A3 (section 6) with 6 parameters and a
single linear fit.

## 3. The nozzle

The nozzle turns about a pivot (the servo axis). Its exit lies at a fixed point in a frame that rotates
with the nozzle:

```
                       ↑ across the axis (h)
                       |
   pivot ⊕─────────────●  exit  ──→ jet leaves in direction θ
         ←──── L ─────→
```

```
x_d(θ) = x_p + L·cos θ − h·sin θ
z_d(θ) = z_p + L·sin θ + h·cos θ
```

- **L**: distance from the pivot axis to the exit orifice, measured along the nozzle centreline.
  Measured on the drone: **L = 0.81 m**.
- **h**: offset of the exit across the axis (h > 0: exit above the pivot at θ = 0). On the drone **h = 0**.
- **x_p, z_p**: position of the pivot in the test setup. These are not known from the setup, so they are
  fitted: for each clip with a clicked nozzle tip, subtract the L/h terms from the clicked tip, then take the
  weighted mean over all clips (weights 1/σ², `einfach.duese_mit_massen`).

L and h are set in one place, at the top of `strahl.py`:

```python
DUESE_L = 0.81   # [m] pivot -> nozzle exit along the axis. None: fit L from the clicks
DUESE_H = 0.0    # [m] exit offset across the axis
```

`--duese-L` / `--duese-h` on the command line override them for a single run. With `DUESE_L = None`, x_p, z_p
and L are fitted together from the clicked tips with h fixed (`einfach.duese_fit_L`). This is linear,
because x + h·sin θ = x_p + L·cos θ and z − h·cos θ = z_p + L·sin θ.

## 4. The mean jet curve

Measured from the nozzle exit, with Δx = x − x_d(θ), Δz = z − z_d(θ) and t = θ / 30°:

```
Δz = tan θ · Δx  +  c2(t) · Δx²  +  c3(t) · Δx³

c2(t) = a0 + a1·t + a2·t²
c3(t) = b0 + b1·t + b2·t²
```

- There is **no constant term**, so the curve passes through the nozzle exit: Δz = 0 at Δx = 0.
- The **slope at the exit** is exactly tan θ, so the jet starts in the direction the nozzle points.
  The tan θ term is fixed. It is not a fitted parameter.
- c2 describes the drop of the jet (gravity), c3 how the drop changes along the jet (drag, jet break-up).
  Both vary smoothly with the angle through t. The scaling by 30° only keeps the numbers of similar size.
- 6 parameters (a0, a1, a2, b0, b1, b2). They appear **linearly**, so the fit is one weighted least-squares
  solve, with no iteration over the model shape.

### Fit

The data are all clicked jet points of all clips plus the clicked landing points (z = 0).
For each point, the known part tan θ · Δx is moved to the left side:

```
y = Δz − tan θ · Δx  =  [Δx², t·Δx², t²·Δx², Δx³, t·Δx³, t²·Δx³] · (a0, a1, a2, b0, b1, b2)
```

Each point is weighted by 1/σ², where σ combines the click error in z and the click error in x converted
through the local slope of the jet: σ² = σ_z² + (slope · σ_x)². For landing points, σ_x also contains the
spray spread (0.10 m at 0°, 0.25 m at 30°). The slope is only known after the fit, so the fit runs twice:
first with the nozzle slope tan θ, then with the slope of the fitted curve.

The parameter covariance is multiplied by χ²_red (so the parameter σ grows by √χ²_red) when χ²_red > 1.
It is never scaled down. This gives σ_curve, the uncertainty of the mean curve itself.

### Variants

E1 adds a constant direction correction d (slope at the exit = tan θ + d, 7 parameters). E2 makes d
quadratic in t (9 parameters). Neither improves the prediction (section 6), so **E0** (d ≡ 0) is the model.
`fit --grad-d 0` gives E1 and `--grad-d 2` gives E2.

## 5. Shot-to-shot scatter (iteratively weighted NNLS)

Two shots at the same angle do not land at the same point. The mean curve (section 4) does not describe this
scatter, so it is estimated afterwards from the residuals r = y − (fitted curve), one per data point.

### Scatter model

Three sources of error, each growing differently with the distance Δx from the nozzle:

```
E[r²] = v(Δx) = C  +  s_ε² · Δx²  +  s_α² · Δx⁴
```

| Term | Meaning | Growth |
|---|---|---|
| C | click noise (the jet is clicked with a few cm accuracy) | constant |
| s_ε² · Δx² | the shot leaves at a slightly different angle ε (rad) | ∝ Δx² (height error ε·Δx) |
| s_α² · Δx⁴ | the shot bends slightly differently, curvature error α (1/m) | ∝ Δx⁴ (height error α·Δx²) |

The task is to find C, s_ε², s_α² from the r² of all points. This is a linear regression of r² on the columns
1, Δx², Δx⁴, with two complications.

### Why non-negative least squares (NNLS)

C, s_ε² and s_α² are variances and cannot be negative. An ordinary least-squares fit does not know this.
With noisy data it can return e.g. s_α² = −0.0001, which would make σ imaginary at large distances.

NNLS (`scipy.optimize.nnls`) solves the same least-squares problem, minimise ‖A·k − b‖², with the constraint
that every coefficient k ≥ 0. It uses the Lawson–Hanson active-set method:

1. Start with all coefficients at 0 (all "clamped").
2. Release the clamped coefficient that would reduce the error most if it were allowed to grow.
3. Solve ordinary least squares for the released coefficients only.
4. If a released coefficient went negative, move back towards the previous solution until it reaches 0,
   clamp it again, and repeat 3.
5. Stop when no clamped coefficient would reduce the error by growing.

The result is the best fit with all coefficients ≥ 0. A coefficient that the data push below zero ends up at
exactly 0, which means "this error source is not visible in the data".

### Why weight, and why iterate

The r² values do not all have the same uncertainty. If r is normally distributed with variance v, then r²
has mean v and **variance 2v²**. A point 10 m from the nozzle has r ≈ 25 cm, so r² ≈ 600 cm². A point near
the nozzle has r ≈ 2 cm, so r² ≈ 4 cm². In an unweighted fit, one far point counts about 20 000 times more
(in squared error) than one near point, and the 12 far points decide everything.

That was the case in the first version: it gave C = 0 although the clicks clearly show about 2 cm of noise
near the nozzle. The σ band was 3× too narrow at the nozzle and 2× too wide at 2–6 m.

The correct weight for each row is 1/v, i.e. divide each row of A and each r² by its predicted variance v.
But v is what is being estimated, so the fit is iterated (`einfach._streuung`):

```python
A = [1, Δx², Δx⁴]                          # one row per data point
v = mean(r²) for every point               # start: equal weights
repeat (at most 50 times):
    k = nnls(A / v, r² / v)                # weighted NNLS, k = (C, s_ε², s_α²)
    v_new = max(A · k, 1 mm²)              # predicted variance per point, floor against division by 0
    stop if v_new ≈ v (relative change < 1e-4)
    v = v_new
return √C, s_ε, s_α
```

This converges within a few iterations. Afterwards the predicted spread agrees with the measured residual
RMS at every distance (table in section 6).

### Using it for a new shot

```
σ_shot²(Δx) = σ_curve²(Δx) + s_ε² · Δx² + s_α² · Δx⁴
```

C is left out, because it is the clicking error of the test data, not a property of a real shot.
At the landing point, σ_shot is converted into a horizontal uncertainty through the slope of the curve.

The landing points are part of the scatter fit. Their residuals also contain the uncertainty of where the
spray lands, which mostly ends up in the s_α term. This is intended: for aiming, the landing spread is what
counts.

## 6. Results (h = 0, L = 0.81 m)

### Model comparison

`python strahl.py vergleich`, leave-one-run-out: fit on two of the three measurement runs, predict the
third, repeat for all three. 18 clips, θ = −3° … 25°.

| Model | Parameters | Landing RMS | Angle RMS |
|---|---|---|---|
| **E0** | **6** | **31 cm** | **0.80°** |
| E1 | 7 | 31 cm | 0.81° |
| E2 | 9 | 30 cm | 0.78° |
| A3 | 12 | 30 cm | 0.79° |
| D2 | 9 | 30 cm | 0.80° |

- Landing RMS: model landing point at the measured θ minus the clicked landing point.
- Angle RMS: measured θ minus the θ at which the model best matches the shot's clicked points, i.e. the
  aiming error in degrees.
- All models lie within 1 cm and 0.03° of each other. The shot-to-shot scatter (about 30 cm at landing) is
  far larger than these differences. E0 is chosen because it gives the same accuracy with the fewest
  parameters and the simplest fit.
- No direction correction is needed: E1's d comes out at −0.5° and changes nothing, E2 bends the start
  direction by up to −2° without improving. D2's δ(θ) was fitting noise.

### Fit on all data (`python strahl.py fit`)

- Pivot: x_p = −0.962 m, z_p = 1.311 m. Nozzle residuals (clicked tip − model): RMS 2.4 cm in x, 2.0 cm in z.
- χ²_red = 2.56, i.e. the click σ alone underestimates the residuals by a factor of about 1.6. The rest is
  shot-to-shot scatter.
- Scatter: √C = 2.3 cm, s_ε = 0.31°, s_α = 0.0024 1/m (±15 cm at 8 m from the nozzle).

| Δx from nozzle | Points | Measured residual RMS | Scatter model |
|---|---|---|---|
| 0–0.5 m | 30 | 2.4 cm | 2.3 cm |
| 0.5–2 m | 26 | 2.1 cm | 2.4 cm |
| 2–6 m | 16 | ~5 cm | ~6 cm |
| 6–8 m | 6 | 11 cm | 13 cm |
| 8–12 m | 12 | 24 cm | 25 cm |

- σ_shot at landing: 26–34 cm, against the measured landing scatter of 31 cm. 11 of 18 landing points lie
  within ±1σ (about 12 expected for a correctly sized band; D2: 17/18, too wide).
- The split between C and s_ε is weakly determined. With L fitted from the clicks instead, it becomes
  √C = 1.7 cm and s_ε = 0.52°. σ_shot at landing hardly changes, so this does not matter for aiming.
- The parameter σ (σ_curve) ignores the correlation between points of the same shot, so it is somewhat
  too small. This hardly matters, because σ_shot is dominated by the scatter terms.

### Measured L vs. clicked nozzle tips

The clicked tips alone prefer L = 0.898 m (with a free h = +5 cm), 9 cm longer than measured. With L = 0.81 m
the fitted pivot simply moves 8 cm forward (x_p −1.040 → −0.962 m), and the nozzle residuals barely change.
Over θ = −3° … 25° the clicks cannot separate a longer L from a pivot further back, so this is no sign of a
click → metre scale error. Fixing L = 0.81 m costs very little: the jet start (first 1.5 m) fits with
2.3 cm RMS instead of 1.8 cm, and the leave-one-run-out angle RMS goes from 0.76° to 0.80°.

## 7. Files

| File | Content |
|---|---|
| `einfach.py` | `duese_mit_massen` (pivot with fixed L, h), `duese_fit_L` (L from clicks, h fixed), `EinfachModell` (z, steigung, sigma_z, sigma_schuss, aufprall, x_start; same interface as `geometrie.GeometrieModell`), `fit_einfach`, `_streuung` (weighted NNLS), `duese_vergleich`, `zusammenfassung`, `speichere` (JSON) |
| `strahl.py` | `DUESE_L` / `DUESE_H` at the top; `fit` (default `--modell einfach`), `--grad-d`, `--duese-L` / `--duese-h`; `ballistik-test` (section 10) |
| `zielmodell.py` | standalone inverse (target → θ) for the ROS package: model E, plus the ballistic branch below −3° if the JSON has a `ballistik` block (section 10) |
| `ballistik.py` | ballistic point-mass model with quadratic drag (`BallistikModell`, `fit_ballistik`; one v0, drag k(θ) polynomial of degree `grad_k`), damped scatter NNLS (`streuung`), extrapolation test (`validierung`, `plot_validierung`). Tested, not adopted (section 10) |
| `geometrie.py` | `modell_varianten(..., duese_masse)` includes E0/E1/E2 in `vergleich`; D2 code unchanged |

## 8. Commands

```bash
python strahl.py fit                       # E0 with DUESE_L / DUESE_H -> auswertung/*_einfach.*
python strahl.py fit --duese-L 0.85        # one-off run with a different L
python strahl.py fit --grad-d 0            # E1
python strahl.py fit --modell geometrie    # old D2 (--grad-theta 2 --grad-delta 2), or --modell frei for A/A3
python strahl.py vergleich                 # all models, leave-one-run-out
python strahl.py ballistik-test            # drag model: fit clips 10/15/20, predict 5 and 0 (section 10)
python strahl.py ballistik-test --grad-k 1 # same, one v0 for all angles, drag k(θ) linear in θ
python strahl.py ballistik-test --train 5,10,15 --val 0 --endsatz 0,5,10,15   # constant v0, k on flat shots
python strahl.py ballistik-test --mit-weg ...  # drag grows with flight distance, k = k0 + k_s·s
```

`fit` writes `fit_ergebnis_einfach.txt`, `fit_punkte_einfach.csv`, `fit_modell_einfach.json` and
`bahnen/residuen/aufprall_einfach.png` to `auswertung/`. On Windows, set `PYTHONIOENCODING=utf-8` if the
console garbles σ/θ.

## 9. Open points

1. **Pivot position in the test setup.** If it is known (x relative to the posts), compare it with
   x_p = −0.962 m (L = 0.81 m) and −1.040 m (L from clicks). That is the only way to check which L the test
   data support. Also make sure the measured L refers to the exit orifice, not the outside of the tip.
2. **`strahl.py winkel`** still uses D2. Aiming itself no longer needs it (see 3), so this only matters for
   quick checks here; `python zielmodell.py auswertung/fit_modell_einfach.json X Z` does the same for E.
3. **ROS node: done (24.11.2026).** `zielmodell.py` (CORD `cord_nozzle_aim` and the copy here) now implements
   only model E (`"art": "einfach"`; a D2 JSON is rejected). `dist_angle_mapping`, `aim_test` and
   `test_hold_target` default to `config/fit_modell_einfach.json`; the old `fit_modell_geometrie.json` was removed
   from the package. The inverse (target → θ, restricted to the fitted θ range, refuses targets outside it) is
   `Zielmodell.angle_for_target`. `test_zielmodell.py` checks it against `einfach.lade_einfach(...)`: forward
   model, monotonicity (min ∂z/∂θ = 1.5 cm/°), round trip of the inverse, rejection of unreachable targets.
   After a refit: copy `auswertung/fit_modell_einfach.json` to `cord_nozzle_aim/config/`, run
   `python test_zielmodell.py`, rebuild the package (`colcon build`).
4. Update the main model write-up (D2 → E0 as the chosen model).
5. **Targets below −3°**: since 24.11.2026 `zielmodell.py` reaches them down to −20° with the ballistic extension
   (section 10, integration). It is validated only 4.7° below the data. Shots at −5° … −20° would validate it.
6. **Far, low targets** (e.g. 8 m ahead, 1.5 m below the pivot) are still rejected (`beyond_landing`): they need
   θ ≈ −3° … 0°, where E0 only describes the jet down to the test ground. The ballistic model (fitted on −3° … 11°)
   could continue those jets behind the E0 landing point. Not done yet.

## 10. Ballistic extension for θ < −3° (tested 24.11.2026, not adopted)

### Idea

E0 is only valid for the measured θ = −3° … 25°. Its polynomials in θ must not be extrapolated, so targets that
need a flatter jet are rejected. In flight this is common: the drone is higher than the test nozzle (1.31 m), so
the target lies steeply below. The idea was a physical model with only two constants, which might extrapolate
to −20° where a polynomial cannot.

### Model (`ballistik.py`)

A point mass with quadratic air drag, starting at the E0 nozzle exit (same pivot, L = 0.81 m, h = 0):

```
x(0) = x_d(θ),  z(0) = z_d(θ),  v(0) = v0·(cos θ, sin θ)
ẍ = −k·|v|·vx,   z̈ = −g − k·|v|·vz
```

v0 (exit speed) and k (drag) are the same for all angles. RK4 with dt = 1 ms, z(x) linear between samples.
The fit uses the same data, weights and two passes as E0 (`scipy.optimize.least_squares`). The shot scatter
uses the same r² ≈ C + s_ε²Δx² + s_α²Δx⁴ model. `einfach._streuung` does not converge on the ballistic
residuals: it alternates between "only C" and "only s_α", because the large systematic far residuals throw
the 1/v weights back and forth. `ballistik.streuung` therefore damps the update (v ← √(v·v_new)). It has the
same fixed point and gives the same result on the E0 residuals (to 0.1 µm).

### Test (`python strahl.py ballistik-test`)

- Fit on clips 10, 15, 20: θ = 6, 7, 11, 15°. These are **6 shots** (10_a/b, 15_a/b, 20_a/b), because clips 10,
  15 and 20 were removed from `punkte/` as unreliable.
- Predict clips 5 (θ = 2°, 4.0° below the lowest training angle) and 0 (θ = −2/−3°, 8.7° below), 6 shots that
  are not in the fit.
- Baselines: E0 fitted on the same training clips and extrapolated, and E0 on all clips (in-sample best case).
- Pass criteria, fixed before the run: landing RMS ≤ 30 cm, angle RMS ≤ 0.8°, at most 10 % worse than E0
  extrapolated, at least 50 % of the validation points within ±1σ (σ_shot plus the click σ of the point), and
  landing RMS of group 0 at most 15 cm above group 5.

Angle error = θ at which the model best fits the shot's points − measured θ (`geometrie.winkel_schaetzen`).
The values in brackets are means, i.e. the systematic part.

| Group | Model | Landing RMS (mean) | Angle RMS (mean) | Points in ±1σ | RMS Δx > 6 m |
|---|---|---|---|---|---|
| 5 (4° gap) | **Ballistic, training fit** | 69 cm (−65) | 1.28° (+1.23) | 50 % | 43 cm |
| | E0 extrapolated | 174 cm (−172) | 2.47° (+2.46) | 43 % | 120 cm |
| | E0 all clips (reference) | 31 cm (−19) | 0.51° | 71 % | 17 cm |
| 0 (9° gap) | **Ballistic, training fit** | 50 cm (−46) | 0.75° (+0.63) | 33 % | 23 cm |
| | E0 extrapolated | 240 cm (−240) | 4.02° (+3.92) | 40 % | 193 cm |
| | E0 all clips (reference) | 20 cm (+9) | 0.28° | 80 % | 7 cm |
| all 6 | **Ballistic, training fit** | **60 cm (−56)** | **1.05° (+0.93)** | **41 %** | 35 cm |
| | E0 extrapolated | 210 cm (−206) | 3.33° | 41 % | 161 cm |

Training fit: v0 = 17.2 ± 0.4 m/s, k = 0.118 ± 0.007 1/m, correlation +0.94, χ²_red = 1.59.

**Result: failed** (landing 60 cm > 30 cm, angle 1.05° > 0.8°, 41 % < 50 % within ±1σ). Passed: much better
than polynomial extrapolation (3.5× smaller landing error), and the error does not grow from the 4° to the 9°
gap. All 6 validation shots land beyond the prediction (model 22–95 cm short), so the error is systematic, not noise.
Plot and full table: `auswertung/ballistik_test.png`, `auswertung/ballistik_test.txt`.

### Why: v0 and k are not constant over θ

Fitted per angle group, the drag model fits each angle well (χ²_red 0.2–3), but the constants drift:

| Group | θ | v0 [m/s] | k [1/m] |
|---|---|---|---|
| 0 | −2.7° | 18.7 ± 1.0 | 0.112 ± 0.024 |
| 5 | 2.0° | 20.1 ± 0.9 | 0.132 ± 0.016 |
| 10 | 6.5° | 18.7 ± 0.6 | 0.141 ± 0.011 |
| 15 | 11.0° | (not determined, 2 short shots) | |
| 20 | 15.0° | 15.4 ± 0.5 | 0.103 ± 0.009 |
| 25 | 19.7° | 17.5 ± 0.5 | 0.116 ± 0.007 |
| 30 | 24.3° | 16.7 ± 0.6 | 0.094 ± 0.007 |

v0 and k are strongly correlated (r ≈ 0.95) within each group. Still, the flat shots want a faster or "less
draggy" jet than the 11–15° shots, and in the sense of the model the steep shots also carry further than one
(v0, k) allows. Consequences:
- A single (v0, k) fitted on **all** clips leaves 74 cm residual RMS at 8–13 m from the nozzle (E0: 24 cm). It
  is too low by 0.9–1.5 m at 24–25° and too high by up to 0.7 m at 15–20°.
- Training → all clips moves v0 by +1.5 m/s (3.8 σ). The constants depend on which angles are used, so they
  are not physical properties of pump and nozzle.

A point mass with constant quadratic drag does not describe this jet: it breaks up into spray, and how far that
spray carries depends on the angle.

### Second try: one v0 for all angles, drag varies with θ

The exit velocity is a property of pump and nozzle, so it must be the same for every shot. Only the drag is
allowed to depend on the angle (`--grad-k`):

```
k(θ) = max(0, k0 + k1·t),   t = θ/30°        (3 parameters: v0, k0, k1)
```

**Shape of k(θ) first.** One shared v0 and a free k per measured angle (11 values), fitted on all clips:

| θ | −3° | −2° | 2° | 6° | 7° | 11° | 15° | 19° | 20° | 24° | 25° |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k [1/m] | 0.102 | 0.106 | 0.096 | 0.138 | 0.131 | 0.121 | 0.141 | 0.121 | 0.125 | 0.109 | 0.106 |

(σ 0.005–0.011, v0 = 18.1 ± 0.3 m/s, χ²_red = 1.64, residual RMS 19 cm at 8–13 m, E0: 24 cm.) So one v0 with an
angle-dependent drag describes all data as well as E0. But k(θ) is a hump: about 0.10 below 3°, 0.12–0.14 at
6–20°, and falling again towards 25°. The training clips (6–15°) only see the rising side.

**Test with linear k(θ)** (`python strahl.py ballistik-test --grad-k 1`, same split and criteria as above):

Training fit: v0 = 16.7 ± 0.4 m/s, k0 = 0.092 ± 0.012, k1 = 0.051 ± 0.021 1/m (k = 0.087 at −3°, 0.118 at 15°,
0.058 at −20°), χ²_red = 1.39.

| Group | Model | Landing RMS (mean) | Angle RMS (mean) | Points in ±1σ | RMS Δx > 6 m |
|---|---|---|---|---|---|
| 5 (4° gap) | **v0 + linear k(θ)** | 47 cm (−40) | 1.06° (+1.00) | 50 % | 25 cm |
| 0 (9° gap) | **v0 + linear k(θ)** | 31 cm (−25) | 0.48° (+0.37) | 53 % | 11 cm |
| all 6 | **v0 + linear k(θ)** | **39 cm (−33)** | **0.83° (+0.68)** | **52 %** | 19 cm |
| all 6 | v0 + constant k (first try) | 60 cm (−56) | 1.05° (+0.93) | 41 % | 35 cm |
| all 6 | E0 extrapolated | 210 cm (−206) | 3.33° | 41 % | 161 cm |
| all 6 | E0 all clips (reference) | 26 cm (−5) | 0.41° | 76 % | 13 cm |

**Result: failed, narrowly** (landing 39 cm > 30 cm, angle 0.83° > 0.80°). Passed: ±1σ coverage (52 %), much
better than E0 extrapolated, and no growth from the 4° to the 9° gap (group 0 alone would pass). All six
shots still land beyond the prediction (model 2–70 cm short, mean −33 cm), so a systematic part remains. Plot: `auswertung/ballistik_test_k1.png`.

Caveats, even though it comes close:
- Linear k(θ) does not have the shape of the data. Fitted on **all** clips, the hump flattens the slope to
  k1 = 0.011 ± 0.009, the fit gets worse (χ²_red 3.8, 83 cm RMS at 8–13 m), and training → all moves v0 by 4 σ.
  The training fit only does well because 6–15° lie on the rising side of the hump, which points the right way
  for lower angles.
- The extrapolated k falls with the angle (0.058 at −20°, half the measured values). Whether the drag keeps
  falling below −3° is not covered by any data. Constant k below the lowest measured angle would be the
  conservative choice.

### Third try: constant v0 and k, fitted only on the flat shots (clips 0–15, θ ≤ 11°)

Physically, v0 (pump, nozzle) and k = ρ_air·C_d·A/(2m) (drag per mass) do not depend on the launch angle, so
the 2-parameter model with constant v0 and k is kept. A k that varies with θ has no physical reason. Spray
break-up would make the drag change with flight time or distance, not with θ. The scatter of the per-angle k
values (2°: 0.096, 6°: 0.138) is more likely the angle measurement (whole degrees in `messungen.csv`, up to
±0.5° error) and pump variation between runs. What changes instead is the fit range. Targets below −3° need flat
and downward shots, so the model is fitted only where the jet is flat (clips 0, 5, 10, 15), not on the 15–25°
shots, which do not fit the same constants.

With clips 0–15 as the final data, the test has to hold out the lowest angles inside that set. Same criteria:

| Test | Training | Validation (gap) | Landing RMS (mean) | Angle RMS (mean) | In ±1σ | Result |
|---|---|---|---|---|---|---|
| 1 | 5/10/15: 7 shots, θ = 2–11° | 0: 3 shots, −2/−3° (4.7°) | **22 cm (−13)** | **0.21° (−0.11)** | 67 % | **passed** |
| | E0 extrapolated, same training | | 271 cm | 4.65° | 27 % | |
| 2 | 10/15: 4 shots, θ = 6–11° | 5 (4.0°) and 0 (8.7°) | 52 cm (−47) | 0.89° (+0.78) | 55 % | failed |
| | E0 extrapolated, same training | | 96 cm | 2.34° | 97 % | |

Commands: `python strahl.py ballistik-test --train 5,10,15 --val 0 --endsatz 0,5,10,15` and
`--train 10,15 --val 5,0`. Plots: `auswertung/ballistik_test_train5-10-15_val0.png`,
`auswertung/ballistik_test_train10-15_val5-0.png`.

Fits (constant v0, k):

| Data | v0 [m/s] | k [1/m] | χ²_red | RMS 8–13 m |
|---|---|---|---|---|
| clips 5/10/15 (test 1 training) | 19.83 ± 0.63 | 0.141 ± 0.010 | 3.8 | |
| **clips 0/5/10/15 (final)** | **19.74 ± 0.50** | **0.139 ± 0.009** | 3.1 | **24 cm** (E0 on all clips: 24 cm) |
| clips 10/15 (test 2 training) | 17.49 ± 0.39 | 0.117 ± 0.007 | 0.9 | |
| all clips | 18.73 ± 0.31 | 0.126 ± 0.005 | 3.5 | 74 cm |

- **Test 1 is clean.** The fit on 2–11° predicts the −2/−3° shots within the shot scatter. Adding clip 0 to the
  fit then moves v0 and k by only 0.1–0.2 σ, so the flat shots are consistent with one (v0, k). Landing error of
  the final fit on its own 10 shots: RMS 30 cm (−57 … +44 cm), the same as the measured landing scatter.
- **Test 2 fails because of v0/k degeneracy.** v0 and k are correlated (r = 0.95): a faster jet with more drag
  gives almost the same curve. Over only 6–11° the two cannot be separated (χ²_red 0.9, so the training fit is
  good), and the pair it picks (17.5 m/s, 0.117) is wrong for flatter shots. Clip 5 (2°) is what pins the pair
  down in test 1.
- Only 3 validation shots in test 1, so its numbers are rough. The test shows 4.7° of extrapolation from a 9°
  wide fit range. The target (−20°) is 17° below a 14° wide range. In favour: steep downward shots to a target a
  few metres away are short, so drag (k, the poorly determined part) matters less there, and v0 matters more.

### Fourth try: drag that grows with the flight distance (jet break-up)

Physical idea: unlike a ball, the jet breaks up into spray the longer it flies, so it is not the same object
anymore. That argues for a drag that depends on how far the jet has travelled, not on θ directly (θ is only a
proxy: steeper shots fly longer). One law for all angles (`--mit-weg`, path length s integrated along the flight):

```
k(s) = max(0, k0 + k_s·s)        (3 constants: v0, k0, k_s)
```

| Fit on | v0 [m/s] | k0 [1/m] | k_s [1/m²] | RMS 8–13 m |
|---|---|---|---|---|
| all clips | 20.6 ± 1.1 | 0.192 ± 0.035 | **−0.015 ± 0.008** | 47 cm (constant k: 74, E0: 24) |
| clips 10–30 | 19.5 ± 1.3 | 0.218 ± 0.047 | **−0.027 ± 0.011** | |
| clips 0–15 | 18.3 ± 1.7 | 0.079 ± 0.069 | +0.016 ± 0.018 | 25 cm |

| Test | Landing RMS | Angle RMS | Result | Constant k, same test |
|---|---|---|---|---|
| train 10/15/20 → 5, 0 | 57 cm | 1.04° | failed | 60 cm, 1.05° |
| train 5/10/15 → 0 | 20 cm | 0.19° | passed | 22 cm, 0.21° |
| train 10–30 → 5, 0 | 66 cm | 1.06° | failed, worse than E0 extrapolated (44 cm) | |

- Where the long flights are in the fit (steep shots), k_s comes out **negative**: the jet carries further on long
  flights than constant drag allows, the opposite of "more drag after break-up". A possible reason is that the
  jet drags the surrounding air along (entrainment), and droplets in each other's wake see less drag, so the
  relative air speed drops along the flight. The clicked spray centre at large distances may also not be the
  centre of mass.
- On the flat shots (0–15) k_s is not significant (+0.016 ± 0.018), and it is almost completely degenerate with
  v0 and k0 (correlations 0.96–0.99). The σ of v0 grows from 0.5 to 1.7 m/s. The hold-out test gives the same
  result as constant k.
- So a growing drag is plausible physics, but these data do not show it. For aiming below −3° (short flights,
  path < ~6 m), constant v0 and k on clips 0–15 (third try) remains the model. It has fewer parameters and the
  same test result.

### Options for targets below −3°

1. **Measure lower angles** (−5 … −20°, e.g. with the nozzle raised or the drone at height). This is the only
   way to an E0 fit that covers the range. Short, steep downward jets are quick to click.
2. **Ballistic model with constant v0 and k, fitted on clips 0–15** (third try). Passed the hold-out test for the
   lowest angles (22 cm, 0.21°). **Chosen and integrated** (next section). Not validated beyond 4.7° below the
   fit data, and the target is 17° below.
3. Keep rejecting (`too_low`) as now.

### Integration (24.11.2026)

**Fit and JSON.** `python strahl.py fit` now also fits the ballistic model on clips 0, 5, 10, 15 (constant v0, k,
**same pivot as E0**, so both start at the same nozzle exit) and writes it as the block `"ballistik"` into
`fit_modell_einfach.json` (`ballistik.fit_fuer_ziel`; `--ohne-ballistik` leaves it out). With the E0 pivot:
v0 = 19.43 ± 0.49 m/s, k = 0.135 ± 0.009 1/m (own pivot: 19.74 / 0.139), s_α = 0.0028 1/m.
The block also holds `theta_min_grad = −20` and the model-error coefficient (below).

**`zielmodell.py`** (both copies, still standard library only):
- θ ≥ θ_min (−3°): model E, unchanged. Targets on or above the −3° jet are solved exactly as before.
  (Test: identical θ for all previously reachable targets; a JSON without the block behaves as before.)
- Targets below the −3° jet: bisection in [−20°, −3°) with the ballistic jet (RK4 in pure Python, dt = 1 ms,
  integrated only up to the target, 3–20 ms per target).
- **Continuity at −3°:** E0 and the ballistic jet agree within 1.5 cm up to 5 m from the nozzle, but differ by
  11 cm at the E0 landing point. The ballistic branch therefore adds c(Δx) = z_E0(−3°) − z_ballistic(−3°)
  (tabulated along the jet, constant behind the E0 landing point), faded linearly to 0 between −3° and −8°.
  Jump with correction: 0.0 cm; dz/dθ > 0 everywhere from −20° to 25° (min 1.5 cm/°).
- **No ground check** in this branch: in flight the target may lie below the test ground. Only "in front of the
  nozzle" and the θ bounds remain. `too_low` now means below the −20° jet. `Result.branch` is `"E0"` or
  `"ballistic"`, and the message says "ballistic extrapolation".
- **Uncertainty:** σ² = σ_curve² (numerical gradient w.r.t. v0, k and their 2×2 covariance) + shot scatter
  (s_ε, s_α of the ballistic fit) + σ_model², with σ_model = c·(θ_min − θ)·Δx. c is calibrated from the hold-out
  test (fit 5/10/15, predict clip 0): RMS of r / (angle gap · Δx) over the validation points with Δx > 1 m,
  c = 0.27 cm per degree per metre. Example: θ = −15°, 4 m from the nozzle: σ_model = 13 cm.

**Reach** (relative to the pivot, from `angle_for_target`):

| below the pivot | 0.5 m | 1.0 m | 1.5 m | 2.0 m | 2.5 m | 3.0 m | 4.0 m |
|---|---|---|---|---|---|---|---|
| new: reachable at (1 m grid, up to 10 m) | 2–10 m | 3–10 m | 4–7 m | 5–8 m | 6–8 m | 7–9 m | 8–10 m |
| before | 5–10 m | 7–10 m | – | – | – | – | – |

At the near end the angle is about −13° … −18°, at the far end about −4°.

Closer targets need a jet flatter than −20° (e.g. 4 m ahead and 2.5 m below: about −30°) and stay `too_low`.
Far targets more than 1.31 m below the pivot (below the test ground) at small angles stay `beyond_landing`
(open point 6).

**Checks.** `python test_zielmodell.py` (new section 5): forward model against `ballistik.BallistikModell` plus
correction (max 3.5 µm), continuity, monotonicity, round trip −20° … −3° (20 targets, 6 below the test ground,
max error 2e-7°), JSON without the block unchanged. ROS: `test_hold_target.py` passes. `dist_angle_mapping` with
the real yaml and fake IMU/target publishers: target 5 m ahead and 1.5 m below the pivot publishes φ = −8.65° on
`/leader/nozzle_degrees` (before: `too_low`); 4 m / 2.5 m below gives the `too_low` warning.

