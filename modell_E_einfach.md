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
| `strahl.py` | `DUESE_L` / `DUESE_H` at the top; `fit` (default `--modell einfach`), `--grad-d`, `--duese-L` / `--duese-h` |
| `geometrie.py` | `modell_varianten(..., duese_masse)` includes E0/E1/E2 in `vergleich`; D2 code unchanged |

## 8. Commands

```bash
python strahl.py fit                       # E0 with DUESE_L / DUESE_H -> auswertung/*_einfach.*
python strahl.py fit --duese-L 0.85        # one-off run with a different L
python strahl.py fit --grad-d 0            # E1
python strahl.py fit --modell geometrie    # old D2 (--grad-theta 2 --grad-delta 2), or --modell frei for A/A3
python strahl.py vergleich                 # all models, leave-one-run-out
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
