# Water jet model: D2 (chosen) and A3 (benchmark)

Data: field test of 06.11.2026, 19 clicked clips (all except `15` and `20`), 3 runs, measured
nozzle angles θ = −3° … 25°. All numbers below come from `python strahl.py vergleich` and
`python strahl.py fit ...` on these data (12.11.2026).
The figures are in `auswertung/`, which is not in git. Run the commands in section 6 to recreate them.

**Summary.** Both models predict equally well: about **40 cm** landing error and **0.7°** angle error
for a run they were not fitted on. **D2** is the chosen model. It starts at the nozzle, in the direction
the nozzle points. It knows how much the jet scatters from shot to shot, so its uncertainties are honest
(on the safe side). And it can be inverted directly: target → nozzle angle. **A3** is a free polynomial.
It serves as an empirical benchmark, but its curve near the nozzle has no physical meaning, and it
underestimates its own uncertainty by a factor of 2–4.

---

## 1. Coordinates and notation

- x = horizontal distance in the jet plane (the posts are measured from the nozzle), z = height above the test ground.
- θ = **measured, absolute** nozzle angle (`messungen.csv`; the drone tilt is already included).
- t = θ / 30° (scaled angle, keeps the polynomial coefficients of similar size).
- Each clip contributes about 5 points: nozzle tip, 3–4 points on the jet centre line, and the landing point.

## 2. Model A3: free polynomial up to x³

```
z(x, θ) = c0(θ) + c1(θ)·x + c2(θ)·x² + c3(θ)·x³
c_j(θ)  = p0j + p1j·t + p2j·t²
```

- 4 × 3 = **12 parameters**, all free. The fit is linear (weighted least squares, `modell.fit`). Each point is weighted
  by its click uncertainty (converted to metres) and, for the landing point, by the width of the spray cone.
- It is the original model `strahl.py fit` with `--grad-x 3` instead of 2.
- Uncertainty: the scatter around the curve is larger than the click errors explain (χ²_red = 2.6), so all σ are
  scaled up by √χ²_red. There is no separate term for the scatter between shots.

**Why x³ at all?** Drag makes the jet slow down along its path, so the descending branch is steeper
than the ascending one. A parabola is symmetric and cannot show that. Going from the free parabola (A)
to A3 halves the angle error (1.44° → 0.74°).

## 3. Model D2: tied to the nozzle geometry

### 3.1 Nozzle position

The nozzle tip moves on a circle around the servo pivot when the angle changes:

```
x_d(θ) = x_p + L·cos θ − h·sin θ
z_d(θ) = z_p + L·sin θ + h·cos θ
```

These four constants are fitted from the clicked nozzle tips of all 19 clips:
pivot x_p = −1.04 m, z_p = 1.25 m, arm L = 0.90 m, h = +0.05 m (offset across the nozzle axis).
The clicked nozzles lie within 2.3 cm (x) and 1.6 cm (z) of this circle (RMS). Only clip `5` is off by 7 cm.
Using the fitted circle instead of each clip's own click averages out the click error,
and it gives the nozzle position for any angle, which the drone needs.

### 3.2 Jet relative to the nozzle

With Δx = x − x_d(θ) and Δz = z − z_d(θ):

```
Δz = tan(θ + δ(θ))·Δx  +  sec²θ·a(t)·Δx²  +  sec³θ·b(t)·Δx³

δ(θ) = δ0 + δ1·t + δ2·t²      exit direction − measured angle
a(t) = a0 + a1·t + a2·t²      curvature (gravity)
b(t) = b0 + b1·t + b2·t²      asymmetry (drag)
```

What each term does:

| Term | Meaning | Why this form |
|---|---|---|
| no constant | the curve starts exactly at the nozzle tip | Δz = 0 at Δx = 0 |
| tan(θ + δ)·Δx | the jet leaves in the direction of the nozzle, plus a small offset δ | the exit direction is known from the geometry; only a small correction is fitted |
| sec²θ·a·Δx² | bending by gravity | with a constant exit speed (pump fixed at 25 Hz), a drag-free jet has exactly this θ-dependence |
| sec³θ·b·Δx³ | drag: the descending branch is steeper | leading-order drag correction (same form for linear and quadratic drag) |
| t, t² in a, b, δ | small empirical corrections in θ | the data show an angle dependence that the pure sec terms don't capture (section 4) |

That gives **9 fixed parameters**. Fitted values:

```
δ(θ):  −3°: +1.4 ± 1.1°   4°: −0.1 ± 0.6°   11°: −1.0 ± 0.8°   18°: −1.3 ± 0.7°   25°: −1.0 ± 1.3°
a0 = −0.0145   a1 = +0.054   a2 = −0.052
b0 = −0.0019   b1 = −0.0078  b2 = +0.0077
```

The exit offset δ stays within about ±1.5°. That is plausible for an inclinometer with 1° steps
plus small effects of the nozzle geometry.

### 3.3 Scatter from shot to shot

Shots at the same measured angle land up to 0.8 m apart (e.g. 20_a 9.76 m and 20_b 9.02 m, both 15°).
Click errors don't explain this. D2 therefore gives every clip two random deviations from the mean curve:

```
ε_k ~ N(0, σ_ε²)   extra angle of this shot      →  changes tan(θ + δ + ε_k)
α_k ~ N(0, σ_α²)   extra curvature of this shot  →  + α_k·sec²θ·Δx²
```

and scales the assumed click uncertainty by a factor f. The three numbers are estimated from the data
(restricted maximum likelihood, REML):

| | Value | Meaning |
|---|---|---|
| σ_ε | 1.5° | per-shot angle scatter: angle reading + exit direction |
| σ_α | 0.0049 1/m | per-shot curvature scatter (pump, wind); ±32 cm in height at 8 m |
| f | 0.65 | the clicks are more precise than the assumed 3 px |

From this, D2 gives two uncertainties:

- **σ_curve**: how well the *mean* curve is known (shaded band in the plots).
- **σ_shot**: how far a *new* shot can be expected to land from the prediction (dotted lines). This is the one
  that matters for aiming.

### 3.4 How the fit works

Everything is linear in the parameters except tan(θ + δ). The fit linearises tan around the current δ,
solves a weighted least-squares problem (generalised least squares, including the shot scatter), updates δ,
and repeats until δ stops changing (a few iterations). The three scatter values are re-estimated in every
iteration. Code: `geometrie.fit_geometrie(clips, grad_x=3, grad_theta=2, grad_delta=2)`.

## 4. Why D2 and not A3

### 4.1 Prediction accuracy: a tie

Main test: **leave-one-run-out (LORO).** Fit on two runs, predict the third, repeat for all three runs.
Two error measures:

- **landing error** = model landing point at the measured θ − clicked landing point
- **angle error** = measured θ − the angle at which the model best matches the clicked points of that shot
  (= how far off you would have aimed with this model)

| Model | Params | Landing RMS (mean, max) | Angle RMS | Angle per run 1 / 2 / 3 |
|---|---|---|---|---|
| A: free parabola (old) | 9 | 54 cm (+21, 112) | 1.44° | 1.37 / 1.12 / 1.74 |
| **A3: free cubic** | 12 | 42 cm (+6, 84) | 0.74° | 0.79 / 0.69 / 0.74 |
| C: tied cubic, θ-dependence from sec only | 3 | 55 cm (+1, 119) | 1.01° | 0.85 / 0.70 / 1.33 |
| **D2: tied cubic + t, t², δ(θ)** | 9 | **40 cm (−1, 84)** | **0.72°** | 0.56 / 0.83 / 0.70 |

A3 and D2 are equally good. The difference (0.02°, 2 cm) is far below what 19 clips can resolve.
Both are clearly better than the old free parabola A and the simple tied cubic C.
Other variants tried (tied parabola, C with only part of the corrections) are in `auswertung/vergleich.txt`;
none comes close.

![Comparison of A, A3, C, D2 at four angles](auswertung/vergleich_bahnen.png)

### 4.2 Behaviour at the nozzle: D2 is physical, A3 is not

Jet direction at the nozzle minus the measured angle:

| Model | −3° | 2° | 6° | 11° | 15° | 20° | 24° | RMS of the points within 1.5 m of the nozzle |
|---|---|---|---|---|---|---|---|---|
| A3 | +1.6° | −2.3° | −4.1° | −4.7° | −4.0° | −1.6° | +1.2° | 1.9 cm |
| D2 | +1.4° | +0.3° | −0.4° | −1.0° | −1.2° | −1.3° | −1.1° | 1.8 cm |

A3 lets the jet leave up to 4.7° below the nozzle direction, then bends it back up with extra curvature.
Both models fit the points near the nozzle equally well, so the data do not ask for this: A3 just has the
freedom to trade slope against curvature. A water jet leaves along the nozzle axis, and D2 respects that.
A3 is a good interpolator *inside* the measured data, but its coefficients have no physical meaning.

### 4.3 Honest uncertainties: D2 knows about shot scatter, A3 does not

Predicted uncertainty of the landing point vs. the error actually found in LORO (≈ 40 cm):

| Model | Predicted σ of the landing point | Compared with the real 40 cm |
|---|---|---|
| A3 | 6–21 cm | **2–4× too optimistic**: it treats the scatter between shots as if it were known |
| D2, σ_shot | 62–79 cm | about 1.6× **on the safe side**: all 19 clicked landing points lie inside ±1σ |

For aiming, an uncertainty that is too small is dangerous (overconfident). One that is somewhat too large
is acceptable. Also, only D2 separates "how well do we know the curve" from "how much does a single shot scatter".

![D2: landing distance vs. angle](auswertung/aufprall_geometrie.png)

### 4.4 Usable for aiming

| | A3 | D2 |
|---|---|---|
| Curve starts at the nozzle | only approximately | exactly, for every angle (nozzle circle) |
| Target relative to the drone | target relative to the test ground only | target relative to the servo pivot |
| Inverse (target → angle) | not implemented | `winkel_fuer_ziel`, with σ_θ |
| Validity check | no | refuses targets outside the measured range |
| Export | npz | npz + json (with the formula) |

**Decision:** D2 for aiming, A3 as the empirical benchmark in the thesis.

### 4.5 Where D2 is weak (state these openly)

- **Edges of the angle range.** With t² terms, the curve is less constrained at −3° and 24–25°
  (σ_curve of the landing point 0.33–0.39 m there, 0.20 m in the middle).
- **Uncertainties too large.** σ_shot is about 1.6× too large (section 4.3). One reason: some of what is counted
  as curvature scatter may really be error in clicking the landing point (the spray cone). The data cannot separate the two.
- **Parameter σ too small.** In a synthetic test (20 simulated data sets drawn from D2 itself), the parameters and
  the scatter values were recovered without bias, but the σ of the strongly correlated parameters δ0, a0, b0 came out
  1.5–1.8× too small. This hardly affects aiming, because σ_shot is dominated by the scatter terms, which were recovered correctly.
- **No exit speed.** The coefficients don't give a realistic exit speed (drag is too strong for the leading-order
  approximation). Treat a and b as empirical.

## 5. How D2 is used in practice (aiming)

### 5.1 From target to nozzle angle

1. **Target relative to the servo pivot.** The camera/IMU gives the fire position relative to the drone, and the pivot
   is fixed to the drone. Input: horizontal distance x and height z of the target relative to the pivot.
2. **Solve for θ.** For each angle in the measured range, the model checks whether the target lies between
   the nozzle and the landing point, and computes the height of the curve at the target's x. Then it finds the θ
   where curve height = target height (`brentq`). This is unique, because the curves rise monotonically with θ at
   every x (smallest ∂z/∂θ = 1.6 cm/°, checked on a grid).
3. **Uncertainty.** σ_θ = σ_shot(target) / |∂z/∂θ|.
4. **Not reachable**, with no extrapolation, if the target is
   - above the steepest measured jet (25°) or beyond its landing point → "zu hoch / zu weit" (too high / too far),
   - below the flattest jet (−3°) → "zu tief" (too low),
   - in front of the nozzle or behind every landing point → "ausserhalb" (outside the range).
5. **Command to the servo** (outside this model): θ is absolute, so θ_rel = θ − pitch_IMU, then invert the
   servo calibration (servo setting ≠ measured angle: 20° → 15°, 30° → 24–25°; still to be calibrated).

### 5.2 Examples (target relative to the pivot; the pivot is 1.25 m above the test ground)

| Target x, z | Meaning | θ | σ_θ | σ_shot at the target |
|---|---|---|---|---|
| 4.0 m, +1.5 m | high, close | 23.0° | 1.6° | 12 cm |
| 5.0 m, +1.0 m | | 15.1° | 1.8° | 15 cm |
| 6.0 m, 0 m | pivot height | 5.3° | 2.0° | 19 cm |
| 8.0 m, 0 m | pivot height | 11.1° | 2.9° | 34 cm |
| 8.0 m, −1.25 m | ground of the test site | 0.1° | 3.0° | 33 cm |
| 10.0 m, −1.25 m | ground of the test site | 11.1° | 4.2° | 52 cm |

How to read this: close targets can be hit within about 15 cm. At ground level, 8–10 m away,
a single shot scatters by ±35–50 cm. That comes from the jet itself (pump, wind), not from the model.
σ_θ is the angle that corresponds to this scatter. It does not mean the computed angle is wrong by that much.

### 5.3 Commands and code

```bash
python strahl.py metrisch                                            # clicks → metres
python strahl.py fit --modell geometrie --grad-theta 2 --grad-delta 2  # fit D2 → auswertung/*_geometrie.*
python strahl.py winkel 8 0                                          # angle for a target 8 m / 0 m from the pivot
python strahl.py winkel 9 0 --bezug boden                            # target in test-ground coordinates
python strahl.py vergleich                                           # all models, leave-one-run-out
python strahl.py fit --grad-x 3                                      # A3 (benchmark) → auswertung/fit_ergebnis.txt etc.
```

In Python (e.g. in `dist_angle_mapping.py`):

```python
import geometrie
m = geometrie.lade_geometrie("auswertung/fit_modell_geometrie.json")
e = m.winkel_fuer_ziel(8.0, 0.0)                 # target relative to the pivot
if e["grund"] == "ok":
    theta_abs, sigma_theta = e["theta"], e["sigma_theta"]
else:
    print("nicht erreichbar:", e["grund"])
```

`fit_modell_geometrie.json` contains all parameters, the nozzle circle, the scatter values, the valid angle range,
and the formula as text, so the model can also be re-implemented without this repository.

### 5.4 Valid range and when to refit

- Valid only for θ = −3° … 25°, between the nozzle and the landing point, and for this nozzle, this pump setting (25 Hz)
  and similar wind. Change any of these → measure again and refit.
- After new measurements: click → `strahl.py metrisch` → `strahl.py vergleich` (check that D2 is still the best)
  → `strahl.py fit --modell geometrie --grad-theta 2 --grad-delta 2`.
- More points on the descending branch, and clips at the edges of the angle range, would make the t² terms
  more reliable.

## 6. Files and figures

| File | Content |
|---|---|
| `geometrie.py` | D2 (and the other tied models), fit, inverse, comparison |
| `modell.py` | A/A3 (free polynomial), shared plots |
| `auswertung/fit_ergebnis_geometrie.txt` | D2: parameters, scatter values, table per clip |
| `auswertung/fit_modell_geometrie.json/.npz` | D2 for aiming |
| `auswertung/vergleich.txt` | all models, leave-one-run-out, behaviour at the nozzle |
| `auswertung/bahnen_geometrie.png` | D2 curves with data, ±1σ curve (band) and ±1σ shot (dotted) |
| `auswertung/residuen_geometrie.png` | residuals to the mean curve and to each clip's own curve |
| `auswertung/vergleich_bahnen.png`, `vergleich_residuen.png` | model comparison |

![D2: curves and data](auswertung/bahnen_geometrie.png)

![D2: residuals](auswertung/residuen_geometrie.png)

The upper residual plot (to the mean curve) shows the scatter between shots, which grows with distance.
The lower one (to each clip's own curve, i.e. with ε_k and α_k) shows that the *shape* of the curve fits every
shot to within a few cm. The one outlier is the landing point of 25_b, which does not match its own trajectory
(probably a click in the spray).
