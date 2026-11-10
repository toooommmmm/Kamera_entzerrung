# Water jet trajectory model – plan (from discussion, 09.11.2026)

Context: BA "CORD Fire Detection and Aiming". Field test on 06.11.2026 (Test 3 in
`Testplan_Testtag_06.11.docx`). Goal: build a **mainly empirical model of the water
jet centre line** for different nozzle angles. The model will later replace the
"straight jet" assumption in
`CORD/cord_ws/src/cord_nozzle_aim/cord_nozzle_aim/dist_angle_mapping.py`
(input: target distance/height → output: nozzle angle).

**Do not start implementing without the user's go-ahead. Read existing code/data first.**

## Data (folder `ipztest_06_11/`)

- `iphone_videos/` – 24 clips `A001_1106HHMM_C0xx.mov` (side view, phone on tripod,
  roughly perpendicular to the jet plane). **The trajectory is built only from these videos.**
  Settings: main camera 26 mm, f/1.6, 1/100 s, 24 fps, ISO 34 (fixed).
  Mapping clip → run/angle still to be established (use timestamps in file names + log).
- `Testresultate.txt` – test log written on site: nozzle tip height 1.3 m above ground
  at 0°, pump 25 Hz (constant), drone tilt −3°, 3 runs ("Messung 1–3"), per entry:
  commanded angle, **directly measured nozzle angle**, landing distance ("Aufprall", m).
- `rosbags/` – OAK-D recordings, **not used** for the trajectory.

Differences to the test plan: angles were **0, 5, 10, 15, 20, 25, 30°** (not −20…45°);
no second phone filming the nozzle – the nozzle tilt was measured directly instead.

## Agreed approach

1. **Undistort** the videos. A colleague already calibrated the iPhone with a
   checkerboard and has an undistortion script – reuse it (ask the user for its location;
   verify it was calibrated with the same lens/resolution).
2. **Make the faint jet visible.** Weather was grey/cloudy; parts of the jet are hardly
   visible even by eye, so **automatic jet extraction is not reliable**. Instead, build one
   composite image per clip from all frames (temporal **standard deviation**, or max /
   median-background subtraction) – static scene goes dark, moving water stands out.
3. **Manual reconstruction of the centre line**: start point (nozzle tip, known geometry)
   + landing point + a few manually clicked points in between on the composite image.
   Preferred tool: small Python click tool (matplotlib `ginput` / OpenCV) that stores
   points with clip/run/angle. Tracker or WebPlotDigitizer are alternatives / cross-check.
4. **Pixel → metre** using the references in the jet plane (1 m vertical reference at the
   ground station, marked pole at 5 m, ground tape/pylons every 1 m). Note: landing
   distances go up to ~11 m, beyond the 5 m pole – check how far the references reach.
5. **Empirical fit** (Python, numpy/scipy):
   - fit **all curves jointly**, e.g. `z = z0 + a(θ)·x + b(θ)·x²` with a, b smooth in θ,
     so sparse curves borrow information from neighbouring angles;
   - weighted fit (`curve_fit(..., sigma=...)`), weight 1/σ²;
   - use the **measured nozzle angle** (not the commanded one) as θ; consider the −3° drone tilt.
6. **Uncertainty**
   - Landing point was estimated by eye on site ("where most water landed"):
     σ grows with angle, ~±10 cm (20 cm range) at small angles up to ~±25 cm
     (50 cm range) at 30° where the jet spreads into a wide cone.
   - Clicked points: σ from clicking accuracy converted to metres.
   - Cross-check written landing points against the video.
   - 3 runs per angle → run-to-run spread (servo/mechanism play).
   - Show error bars / uncertainty bands in thesis plots.
7. **Inverse for aiming**: target (distance, height) → nozzle angle, via numerical
   solve or a lookup table that `dist_angle_mapping.py` can read.

## Open points

- Location of colleague's undistortion script and calibration parameters.
- Was iPhone video stabilisation on? If yes, check that references stay fixed in frame.
- Clip ↔ run/angle mapping (24 clips vs. log entries; Messung 3 has an empty 0° entry).
- Exact meaning of the start/end 0° entries in the log (repeatability check).
