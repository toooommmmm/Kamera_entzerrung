# Kamera-Entzerrung (Blackmagic Camera App)

Werkzeuge, um die Linsenverzerrung der Handykamera (iPhone, Blackmagic Camera App) zu bestimmen und Bilder, Videos oder einzelne gemessene Punkte zu entzerren. Gedacht für die Bildauswertung der Wasserstrahl-Messungen (z.B. Auftreffpunkte auf 10 m).

**Kurzfassung:** Für Messungen mit *Stabilisierung aus* die Datei `kalibrierungen/blackmagic_4k_stab_aus.json` verwenden und gemessene Punkte mit `entzerren.py punkte …` korrigieren. Ohne Korrektur liegt der Fehler bei 10 m im Bereich von 2–7 cm, mit Korrektur bei etwa 2 mm.

## Inhalt

| Datei | Zweck |
|---|---|
| `entzerren.py` | Punkte, Bilder oder Videos mit einer Kalibrierung entzerren |
| `kalibrieren.py` | Neue Kalibrierung aus Schachbrett-Videos erstellen |
| `kalibrierungen/blackmagic_4k_stab_aus.json` | **Kalibrierung für die Messungen** (Stabilisierung aus) |
| `kalibrierungen/blackmagic_4k_stab_standard.json` | Kalibrierung für Stabilisierung „Standard“ (nur falls so gefilmt wurde) |
| `kalibrierungen/verzerrung_karte_*.png` | Wie stark die Verzerrung jeden Bildpunkt verschiebt |
| `schachbrett_9x6_25mm_A4.pdf` | Druckvorlage für eine neue Kalibrierung |

## Installation

```bash
pip install -r requirements.txt     # opencv-python, numpy, matplotlib
```

## Welche Kalibrierung?

Eine Kalibrierung gilt nur für **genau die Kameraeinstellungen**, mit denen sie gemacht wurde:

| | `stab_aus` (Messungen) | `stab_standard` |
|---|---|---|
| App | Blackmagic Camera | Blackmagic Camera |
| Auflösung / Bildrate | 3840×2160, 24 fps | 3840×2160, 24 fps |
| Objektiv | 26 mm (1×), kein Zoom | 26 mm (1×), kein Zoom |
| Bildstabilisierung | **Aus** | **Standard** |
| Brennweite fx | 2864 px | 3084 px |
| Reprojektionsfehler | 0,63 px | 0,64 px |

Die Stabilisierung schneidet das Bild zu (ca. 7 %), darum sind die beiden **nicht austauschbar**. Egal für die Geometrie sind Verschlusszeit, ISO, Weissabgleich und LUT. Der Fokus (auch Auto) ändert die Brennweite um weniger als 1 %. Während einer Aufnahme den Fokus sperren (lange auf das Motiv tippen).

**Prüfen, mit welchen Einstellungen ein Video gefilmt wurde:** In der Blackmagic-App unter Einstellungen → Kamera. Die Auflösung muss 3840×2160 sein, sonst bricht `entzerren.py` mit einer Meldung ab.

## Benutzung: entzerren

### Einzelne Punkte (Normalfall für Messungen)

Punkte im **Originalvideo** messen (Pixelkoordinaten im 3840×2160-Bild), dann nur diese Punkte umrechnen:

```bash
python3 entzerren.py punkte kalibrierungen/blackmagic_4k_stab_aus.json  3000 1500  1920 1080
```

Ausgabe pro Punkt: entzerrte Pixelkoordinate und Winkel von der optischen Achse (rechts/unten). Liegt ein Punkt ausserhalb des kalibrierten Bereichs, kommt eine Warnung.

In eigenem Python-Code:

```python
from entzerren import lade_kalibrierung, entzerre_punkte
K, D, groesse, bereich = lade_kalibrierung("kalibrierungen/blackmagic_4k_stab_aus.json")
entzerrt = entzerre_punkte([[3000, 1500], [1920, 1080]], K, D)   # -> Array (N, 2) in Pixeln
```

### Ganzes Bild oder Video (zum Anschauen)

```bash
python3 entzerren.py bild  kalibrierungen/blackmagic_4k_stab_aus.json  frame.png  frame_entzerrt.png
python3 entzerren.py video kalibrierungen/blackmagic_4k_stab_aus.json  clip.mov   clip_entzerrt.mp4
```

Am Rand entstehen schwarze, leicht gekrümmte Bereiche, das ist normal.

**Einzelbild aus dem Video holen** (nicht per Screenshot vom Handy, der hat eine andere Auflösung):

```bash
ffmpeg -ss 00:00:12 -i clip.mov -frames:v 1 frame.png      # Bild bei 12 s, volle Auflösung
```

## Wie gross ist der Fehler ohne Korrektur?

Bei 10 m Abstand entspricht 1 px etwa 3,5 mm (Stabilisierung aus).

| Bereich im Bild | Verschiebung | Fehler bei 10 m |
|---|---|---|
| Bildmitte | 0 px | 0 |
| mittlerer Bereich (Median) | 6–7,5 px | ca. 2–2,6 cm |
| fast ganzer kalibrierter Bereich | bis 15–18 px | ca. 5–6 cm |
| Rand des kalibrierten Bereichs | bis 19–23 px | ca. 7–8 cm |
| linker/rechter Bildrand, Ecken | – | **nicht kalibriert, meiden** |

Mit Korrektur bleibt ein Restfehler von ca. 0,6 px, also etwa 2 mm bei 10 m. Siehe `kalibrierungen/verzerrung_karte_stab_aus.png` (grau = keine Kalibrierdaten).

## Wichtig für Messungen in Metern

Entzerren macht das Bild geometrisch korrekt (gerade Linien bleiben gerade), aber ein einzelnes Bild hat **keine Tiefe**. Strecken in Metern stimmen nur in einer Ebene, und nur wenn ein Massstab bekannt ist:

1. Kamera auf Stativ, möglichst **senkrecht zur Ebene**, in der der Strahl fliegt (seitlich).
2. **Massstab in diese Ebene** stellen (Meterstab oder zwei Marken mit bekanntem Abstand) → m/px.
3. Zur Kontrolle eine zweite bekannte Strecke an einer anderen Bildstelle: gleich lang nach dem Entzerren → Kamera steht senkrecht genug.
4. Robuster: **vier Marken** mit bekannten Positionen in der Ebene (z.B. Ecken eines Rechtecks). Damit lässt sich auch eine schräge Ansicht herausrechnen (Homographie, `cv2.findHomography`).

Objekte vor oder hinter der Messebene haben einen anderen Massstab.

## Auswertung Wasserstrahl (Kompositbild → Klicken → Meter → Fit)

Ablauf für die Strahl-Videos (Plan: `waterjet_trajectory_plan.md`, Schritte 2–6):

| Datei | Zweck |
|---|---|
| `zeitfenster.csv` | pro Clip Start/Ende (s), in denen der Strahl konstant läuft und niemand durchs Bild geht |
| `kompositbild.py` | Schritt 2: Kompositbilder pro Clip → `komposite/<clip>/` (Std, Hell, Dunkel, Hintergrund in Farbe, Overlay) |
| `referenzen.csv` | Pfosten und ihr horizontaler Abstand `x_m` von der Düse (selbst ausfüllen) |
| `messungen.csv` | pro Clip: Lauf, gemessener Düsenwinkel, Aufprall laut Protokoll (selbst ausfüllen) |
| `strahl.py` | Schritte 3–6 (Klickfenster, Meter, Fit, Plots) |
| `klickfenster.py`, `ebene.py`, `modell.py`, `daten.py` | Bausteine von `strahl.py` |
| `punkte/<clip>.json` | Klickpunkte (entzerrte Pixel) – Handarbeit, wird mit versioniert |

```bash
python kompositbild.py "../ipztest_06_11/iphone_videos/*.mov"   # Schritt 2
python strahl.py klicken 0_a                                     # Schritt 3 (+4 live)
python strahl.py metrisch                                        # Schritt 4 für alle Clips
python strahl.py fit                                             # Schritte 5+6 → auswertung/
```

**Klickfenster:** Die Kompositbilder werden beim Laden entzerrt, geklickt wird im entzerrten Bild (gestrichelt = kalibrierter Bereich). Zoomen/Verschieben mit der Werkzeugleiste; solange Lupe oder Hand aktiv ist, zählen Klicks nicht.

| Taste | Bedeutung |
|---|---|
| `d` | Düsenspitze (1 Punkt) |
| `s` | Strahl-Mittellinie (beliebig viele Punkte) |
| `a` | Aufprall im Video (1 Punkt, Vergleich mit Protokoll) |
| `g` | Bodenlinie: ≥ 2 Punkte entlang des Bodens, gerade Linie = z = 0 |
| `r` | Pfosten-Marken: Pfosten aus `referenzen.csv` der Reihe nach, je ≥ 2 aufeinanderfolgende 1-m-Marken; `Enter` nächster, `n` überspringen |
| `1` `2` `3` `4` | Bild Overlay / Std / Hintergrund / Hell |
| `m` | Bodenmarke (Klebeband/Pylone) auf dem Boden: klicken, dann x [m] (Abstand von der Düse) eingeben; wird auf die Bodenlinie projiziert = (x, 0) |
| `z`, rechte Maustaste | letzten Punkt löschen |
| `x`, `Entf` | Punkt unter der Maus löschen (beliebiger Punkt, nicht nur der letzte) |
| `w` | speichern (auch automatisch beim Schliessen) |

Sobald Bodenlinie und ≥ 2 Pfosten geklickt sind, zeigt das Fenster Meter an (unten in der Statuszeile für die Mausposition, an den Strahlpunkten, und die Residuen der Referenzen).

**Pixel → Meter (Schritt 4):** Pfostenachse ∩ Bodenlinie = Bodenpunkt (x_p, 0); Marken = (x_p, h_p + k·1 m) mit unbekannter Höhe h_p der untersten Marke. Optional Bodenmarken = (x, 0), legen den Massstab entlang des Bodens fest (auch jenseits des letzten Pfostens). Homographie und alle h_p werden gemeinsam gefittet. Gilt nur in der Strahlebene (Pfosten müssen in dieser Ebene stehen).

**Fit (Schritte 5+6):** z(x, θ) = Σ c_j(θ)·x^j mit c_j polynomial in θ, alle Clips gemeinsam, gewichtet (Klick-σ über die Abbildung in Meter, beim Aufprall zusätzlich σ 0,10–0,25 m je nach Winkel für den Sprühkegel).

- θ = gemessener Düsenwinkel aus `messungen.csv`, **absolut** (Weltkoordinaten); die Drohnenneigung ist schon enthalten, darum kein Offset.
- Aufprall: Standardmässig zählt der **im Bild geklickte** Aufprall (`a`). Der Protokollwert wird nur zum Vergleich gezeigt (vor Ort schwer zu schätzen, Lauf 1 von einer anderen Person) und kommt nur mit `--mit-protokoll-aufprall` in den Fit. Das Klickfenster zeigt den Protokollwert bewusst nicht an.

Optionen: `python strahl.py fit --help` (z.B. `--grad-x 3`, `--mit-protokoll-aufprall`, global `--sigma-klick 3`).

### Geometriemodell (`geometrie.py`)

Erklärung und Begründung der Modellwahl (D2 gewählt, A3 als Vergleich): `strahlmodell_D2_A3.md`.

```bash
python strahl.py fit --modell geometrie           # → auswertung/*_geometrie.*, fit_modell_geometrie.npz/.json
python strahl.py vergleich                        # Modelle A/B/C/C1/C2, leave-one-run-out → vergleich.txt/.png
python strahl.py winkel 8 0                       # Ziel 8 m vor / 0 m über dem Servo-Drehpunkt → θ ± σ_θ
python strahl.py winkel 9 0 --bezug boden         # Ziel in Versuchskoordinaten
```

Die Kurve beginnt an der Düse und hat dort die Richtung der Düse:

- **Düse:** Spitze = Servo-Drehpunkt + Arm, `x_d = x_p + L cos θ − h sin θ`, `z_d = z_p + L sin θ + h cos θ`, aus den geklickten Düsenpunkten aller Clips gefittet (mittelt den Klickfehler heraus; `--ursprung klick` nimmt stattdessen die Düse jedes Clips).
- **Strahl:** `Δz = tan(θ+δ)·Δx + sec²θ·a(t)·Δx² + sec³θ·b(t)·Δx³`, `Δx = x − x_d`. δ = Abweichung der Strahlrichtung vom gemessenen Düsenwinkel (ein Wert für alle Winkel). sec²θ/sec³θ: Abhängigkeit vom Winkel bei konstanter Pumpe (Wurf mit Luftwiderstand, führende Ordnung). Der x³-Term beschreibt die Asymmetrie durch den Luftwiderstand (absteigender Ast steiler). `--grad-x 2` = Parabel, `--grad-theta 1/2` = zusätzliche Korrektur in t = θ/30°.
- **Streuung von Schuss zu Schuss:** pro Clip ein zusätzlicher Winkel ε_k (σ_ε) und eine zusätzliche Krümmung α_k (σ_α), dazu ein Faktor f auf die Klick-σ. Alle drei werden aus den Daten geschätzt (REML). Das Modell gibt darum zwei Unsicherheiten: σ der mittleren Kurve (Band) und σ eines neuen Schusses (gepunktet), die für das Zielen zählt.
- **Umkehrung** (`winkel`, `GeometrieModell.winkel_fuer_ziel`): nur im gemessenen Winkelbereich und zwischen Düse und Landepunkt, sonst „nicht erreichbar“ (keine Extrapolation). σ_θ = σ_z(neuer Schuss) / |∂z/∂θ|. θ ist absolut; für die Drohne noch θ_rel = θ − Neigung (IMU) und die Servo-Kalibrierung.
- **Winkeloffset δ(θ):** `--grad-delta 2` erlaubt δ(θ) = δ_0 + δ_1·t + δ_2·t² statt eines festen δ (Abweichung zwischen gemessenem Winkel und Strahlrichtung, abhängig vom Winkel).
- **Vergleich:** A = freie Parabel (Basis), A3/A3' = freies Polynom bis x³ (Koeffizienten Grad 2 bzw. 1 in θ), B = Geometrie-Parabel, C = Geometrie kubisch, C1/C2 = C mit Korrektur in t bzw. t, t², D = C mit δ(θ), D2 = C2 mit δ(θ) (`fit --modell geometrie --grad-theta 2 --grad-delta 2`). Leave-one-run-out: Fit auf zwei Läufen, Vorhersage des dritten (Fehler des Landepunkts in cm und Winkelfehler in Grad). Dazu eine Tabelle zum Strahlbeginn: RMS der Punkte bis 1,5 m nach der Düse und Strahlrichtung an der Düse − θ (zeigt, ob ein Modell an der Düse physikalisch plausibel bleibt).

Für `dist_angle_mapping.py`: `auswertung/fit_modell_geometrie.json` enthält alle Parameter und die Formel; `geometrie.lade_geometrie(pfad).winkel_fuer_ziel(x, z)` löst pro Ziel.

## So funktioniert die Kalibrierung

**Kameramodell (OpenCV):** Kameramatrix `K` mit Brennweite `fx, fy` und Bildmitte `cx, cy`, dazu Verzerrung `D = (k1, k2, p1, p2, k3)`. Radial (k): Punkte werden mit wachsendem Abstand von der Bildmitte nach aussen/innen geschoben. Tangential (p): Linse nicht ganz parallel zum Sensor.

**Ablauf in `kalibrieren.py`:**

1. **Bilder lesen** – jedes 3. Bild aus dem Video (`cv2.VideoCapture`).
2. **Schachbrett finden** – `cv2.findChessboardCorners` (auf verkleinertem Bild, schnell), dann `cv2.cornerSubPix` im Originalbild (Subpixel-genau). Schärfe über die Varianz des Laplace-Filters, um verwischte Bilder auszusortieren.
3. **Auswahl** – pro Zeitabschnitt das schärfste Bild, plus die Bilder mit dem Brett am weitesten am Rand.
4. **Kalibrieren** – `cv2.calibrateCamera` passt `K` und `D` so an, dass die bekannten Brettecken (25 mm Raster) möglichst genau auf die gemessenen Pixel abgebildet werden. Bilder mit über 2,5× dem typischen Fehler werden aussortiert.
5. **Bewerten** – `cv2.undistortPoints` liefert für jeden Bildpunkt die Verschiebung; Karte und JSON werden gespeichert.

**Entzerren:** `cv2.undistortPoints` für einzelne Punkte; für ganze Bilder einmal `cv2.initUndistortRectifyMap`, dann `cv2.remap` pro Bild.

**Gütemass:** Der RMS-Reprojektionsfehler sagt, wie gut das Modell zu den Kalibrierbildern passt (unter ~1 px ist gut). Er sagt nicht, ob man korrigieren muss.

## Neue Kalibrierung erstellen

Nötig, wenn sich eine Einstellung aus der Tabelle oben ändert (anderes Handy, andere Auflösung, Stabilisierung, Zoom …).

1. `schachbrett_9x6_25mm_A4.pdf` mit **100 % / tatsächlicher Grösse** drucken, den 100-mm-Massstab nachmessen, flach auf eine steife Unterlage kleben.
2. Mit **genau den Messeinstellungen** 30–60 s filmen: Brett langsam bewegen, verschiedene Neigungen (bis ~45°) und Abstände, auch **an den Bildrändern und in den Ecken**.
3. Ausführen:

   ```bash
   python3 kalibrieren.py video.mov --feld-mm 25 --ausgabe kalibrierungen/neu.json --karte kalibrierungen/neu_karte.png
   ```

   Optionen: `--ecken 9 6` (innere Ecken), `--jedes-nte 3`. Dauer für ein 4K-Video: ca. 1–2 Minuten.
