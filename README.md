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
