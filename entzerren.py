"""Bilder, Videos oder einzelne Punkte mit einer Kalibrierung entzerren.

Aufruf (Beispiele):
  python3 entzerren.py bild    kalibrierung.json foto.jpg       foto_entzerrt.jpg
  python3 entzerren.py video   kalibrierung.json clip.mov       clip_entzerrt.mp4
  python3 entzerren.py punkte  kalibrierung.json 3500 1200  100 200

Die Kalibrierung muss mit denselben Kamera-Einstellungen gemacht worden sein
(gleiche App, Auflösung, Linse, kein Zoom, gleiche Stabilisierung).

Benötigt: opencv-python, numpy.
"""

import argparse
import json

import cv2
import numpy as np


def lade_kalibrierung(pfad):
    """Liest K, D, Bildgrösse und den abgedeckten Bereich aus der JSON-Datei."""
    with open(pfad) as f:
        daten = json.load(f)
    K = np.array(daten["K"], dtype=np.float64)
    D = np.array(daten["D_k1_k2_p1_p2_k3"], dtype=np.float64)
    groesse = tuple(daten["bildgroesse_px"])
    huelle = daten.get("abgedeckter_bereich_px")
    huelle = None if huelle is None else np.array(huelle, np.float32).reshape(-1, 1, 2)
    return K, D, groesse, huelle


def pruefe_groesse(bild_breite, bild_hoehe, groesse):
    if (bild_breite, bild_hoehe) != tuple(groesse):
        raise SystemExit(f"Bildgrösse {bild_breite}x{bild_hoehe} passt nicht zur Kalibrierung "
                         f"{groesse[0]}x{groesse[1]}. Andere Einstellungen = andere Kalibrierung.")


# ---------------------------------------------------------------------------
# Punkte entzerren (z.B. gemessene Auftreffpunkte)
# ---------------------------------------------------------------------------

def entzerre_punkte(punkte_px, K, D):
    """Verzerrte Pixel -> entzerrte Pixel (gleiche Kameramatrix K).

    Das ist der Normalfall für Messungen: Man misst Punkte im Originalvideo
    und rechnet nur diese Punkte um, statt das ganze Video zu entzerren.
    """
    p = np.asarray(punkte_px, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, D, P=K).reshape(-1, 2)


def punkte_zu_richtungen(punkte_px, K, D):
    """Verzerrte Pixel -> Blickrichtung als (x/z, y/z), Kamera-Konvention.

    Damit kann man Winkel ausrechnen: atan(x/z) = Winkel nach rechts,
    atan(y/z) = Winkel nach unten, jeweils von der optischen Achse aus.
    """
    p = np.asarray(punkte_px, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, D).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Bilder und Videos entzerren
# ---------------------------------------------------------------------------

def entzerrungs_tabellen(K, D, groesse):
    """Berechnet einmal, woher jeder Pixel des entzerrten Bildes kommt.

    Für ein Video ist das viel schneller als cv2.undistort pro Bild, weil die
    Rechnung nur einmal gemacht und dann mit cv2.remap angewendet wird.
    Die Kameramatrix bleibt K, damit Pixelkoordinaten vergleichbar bleiben.
    Am Rand entstehen dadurch schwarze Bereiche bzw. etwas wird abgeschnitten.
    """
    return cv2.initUndistortRectifyMap(K, D, None, K, groesse, cv2.CV_16SC2)


def entzerre_bild(eingabe, ausgabe, K, D, groesse):
    bild = cv2.imread(eingabe)
    if bild is None:
        raise SystemExit(f"Bild kann nicht gelesen werden: {eingabe}")
    pruefe_groesse(bild.shape[1], bild.shape[0], groesse)
    karte1, karte2 = entzerrungs_tabellen(K, D, groesse)
    cv2.imwrite(ausgabe, cv2.remap(bild, karte1, karte2, cv2.INTER_LINEAR))
    print(f"Gespeichert: {ausgabe}")


def entzerre_video(eingabe, ausgabe, K, D, groesse):
    video = cv2.VideoCapture(eingabe)
    if not video.isOpened():
        raise SystemExit(f"Video kann nicht geöffnet werden: {eingabe}")
    breite = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    hoehe = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    pruefe_groesse(breite, hoehe, groesse)

    karte1, karte2 = entzerrungs_tabellen(K, D, groesse)
    schreiber = cv2.VideoWriter(ausgabe, cv2.VideoWriter_fourcc(*"mp4v"), fps, (breite, hoehe))
    anzahl = 0
    while True:
        ok, bild = video.read()
        if not ok:
            break
        schreiber.write(cv2.remap(bild, karte1, karte2, cv2.INTER_LINEAR))
        anzahl += 1
    video.release()
    schreiber.release()
    print(f"{anzahl} Bilder entzerrt, gespeichert: {ausgabe}")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entzerren mit einer Kamera-Kalibrierung")
    unter = parser.add_subparsers(dest="modus", required=True)

    p_bild = unter.add_parser("bild", help="ein Bild entzerren")
    p_bild.add_argument("kalibrierung")
    p_bild.add_argument("eingabe")
    p_bild.add_argument("ausgabe")

    p_video = unter.add_parser("video", help="ein Video entzerren")
    p_video.add_argument("kalibrierung")
    p_video.add_argument("eingabe")
    p_video.add_argument("ausgabe")

    p_punkte = unter.add_parser("punkte", help="Pixelkoordinaten entzerren: x1 y1 x2 y2 ...")
    p_punkte.add_argument("kalibrierung")
    p_punkte.add_argument("koordinaten", type=float, nargs="+")

    args = parser.parse_args()
    K, D, groesse, huelle = lade_kalibrierung(args.kalibrierung)

    if args.modus == "bild":
        entzerre_bild(args.eingabe, args.ausgabe, K, D, groesse)
    elif args.modus == "video":
        entzerre_video(args.eingabe, args.ausgabe, K, D, groesse)
    else:
        if len(args.koordinaten) % 2 != 0:
            raise SystemExit("Koordinaten paarweise angeben: x1 y1 x2 y2 ...")
        punkte = np.array(args.koordinaten).reshape(-1, 2)
        entzerrt = entzerre_punkte(punkte, K, D)
        richtung = punkte_zu_richtungen(punkte, K, D)
        for (x, y), (ex, ey), (rx, ry) in zip(punkte, entzerrt, richtung):
            innen = huelle is None or cv2.pointPolygonTest(huelle, (float(x), float(y)), False) >= 0
            hinweis = "" if innen else "   ACHTUNG: ausserhalb des kalibrierten Bereichs"
            print(f"({x:.1f}, {y:.1f}) -> ({ex:.1f}, {ey:.1f}) px, "
                  f"Winkel rechts {np.degrees(np.arctan(rx)):+.2f}°, unten {np.degrees(np.arctan(ry)):+.2f}°{hinweis}")


if __name__ == "__main__":
    main()
