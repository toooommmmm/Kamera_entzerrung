"""Kompositbild aus einem Video: macht den schwachen Wasserstrahl sichtbar.

Idee: Die Kamera steht still, die Szene (Boden, Referenzen, Himmel) ändert sich
kaum, das Wasser bewegt sich ständig. Rechnet man pro Pixel über alle Bilder
eine Statistik, wird die ruhige Szene dunkel und der Strahl hell.

Erzeugte Bilder pro Video (alle in ORIGINAL-Pixelkoordinaten, also verzerrt):
  _std.png         zeitliche Standardabweichung (Standard für den Strahl)
  _hell.png        Maximum minus Hintergrund (Wasser heller als Hintergrund)
  _dunkel.png      Hintergrund minus Minimum (Wasser dunkler, z.B. vor hellem Himmel)
  _hintergrund.png Median-Hintergrund in Farbe (Referenzen gut sichtbar)
  _overlay.png     Hintergrund grau + Strahl (std) farbig darüber -> zum Klicken

Die Bilder werden bewusst NICHT entzerrt (kein Umrechnen = keine Unschärfe).
Gemessene Punkte danach mit entzerren.entzerre_punkte() umrechnen, die
Referenzpunkte (Meterstab, Pfosten, Bodenmarken) genauso.

Zeitfenster: Pro Clip nur die Bilder verwenden, in denen der Strahl konstant
läuft und niemand durchs Bild geht. Die Zeiten stehen in zeitfenster.csv
(clip,start_s,ende_s in Sekunden ab Videoanfang). Leere Felder = Videoanfang/-ende.
--start/--ende auf der Kommandozeile überschreiben die Tabelle.

Aufruf (Beispiele):
  python3 kompositbild.py clip.mov
  python3 kompositbild.py ../ipztest_06_11/iphone_videos/*.mov --ausgabe komposite
  python3 kompositbild.py clip.mov --start 3 --ende 18 --gamma 0.4

Benötigt: opencv-python, numpy.
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Statistik über alle Bilder (ein Durchlauf)
# ---------------------------------------------------------------------------

def statistik(pfad, start_s=0.0, ende_s=None, jedes_nte=1, anzahl_median=31):
    """Liest das Video einmal und rechnet pro Pixel Mittel, Std, Max, Min und Median.

    Gerechnet wird auf dem Graustufenbild. Für den Median werden etwa
    `anzahl_median` gleichmässig verteilte Bilder zwischengespeichert.
    Gibt ein dict mit float32/uint8-Bildern und Infos zurück.
    """
    video = cv2.VideoCapture(pfad)
    if not video.isOpened():
        raise RuntimeError(f"Video kann nicht geöffnet werden: {pfad}")
    fps = video.get(cv2.CAP_PROP_FPS) or 24.0
    gesamt = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    erstes = int(round(start_s * fps))
    letztes = gesamt if ende_s is None else min(gesamt, int(round(ende_s * fps)))
    if erstes >= letztes:
        raise RuntimeError(f"Leerer Zeitbereich ({start_s} s bis {ende_s} s, Video hat {gesamt / fps:.1f} s)")
    if erstes > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, erstes)

    erwartet = (letztes - erstes + jedes_nte - 1) // jedes_nte
    median_schritt = max(1, erwartet // anzahl_median)

    summe = quadrate = maximum = minimum = None
    median_bilder = []
    n = 0
    nummer = erstes
    while nummer < letztes:
        ok, bild = video.read()
        if not ok:
            break
        if (nummer - erstes) % jedes_nte == 0:
            grau = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
            if summe is None:
                # float64, damit Summe der Quadrate über ~1000 Bilder genau bleibt
                summe = np.zeros(grau.shape, np.float64)
                quadrate = np.zeros(grau.shape, np.float64)
                maximum = grau.copy()
                minimum = grau.copy()
            f = grau.astype(np.float64)
            summe += f
            quadrate += f * f
            np.maximum(maximum, grau, out=maximum)
            np.minimum(minimum, grau, out=minimum)
            if n % median_schritt == 0:
                median_bilder.append(bild)  # in Farbe, für den Hintergrund
            n += 1
        nummer += 1
    video.release()

    if n < 2:
        raise RuntimeError(f"Zu wenige Bilder gelesen ({n}): {pfad}")

    mittel = summe / n
    std = np.sqrt(np.maximum(quadrate / n - mittel ** 2, 0.0))
    hintergrund_farbe = median_farbe(median_bilder)
    hintergrund = cv2.cvtColor(hintergrund_farbe, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "std": std.astype(np.float32),
        "hell": np.maximum(maximum.astype(np.float32) - hintergrund, 0),
        "dunkel": np.maximum(hintergrund - minimum.astype(np.float32), 0),
        "hintergrund": hintergrund,
        "hintergrund_farbe": hintergrund_farbe,
        "anzahl_bilder": n,
        "anzahl_median": len(median_bilder),
        "zeit_s": (erstes / fps, nummer / fps),
    }


def median_farbe(bilder, streifen=270):
    """Median pro Pixel und Farbkanal über eine Liste von BGR-Bildern (uint8).

    Wird in waagrechten Streifen gerechnet, damit bei 4K nicht mehrere GB
    Speicher auf einmal nötig sind.
    """
    hoehe = bilder[0].shape[0]
    ergebnis = np.empty_like(bilder[0])
    for y in range(0, hoehe, streifen):
        stapel = np.stack([b[y:y + streifen] for b in bilder])
        ergebnis[y:y + streifen] = np.median(stapel, axis=0).round().astype(np.uint8)
    return ergebnis


# ---------------------------------------------------------------------------
# Darstellung
# ---------------------------------------------------------------------------

def strecken(bild, unten=1.0, oben=99.8, gamma=0.5):
    """Kontrast strecken: Perzentile `unten`..`oben` -> 0..1, dann Gamma.

    Gamma < 1 hebt schwache Werte an (schwacher Strahl wird sichtbarer).
    """
    lo, hi = np.percentile(bild, [unten, oben])
    x = np.clip((bild - lo) / max(hi - lo, 1e-6), 0, 1)
    return x ** gamma


def als_png(x01):
    return (np.clip(x01, 0, 1) * 255).round().astype(np.uint8)


def overlay(hintergrund, strahl01, farbe_bgr=(0, 80, 255), staerke=1.0):
    """Hintergrund grau, Strahl farbig darüber (Deckkraft = Strahlstärke)."""
    grau = cv2.cvtColor(hintergrund.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    alpha = np.clip(strahl01 * staerke, 0, 1)[..., None]
    farbe = np.array(farbe_bgr, np.float32)[None, None, :]
    return (grau * (1 - alpha) + farbe * alpha).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def lade_zeitfenster(pfad):
    """Liest die Tabelle clip,start_s,ende_s[,bemerkung] -> {clip: (start, ende)}.

    Leere Felder werden zu None (= Videoanfang bzw. -ende).
    """
    fenster = {}
    if not pfad or not os.path.exists(pfad):
        return fenster
    with open(pfad, newline="", encoding="utf-8") as f:
        for zeile in csv.DictReader(f):
            clip = (zeile.get("clip") or "").strip()
            if not clip:
                continue
            fenster[clip] = (_zahl(zeile.get("start_s")), _zahl(zeile.get("ende_s")))
    return fenster


def _zahl(wert):
    """'12.5' -> 12.5, leer -> None (Komma als Dezimaltrenner erlaubt)."""
    return float(wert.replace(",", ".")) if wert and wert.strip() else None


def zeitbereich(pfad, args, fenster):
    """Start/Ende für ein Video: Kommandozeile vor Tabelle, sonst ganzes Video."""
    start, ende = fenster.get(os.path.basename(pfad), (None, None))
    if args.start is not None:
        start = args.start
    if args.ende is not None:
        ende = args.ende
    if start is None and ende is None:
        print(f"WARNUNG {os.path.basename(pfad)}: kein Zeitfenster, ganzes Video wird verwendet")
    return start or 0.0, ende


def verarbeite(pfad, ausgabe, args, fenster):
    name = os.path.splitext(os.path.basename(pfad))[0]
    start, ende = zeitbereich(pfad, args, fenster)
    s = statistik(pfad, start, ende, args.jedes_nte, args.anzahl_median)
    t0, t1 = s["zeit_s"]
    print(f"{name}: {s['anzahl_bilder']} Bilder ({t0:.1f}-{t1:.1f} s), "
          f"Median aus {s['anzahl_median']} Bildern")

    ausgabe = os.path.join(ausgabe, name)      # ein Unterordner pro Clip: komposite/<clip>/
    os.makedirs(ausgabe, exist_ok=True)
    gestreckt = {}
    for methode in ("std", "hell", "dunkel"):
        gestreckt[methode] = strecken(s[methode], args.unten, args.oben, args.gamma)
        cv2.imwrite(os.path.join(ausgabe, f"{name}_{methode}.png"), als_png(gestreckt[methode]))
    cv2.imwrite(os.path.join(ausgabe, f"{name}_hintergrund.png"), s["hintergrund_farbe"])
    cv2.imwrite(os.path.join(ausgabe, f"{name}_overlay.png"),
                overlay(s["hintergrund"], gestreckt[args.overlay_methode]))

    if args.roh:
        np.savez_compressed(os.path.join(ausgabe, f"{name}_roh.npz"),
                            std=s["std"], hell=s["hell"], dunkel=s["dunkel"],
                            hintergrund=s["hintergrund_farbe"], anzahl_bilder=s["anzahl_bilder"],
                            zeit_s=np.array(s["zeit_s"]))


def main():
    parser = argparse.ArgumentParser(description="Kompositbild (Std/Max/Min) aus Videos, macht den Strahl sichtbar")
    parser.add_argument("videos", nargs="+", help="Videodateien (Platzhalter wie *.mov erlaubt)")
    parser.add_argument("--ausgabe", default="komposite", help="Ausgabeordner (Standard: komposite)")
    parser.add_argument("--zeitfenster", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               "zeitfenster.csv"),
                        help="Tabelle clip,start_s,ende_s (Standard: zeitfenster.csv neben dem Skript)")
    parser.add_argument("--start", type=float, default=None, help="erst ab dieser Zeit in s (überschreibt Tabelle)")
    parser.add_argument("--ende", type=float, default=None, help="nur bis zu dieser Zeit in s (überschreibt Tabelle)")
    parser.add_argument("--jedes-nte", type=int, default=1, help="nur jedes n-te Bild verwenden")
    parser.add_argument("--anzahl-median", type=int, default=31, help="Bilder für den Median-Hintergrund")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma der Darstellung (<1 hebt Schwaches an)")
    parser.add_argument("--unten", type=float, default=1.0, help="unteres Perzentil für die Streckung")
    parser.add_argument("--oben", type=float, default=99.8, help="oberes Perzentil für die Streckung")
    parser.add_argument("--overlay-methode", choices=["std", "hell", "dunkel"], default="std",
                        help="welches Komposit im Overlay gezeigt wird")
    parser.add_argument("--roh", action="store_true", help="zusätzlich Rohwerte als .npz speichern")
    args = parser.parse_args()

    pfade = []
    for muster in args.videos:  # Windows-Shells expandieren *.mov nicht selbst
        treffer = sorted(glob.glob(muster))
        pfade += treffer if treffer else [muster]

    fenster = lade_zeitfenster(args.zeitfenster)
    os.makedirs(args.ausgabe, exist_ok=True)
    for pfad in pfade:
        try:
            verarbeite(pfad, args.ausgabe, args, fenster)
        except RuntimeError as fehler:
            print(f"FEHLER {pfad}: {fehler}")


if __name__ == "__main__":
    main()
