"""Kamera-Kalibrierung aus Schachbrett-Videos.

Ablauf:
  1. Jedes n-te Bild aus den Videos lesen.
  2. In jedem Bild das Schachbrett suchen (innere Ecken) und die Schärfe bewerten.
  3. Eine gut verteilte Auswahl scharfer Bilder treffen.
  4. cv2.calibrateCamera rechnen, schlechte Bilder aussortieren, nochmal rechnen.
  5. Ergebnis als JSON speichern und eine Karte der Verzerrung zeichnen.

Aufruf (Beispiel):
  python3 kalibrieren.py video1.mov video2.mov --feld-mm 25 --ausgabe kalibrierung.json

Benötigt: opencv-python, numpy, matplotlib (nur für die Karte).
"""

import argparse
import json

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Bilder aus den Videos lesen
# ---------------------------------------------------------------------------

def bilder_aus_video(pfad, jedes_nte):
    """Liefert (Bildnummer, Graustufenbild) für jedes n-te Bild des Videos."""
    video = cv2.VideoCapture(pfad)
    if not video.isOpened():
        raise RuntimeError(f"Video kann nicht geöffnet werden: {pfad}")
    nummer = 0
    while True:
        ok, bild = video.read()
        if not ok:
            break
        if nummer % jedes_nte == 0:
            yield nummer, cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
        nummer += 1
    video.release()


# ---------------------------------------------------------------------------
# 2. Schachbrett finden und Schärfe bewerten
# ---------------------------------------------------------------------------

def finde_schachbrett(grau, muster):
    """Sucht die inneren Ecken des Schachbretts.

    Die Suche läuft auf einem verkleinerten Bild (schnell), danach werden die
    Ecken im Originalbild auf Subpixel genau verfeinert (genau).
    Gibt ein Array (Anzahl Ecken, 2) zurück oder None.
    """
    skalierung = 0.25
    klein = cv2.resize(grau, None, fx=skalierung, fy=skalierung, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    gefunden, ecken = cv2.findChessboardCorners(klein, muster, flags)
    if not gefunden:
        return None

    ecken = (ecken / skalierung).astype(np.float32)
    kriterium = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    ecken = cv2.cornerSubPix(grau, ecken, (11, 11), (-1, -1), kriterium)
    return ecken.reshape(-1, 2)


def schaerfe(grau, ecken):
    """Schärfemass: Varianz des Laplace-Filters im Bereich des Bretts.

    Verwischte Bilder (Bewegung) haben weiche Kanten und damit einen kleinen Wert.
    """
    x, y, b, h = cv2.boundingRect(ecken.astype(np.int32))
    ausschnitt = grau[max(0, y):y + h, max(0, x):x + b]
    return float(cv2.Laplacian(ausschnitt, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# 3. Bilder auswählen
# ---------------------------------------------------------------------------

def waehle_bilder(funde, bildgroesse, anzahl_abschnitte=80):
    """Wählt eine gut verteilte Menge scharfer Bilder aus.

    a) Zeitlich: Die Funde werden in Abschnitte geteilt, aus jedem Abschnitt
       wird das schärfste Bild genommen (aufeinanderfolgende Bilder sind fast
       gleich und bringen keine neue Information).
    b) Ränder: Zusätzlich die Bilder, in denen das Brett am weitesten Richtung
       Bildrand bzw. Bildecken liegt. Dort ist die Verzerrung am grössten.
    """
    breite, hoehe = bildgroesse
    alle_schaerfen = np.array([f["schaerfe"] for f in funde])
    median_schaerfe = np.median(alle_schaerfen)

    auswahl = {}
    for abschnitt in np.array_split(np.arange(len(funde)), anzahl_abschnitte):
        if len(abschnitt) == 0:
            continue
        bestes = max(abschnitt, key=lambda i: funde[i]["schaerfe"])
        if funde[bestes]["schaerfe"] >= 0.8 * median_schaerfe:
            auswahl[bestes] = funde[bestes]

    richtungen = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
    for dx, dy in richtungen:
        def wie_weit_aussen(f):
            c = f["ecken"]
            return ((c[:, 0] - breite / 2) / breite * dx + (c[:, 1] - hoehe / 2) / hoehe * dy).max()
        kandidaten = [i for i, f in enumerate(funde) if f["schaerfe"] >= 0.5 * median_schaerfe]
        kandidaten.sort(key=lambda i: wie_weit_aussen(funde[i]), reverse=True)
        for i in kandidaten[:4]:
            auswahl[i] = funde[i]

    return list(auswahl.values())


# ---------------------------------------------------------------------------
# 4. Kalibrieren
# ---------------------------------------------------------------------------

def schachbrett_punkte_3d(muster, feld_mm):
    """3D-Koordinaten der inneren Ecken auf dem Brett (z = 0), in mm."""
    spalten, zeilen = muster
    punkte = np.zeros((spalten * zeilen, 3), np.float32)
    punkte[:, :2] = np.mgrid[0:spalten, 0:zeilen].T.reshape(-1, 2) * feld_mm
    return punkte


def kalibriere(auswahl, punkte_3d, bildgroesse):
    """Eine Kalibrierung mit OpenCV. Gibt auch den Fehler pro Bild zurück."""
    objekt = [punkte_3d] * len(auswahl)
    bild = [f["ecken"].reshape(-1, 1, 2) for f in auswahl]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objekt, bild, bildgroesse, None, None)

    fehler_pro_bild = []
    for f, rv, tv in zip(auswahl, rvecs, tvecs):
        projiziert, _ = cv2.projectPoints(punkte_3d, rv, tv, K, D)
        abweichung = projiziert.reshape(-1, 2) - f["ecken"]
        fehler_pro_bild.append(float(np.sqrt(np.mean(np.sum(abweichung ** 2, axis=1)))))
    return rms, K, D, np.array(fehler_pro_bild)


def kalibriere_mit_aussortieren(auswahl, punkte_3d, bildgroesse, min_bilder=15):
    """Kalibriert und wirft schrittweise Ausreisser-Bilder raus.

    Ein Bild gilt als Ausreisser, wenn sein Fehler mehr als 2,5x so gross ist
    wie der Median aller Bilder (z.B. verwischt oder Ecke falsch erkannt).
    """
    auswahl = list(auswahl)
    rms, K, D, fehler = kalibriere(auswahl, punkte_3d, bildgroesse)
    while fehler.max() > 2.5 * np.median(fehler) and len(auswahl) > min_bilder:
        auswahl.pop(int(fehler.argmax()))
        rms, K, D, fehler = kalibriere(auswahl, punkte_3d, bildgroesse)
    return auswahl, rms, K, D, fehler


# ---------------------------------------------------------------------------
# 5. Verzerrung bewerten
# ---------------------------------------------------------------------------

def verschiebung_px(punkte, K, D):
    """Wie weit verschiebt die Verzerrung einen Bildpunkt? (in Pixeln)

    undistortPoints rechnet für einen gemessenen (verzerrten) Pixel aus, wo er
    ohne Verzerrung läge. Die Differenz ist die Verschiebung.
    """
    p = np.asarray(punkte, np.float64).reshape(-1, 1, 2)
    entzerrt = cv2.undistortPoints(p, K, D, P=K).reshape(-1, 2)
    return np.hypot(*(entzerrt - p.reshape(-1, 2)).T)


def zeichne_karte(K, D, bildgroesse, huelle, alle_ecken, pfad):
    """Karte der Verschiebung, nur im Bereich, den das Brett abgedeckt hat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    breite, hoehe = bildgroesse
    xs, ys = np.meshgrid(np.arange(0, breite, 40), np.arange(0, hoehe, 40))
    gitter = np.c_[xs.ravel(), ys.ravel()]
    innen = np.array([cv2.pointPolygonTest(huelle, (float(x), float(y)), False) >= 0 for x, y in gitter])
    werte = verschiebung_px(gitter, K, D)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bild = ax.imshow(np.where(innen, werte, np.nan).reshape(xs.shape),
                     extent=[0, breite, hoehe, 0], cmap="viridis", vmin=0)
    ax.imshow(np.where(innen, np.nan, 1.0).reshape(xs.shape),
              extent=[0, breite, hoehe, 0], cmap="Greys", vmin=0, vmax=3, alpha=0.6)
    ax.scatter(alle_ecken[::4, 0], alle_ecken[::4, 1], s=1, c="w", alpha=0.3)
    h = huelle.reshape(-1, 2)
    ax.plot(np.r_[h[:, 0], h[0, 0]], np.r_[h[:, 1], h[0, 1]], "w-", lw=1)
    ax.set_xlim(0, breite)
    ax.set_ylim(hoehe, 0)
    ax.set_aspect("equal")
    ax.set_title("Verschiebung durch Verzerrung [px] – grau = keine Daten")
    fig.colorbar(bild, ax=ax, fraction=0.03, label="px")
    fig.tight_layout()
    fig.savefig(pfad, dpi=110)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kamera-Kalibrierung aus Schachbrett-Videos")
    parser.add_argument("videos", nargs="+", help="ein oder mehrere Videodateien")
    parser.add_argument("--ecken", type=int, nargs=2, default=[9, 6], metavar=("SPALTEN", "ZEILEN"),
                        help="innere Ecken des Schachbretts (Standard: 9 6)")
    parser.add_argument("--feld-mm", type=float, default=25.0, help="Seitenlänge eines Feldes in mm")
    parser.add_argument("--jedes-nte", type=int, default=3, help="nur jedes n-te Bild verwenden")
    parser.add_argument("--ausgabe", default="kalibrierung.json", help="Ausgabe-JSON")
    parser.add_argument("--karte", default="verzerrung_karte.png", help="Bild der Verzerrungskarte")
    args = parser.parse_args()

    muster = tuple(args.ecken)
    punkte_3d = schachbrett_punkte_3d(muster, args.feld_mm)

    # 1 + 2: Schachbrett in den Videos suchen
    funde = []
    bildgroesse = None
    for video in args.videos:
        anzahl_vorher = len(funde)
        for nummer, grau in bilder_aus_video(video, args.jedes_nte):
            bildgroesse = (grau.shape[1], grau.shape[0])
            ecken = finde_schachbrett(grau, muster)
            if ecken is not None:
                funde.append({"quelle": f"{video}#{nummer}", "ecken": ecken, "schaerfe": schaerfe(grau, ecken)})
        print(f"{video}: Brett in {len(funde) - anzahl_vorher} Bildern gefunden")

    if len(funde) < 15:
        raise SystemExit("Zu wenige Bilder mit Schachbrett gefunden (mind. 15).")

    # 3: Auswahl
    auswahl = waehle_bilder(funde, bildgroesse)
    print(f"Ausgewählt: {len(auswahl)} Bilder")

    # 4: Kalibrieren
    auswahl, rms, K, D, fehler = kalibriere_mit_aussortieren(auswahl, punkte_3d, bildgroesse)
    print(f"Verwendet: {len(auswahl)} Bilder, RMS-Fehler {rms:.3f} px "
          f"(pro Bild: Median {np.median(fehler):.2f}, max {fehler.max():.2f} px)")
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    print("D (k1, k2, p1, p2, k3) =", np.round(D.ravel(), 5))

    # 5: Bewertung im abgedeckten Bereich
    alle_ecken = np.vstack([f["ecken"] for f in auswahl]).astype(np.float32)
    huelle = cv2.convexHull(alle_ecken)
    breite, hoehe = bildgroesse
    for name, (x, y) in {"Bildmitte": (breite / 2, hoehe / 2),
                         "Rand rechts": (breite - 1, hoehe / 2),
                         "Ecke unten rechts": (breite - 1, hoehe - 1)}.items():
        innen = cv2.pointPolygonTest(huelle, (float(x), float(y)), False) >= 0
        hinweis = "" if innen else "  (ausserhalb der Daten, unsicher)"
        print(f"Verschiebung {name}: {verschiebung_px([[x, y]], K, D)[0]:.1f} px{hinweis}")

    ergebnis = {
        "bildgroesse_px": list(bildgroesse),
        "schachbrett": {"innere_ecken": list(muster), "feld_mm": args.feld_mm},
        "rms_reprojektionsfehler_px": round(float(rms), 3),
        "anzahl_bilder": len(auswahl),
        "K": K.round(3).tolist(),
        "D_k1_k2_p1_p2_k3": D.ravel().round(6).tolist(),
        "abgedeckter_bereich_px": huelle.reshape(-1, 2).round(1).tolist(),
        "quellen": args.videos,
    }
    with open(args.ausgabe, "w") as f:
        json.dump(ergebnis, f, indent=2, ensure_ascii=False)
    print(f"Gespeichert: {args.ausgabe}")

    zeichne_karte(K, D, bildgroesse, huelle, alle_ecken, args.karte)
    print(f"Karte gespeichert: {args.karte}")


if __name__ == "__main__":
    main()
