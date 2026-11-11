"""Schritt 4: entzerrte Pixel -> Meter in der Strahlebene.

Referenzen im Bild:
  - Bodenlinie: eine gerade Linie entlang des Bodens in der Strahlebene (z = 0).
    Der Boden ist uneben, darum gilt diese eine Gerade als gemeinsamer Boden.
  - Pfosten: stehen bei bekanntem horizontalem Abstand x von der Düse. Auf jedem
    Pfosten sind Marken mit genau 1 m senkrechtem Abstand, aber keine Marke hat
    eine bekannte absolute Höhe.

Optional: Bodenmarken (Klebeband, Pylonen ...) bei bekanntem x auf dem Boden.
Der Klick wird senkrecht (im Bild) auf die Bodenlinie projiziert und gilt als
(x, 0). Sie legen den Massstab entlang des Bodens fest, auch über den letzten
Pfosten hinaus.

Pro Pfosten wird die Achse (Gerade durch die Marken) mit der Bodenlinie
geschnitten -> Bodenpunkt (x_p, 0). Die Marken liegen bei (x_p, h_p + k * 1 m),
wobei k = 0 die unterste geklickte Marke ist und h_p (Höhe der untersten Marke
über dem Boden) unbekannt ist.

Gesucht ist eine Homographie H (Ebene -> Ebene) von entzerrten Pixeln nach
(x, z) in Metern. H und alle h_p werden gemeinsam mit kleinsten Quadraten
bestimmt (scipy.optimize.least_squares). Das gilt nur für Punkte in der
Ebene der Pfosten, also in der Strahlebene.

Alle Pixel sind ENTZERRTE Pixel (Kameramatrix K der Kalibrierung).
"""

import cv2
import numpy as np
from scipy.optimize import least_squares

MARKEN_ABSTAND_M = 1.0

# Pixel werden intern normiert (Mitte abziehen, durch 1000 teilen), damit die
# Homographie numerisch gut konditioniert ist.
_NORM = np.array([[1 / 1000, 0, -1920 / 1000],
                  [0, 1 / 1000, -1080 / 1000],
                  [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometrie im Bild
# ---------------------------------------------------------------------------

def gerade_fit(punkte):
    """Gerade durch >= 2 Punkte (kleinste senkrechte Abstände). -> (Punkt, Richtung)."""
    p = np.asarray(punkte, np.float64).reshape(-1, 2)
    mitte = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - mitte)
    return mitte, vt[0]


def schnittpunkt(gerade1, gerade2):
    """Schnittpunkt zweier Geraden (Punkt, Richtung) oder None, falls parallel."""
    (p1, d1), (p2, d2) = gerade1, gerade2
    A = np.column_stack([d1, -d2])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, np.asarray(p2) - np.asarray(p1))
    return np.asarray(p1) + t[0] * np.asarray(d1)


def sortiere_marken(marken):
    """Marken von unten nach oben sortieren (grösstes Bild-y zuerst). Index = k."""
    m = np.asarray(marken, np.float64).reshape(-1, 2)
    return m[np.argsort(-m[:, 1])]


# ---------------------------------------------------------------------------
# Ebene (Homographie + Pfosten-Offsets)
# ---------------------------------------------------------------------------

def _h_aus_param(p8):
    return np.append(p8, 1.0).reshape(3, 3)


def _abbilden(H, punkte_norm):
    q = np.c_[punkte_norm, np.ones(len(punkte_norm))] @ H.T
    return q[:, :2] / q[:, 2:3]


def _normieren(punkte_px):
    p = np.asarray(punkte_px, np.float64).reshape(-1, 2)
    return p * _NORM[0, 0] + _NORM[:2, 2]


class Ebene:
    """Ergebnis von Schritt 4: Abbildung entzerrte Pixel -> (x, z) in Metern."""

    def __init__(self, param, kovarianz, n_pfosten, pfosten_namen, residuen, bodenpunkte):
        self.param = param                      # 8 Homographie-Parameter + h_p je Pfosten
        self.kovarianz = kovarianz              # Kovarianz von param (oder None)
        self.H_norm = _h_aus_param(param[:8])
        self.H = self.H_norm @ _NORM            # direkt auf Pixel anwendbar
        self.offsets = dict(zip(pfosten_namen, param[8:8 + n_pfosten]))
        self.residuen = residuen                # Liste von dicts (Pfosten, Art, k, dx, dz in m)
        self.bodenpunkte = bodenpunkte          # {Pfosten: (u, v)}

    def nach_metern(self, punkte_px):
        """(N, 2) entzerrte Pixel -> (N, 2) Meter (x, z)."""
        return _abbilden(self.H_norm, _normieren(punkte_px))

    def rms_cm(self):
        r = np.array([[e["dx_m"], e["dz_m"]] for e in self.residuen])
        return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))) * 100 if len(r) else float("nan")

    def max_cm(self):
        r = np.array([np.hypot(e["dx_m"], e["dz_m"]) for e in self.residuen])
        return float(r.max()) * 100 if len(r) else float("nan")

    def jacobi_pixel(self, punkt_px, schritt=0.5):
        """2x2-Matrix d(x, z)/d(u, v): wie stark ein Klickfehler in Metern wirkt."""
        u, v = np.asarray(punkt_px, np.float64).ravel()
        J = np.empty((2, 2))
        for i, (du, dv) in enumerate([(schritt, 0), (0, schritt)]):
            a = self.nach_metern([[u + du, v + dv]])[0]
            b = self.nach_metern([[u - du, v - dv]])[0]
            J[:, i] = (a - b) / (2 * schritt)
        return J

    def kovarianz_abbildung(self, punkt_px):
        """2x2-Kovarianz von (x, z) eines Punktes durch die Unsicherheit von H."""
        if self.kovarianz is None:
            return np.full((2, 2), np.nan)
        pn = _normieren(punkt_px)
        J = np.empty((2, 8))
        for i in range(8):
            h = 1e-6 * max(1.0, abs(self.param[i]))
            pp, pm = self.param[:8].copy(), self.param[:8].copy()
            pp[i] += h
            pm[i] -= h
            J[:, i] = (_abbilden(_h_aus_param(pp), pn)[0] - _abbilden(_h_aus_param(pm), pn)[0]) / (2 * h)
        return J @ self.kovarianz[:8, :8] @ J.T


def projiziere_auf_gerade(punkt, gerade):
    """Fusspunkt des Lots von punkt auf die Gerade (Punkt, Richtung)."""
    p, d = gerade
    d = np.asarray(d) / np.linalg.norm(d)
    return np.asarray(p) + np.dot(np.asarray(punkt, np.float64) - p, d) * d


def bestimme_ebene(boden_punkte, pfosten, bodenmarken=()):
    """Bestimmt die Abbildung aus Bodenlinie, Pfosten und optionalen Bodenmarken.

    boden_punkte: >= 2 Punkte (u, v) entlang des Bodens.
    pfosten: Liste von dicts {"name", "x_m", "marken": [(u, v), ...]} mit >= 2 Marken.
    bodenmarken: Liste von (u, v, x_m): Marke auf dem Boden bei bekanntem x (z = 0).
    Gibt eine Ebene zurück oder wirft ValueError mit einer verständlichen Meldung.
    """
    if len(boden_punkte) < 2:
        raise ValueError("Bodenlinie fehlt (mind. 2 Punkte)")
    pfosten = [p for p in pfosten if p.get("x_m") is not None and len(p["marken"]) >= 2]
    if len(pfosten) < 2:
        raise ValueError("mind. 2 Pfosten mit je >= 2 Marken nötig")
    boden = gerade_fit(boden_punkte)

    # Korrespondenzen: (Pfosten-Index, Pixel, x_m, k oder None für z = 0, Art)
    eintraege = []
    bodenpunkte = {}
    start_offsets = []
    for i, p in enumerate(pfosten):
        marken = sortiere_marken(p["marken"])
        fuss = schnittpunkt(gerade_fit(marken), boden)
        if fuss is None:
            raise ValueError(f"Pfosten {p['name']}: Achse parallel zur Bodenlinie")
        bodenpunkte[p["name"]] = tuple(fuss)
        eintraege.append((i, fuss, p["x_m"], None, "boden"))
        for k, m in enumerate(marken):
            eintraege.append((i, m, p["x_m"], k, "marke"))
        # Startwert h_p: Abstand Boden -> unterste Marke in "Markenabständen"
        abstand_marken = np.mean(np.linalg.norm(np.diff(marken, axis=0), axis=1))
        start_offsets.append(np.linalg.norm(marken[0] - fuss) / abstand_marken * MARKEN_ABSTAND_M)

    for u, v, x_m in bodenmarken:
        eintraege.append((0, projiziere_auf_gerade((u, v), boden), float(x_m), None, "bodenmarke"))

    n_p = len(pfosten)
    idx = np.array([e[0] for e in eintraege])
    pix_norm = _normieren(np.array([e[1] for e in eintraege]))
    x_soll = np.array([e[2] for e in eintraege], np.float64)
    k_arr = np.array([np.nan if e[3] is None else e[3] for e in eintraege])
    ist_marke = ~np.isnan(k_arr)

    def ziel(offsets):
        z = np.zeros(len(eintraege))
        z[ist_marke] = offsets[idx[ist_marke]] + k_arr[ist_marke] * MARKEN_ABSTAND_M
        return np.c_[x_soll, z]

    # Startwert H aus einer linearen Schätzung (DLT) mit den Start-Offsets
    H0, _ = cv2.findHomography(pix_norm, ziel(np.array(start_offsets)), 0)
    if H0 is None:
        raise ValueError("Homographie nicht bestimmbar (Punkte zu ungünstig verteilt?)")
    H0 = H0 / H0[2, 2]
    p0 = np.r_[H0.ravel()[:8], start_offsets]

    def residuen(p):
        return (_abbilden(_h_aus_param(p[:8]), pix_norm) - ziel(p[8:])).ravel()

    erg = least_squares(residuen, p0, method="lm" if 2 * len(eintraege) >= len(p0) else "trf")
    r = erg.fun.reshape(-1, 2)
    freiheitsgrade = r.size - len(p0)
    kov = None
    if freiheitsgrade > 0:
        s2 = float(np.sum(r ** 2) / freiheitsgrade)
        try:
            kov = np.linalg.inv(erg.jac.T @ erg.jac) * s2
        except np.linalg.LinAlgError:
            kov = None

    namen = [p["name"] for p in pfosten]
    liste = [{"pfosten": None if e[4] == "bodenmarke" else namen[e[0]], "art": e[4], "x_m": e[2],
              "k": e[3], "dx_m": float(dr[0]), "dz_m": float(dr[1])}
             for e, dr in zip(eintraege, r)]
    return Ebene(erg.x, kov, n_p, namen, liste, bodenpunkte)


# ---------------------------------------------------------------------------
# Aus gespeicherten Klickpunkten
# ---------------------------------------------------------------------------

def aus_klickpunkten(punkte, referenzen):
    """Ebene aus der Punktliste des Klickfensters.

    punkte: Liste von dicts {"typ", "x", "y", ["pfosten"], ["x_m"]} (entzerrte Pixel).
    referenzen: Liste von dicts {"name", "x_m"} (aus referenzen.csv).
    Wirft ValueError, wenn die Referenzen nicht reichen.
    """
    boden = [(p["x"], p["y"]) for p in punkte if p["typ"] == "boden"]
    pfosten = []
    for ref in referenzen:
        marken = [(p["x"], p["y"]) for p in punkte
                  if p["typ"] == "pfosten_marke" and p.get("pfosten") == ref["name"]]
        pfosten.append({"name": ref["name"], "x_m": ref["x_m"], "marken": marken})
    bodenmarken = [(p["x"], p["y"], p["x_m"]) for p in punkte
                   if p["typ"] == "bodenmarke" and p.get("x_m") is not None]
    return bestimme_ebene(boden, pfosten, bodenmarken)
