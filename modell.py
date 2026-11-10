"""Schritte 5 + 6: gemeinsamer empirischer Fit aller Strahlkurven und Unsicherheiten.

Modell (Höhe z über dem Boden, horizontaler Abstand x von der Düse, Düsenwinkel θ):

    z(x, θ) = Σ_j c_j(θ) · x^j        j = 0 .. grad_x        (Standard: Parabel)
    c_j(θ)  = Σ_i p_ij · t^i          i = 0 .. grad_theta,  t = θ / 30°

Alle Kurven werden gemeinsam gefittet: Winkel mit wenigen Punkten profitieren
von den Nachbarwinkeln. Das Modell ist linear in den Parametern p_ij, darum
gewichtete lineare kleinste Quadrate (wie curve_fit mit sigma=..., aber exakt).

θ ist der gemessene Düsenwinkel aus messungen.csv: absolut (Weltkoordinaten,
gegen die Horizontale), die Drohnenneigung ist darin schon enthalten.

Aufprallpunkt: Standardmässig zählt der im Bild geklickte Aufprall (Taste a).
Der Wert aus dem Protokoll wird nur zum Vergleich gezeigt (vor Ort schwer zu
schätzen, im ersten Lauf von einer anderen Person) und kommt nur auf Wunsch
in den Fit.

Gewichte (Schritt 6):
  - geklickte Punkte: Klickfehler in Pixeln -> Meter über die Abbildung aus
    Schritt 4 (Jacobi-Matrix), plus Unsicherheit der Abbildung selbst
  - Aufprall (geklickt oder Protokoll): zusätzlich σ horizontal für den
    breiten Sprühkegel, wächst mit dem Winkel (Standard 0,10 m bei 0° bis
    0,25 m bei 30°)
  - Fehler in x werden über die Kurvensteigung in z umgerechnet:
    σ_eff² = σ_z² + (dz/dx)² σ_x²  (zweiter Durchgang mit Steigung aus dem ersten)
"""

import csv
import os

import numpy as np

import daten
import ebene

SKALA_GRAD = 30.0


# ---------------------------------------------------------------------------
# Daten in Metern
# ---------------------------------------------------------------------------

def sigma_aufprall(winkel, s0=0.10, s30=0.25):
    """σ des Aufprallpunkts aus dem Protokoll [m], linear im Winkel."""
    return float(np.interp(winkel, [0.0, 30.0], [s0, s30]))


def clip_in_metern(d, referenzen, sigma_klick_px):
    """Klickdaten eines Clips -> (Ebene, Liste der Strahlpunkte in Metern mit σ)."""
    e = ebene.aus_klickpunkten(d["punkte"], referenzen)
    punkte = []
    for p in d["punkte"]:
        if p["typ"] not in ("duese", "strahl", "aufprall"):
            continue
        u = [p["x"], p["y"]]
        xm, zm = e.nach_metern([u])[0]
        J = e.jacobi_pixel(u)
        C = J @ J.T * sigma_klick_px ** 2
        Ca = e.kovarianz_abbildung(u)
        if np.all(np.isfinite(Ca)):
            C = C + Ca
        punkte.append({"typ": p["typ"], "u": p["x"], "v": p["y"], "x": float(xm), "z": float(zm),
                       "sx": float(np.sqrt(C[0, 0])), "sz": float(np.sqrt(C[1, 1])),
                       "im_kalibrierten_bereich": p.get("im_kalibrierten_bereich", True)})
    return e, punkte


def sammle_clips(punkte_ordner, referenzen, messungen, sigma_klick_px, winkel_offset=0.0,
                 s0=0.10, s30=0.25):
    """Alle Klickdateien mit gemessenem Winkel -> Liste von Clip-dicts für den Fit."""
    clips, warnungen = [], []
    for pfad in daten.alle_punktdateien(punkte_ordner):
        d = daten.lade_punkte(pfad)
        clip = d["clip"]
        m = messungen.get(clip)
        if m is None or m["winkel"] is None:
            warnungen.append(f"{clip}: kein gemessener Winkel in messungen.csv -> nicht im Fit")
            continue
        try:
            e, punkte = clip_in_metern(d, referenzen, sigma_klick_px)
        except ValueError as fehler:
            warnungen.append(f"{clip}: Referenzen unvollständig ({fehler}) -> nicht im Fit")
            continue
        theta = m["winkel"] + winkel_offset
        clips.append({"clip": clip, "winkel_soll": d.get("winkel_soll"), "messung": m["messung"],
                      "theta": theta, "aufprall_log": m["aufprall_m"],
                      "sigma_aufprall": sigma_aufprall(theta, s0, s30),
                      "punkte": punkte, "ebene_rms_cm": e.rms_cm()})
    return clips, warnungen


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------

class Modell:
    def __init__(self, grad_x, grad_theta, param, kovarianz, chi2_red, freiheitsgrade):
        self.grad_x, self.grad_theta = grad_x, grad_theta
        self.param = param
        self.kovarianz_roh = kovarianz
        self.chi2_red = chi2_red
        self.freiheitsgrade = freiheitsgrade
        # Wenn die Streuung grösser ist als die angenommenen σ, Kovarianz hochskalieren
        self.kovarianz = kovarianz * max(1.0, chi2_red)

    def zeilen(self, x, theta):
        """Designmatrix: Spalten t^i x^j (i schnell, j langsam)."""
        x = np.atleast_1d(np.asarray(x, np.float64))
        t = np.broadcast_to(np.asarray(theta, np.float64) / SKALA_GRAD, x.shape)
        return np.column_stack([t ** i * x ** j for j in range(self.grad_x + 1)
                                for i in range(self.grad_theta + 1)])

    def koeffizienten(self, theta):
        """c_j(θ) für j = 0 .. grad_x."""
        t = theta / SKALA_GRAD
        p = self.param.reshape(self.grad_x + 1, self.grad_theta + 1)
        return p @ (t ** np.arange(self.grad_theta + 1))

    def z(self, x, theta):
        return self.zeilen(x, theta) @ self.param

    def sigma_z(self, x, theta):
        g = self.zeilen(x, theta)
        return np.sqrt(np.einsum("ij,jk,ik->i", g, self.kovarianz, g))

    def steigung(self, x, theta):
        c = self.koeffizienten(theta)
        x = np.atleast_1d(np.asarray(x, np.float64))
        return sum(j * c[j] * x ** (j - 1) for j in range(1, self.grad_x + 1))

    def aufprall(self, theta, x_max=40.0):
        """Kleinste positive Nullstelle von z(x) (Landepunkt) und ihr σ."""
        c = self.koeffizienten(theta)
        wurzeln = np.roots(c[::-1])
        kandidaten = [w.real for w in wurzeln if abs(w.imag) < 1e-9 and 0 < w.real < x_max
                      and self.steigung(w.real, theta)[0] < 0]
        if not kandidaten:
            return np.nan, np.nan
        xl = min(kandidaten)
        s = abs(self.steigung(xl, theta)[0])
        return xl, float(self.sigma_z(xl, theta)[0] / s)


def _wls(A, b, sigma):
    Aw, bw = A / sigma[:, None], b / sigma
    param, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    kov = np.linalg.inv(Aw.T @ Aw)
    r = Aw @ param - bw
    fg = len(b) - len(param)
    return param, kov, float(r @ r / fg) if fg > 0 else np.nan, fg


def fit(clips, grad_x=2, grad_theta=2, mit_log_aufprall=False, mit_video_aufprall=True):
    """Gemeinsamer Fit. -> (Modell, Liste der verwendeten Datenpunkte mit Residuen).

    Standard: geklickter Aufprall im Fit, Protokoll-Aufprall nur zum Vergleich.
    """
    zeilen = []   # dicts: clip, theta, art, x, z, sx, sz
    for c in clips:
        for p in c["punkte"]:
            if p["typ"] in ("duese", "strahl") or (mit_video_aufprall and p["typ"] == "aufprall"):
                sx = p["sx"]
                if p["typ"] == "aufprall":      # Sprühkegel: Landepunkt nur ungenau bestimmbar
                    sx = float(np.hypot(sx, c["sigma_aufprall"]))
                zeilen.append({"clip": c["clip"], "winkel_soll": c["winkel_soll"], "theta": c["theta"],
                               "art": p["typ"], "x": p["x"], "z": p["z"], "sx": sx, "sz": p["sz"]})
        if mit_log_aufprall and c["aufprall_log"] is not None:
            zeilen.append({"clip": c["clip"], "winkel_soll": c["winkel_soll"], "theta": c["theta"],
                           "art": "aufprall_log", "x": c["aufprall_log"], "z": 0.0,
                           "sx": c["sigma_aufprall"], "sz": 0.0})
    if not zeilen:
        raise ValueError("keine Datenpunkte für den Fit")

    winkel = {round(z["theta"], 3) for z in zeilen}
    grad_theta = min(grad_theta, len(winkel) - 1)
    x = np.array([z["x"] for z in zeilen])
    th = np.array([z["theta"] for z in zeilen])
    zz = np.array([z["z"] for z in zeilen])
    sx = np.array([z["sx"] for z in zeilen])
    sz = np.array([z["sz"] for z in zeilen])

    vorlage = Modell(grad_x, grad_theta, np.zeros((grad_x + 1) * (grad_theta + 1)), np.eye(1), 1.0, 0)
    A = np.vstack([vorlage.zeilen(xi, ti) for xi, ti in zip(x, th)])
    if len(zz) <= A.shape[1]:
        raise ValueError(f"zu wenige Punkte ({len(zz)}) für {A.shape[1]} Parameter")

    # 1. Durchgang: Fehler in x grob als Fehler in z (Steigung ~1)
    sigma = np.sqrt(sz ** 2 + sx ** 2)
    param, kov, chi2, fg = _wls(A, zz, sigma)
    m1 = Modell(grad_x, grad_theta, param, kov, chi2, fg)
    # 2. Durchgang: x-Fehler über die echte Steigung
    steig = np.array([m1.steigung(xi, ti)[0] for xi, ti in zip(x, th)])
    sigma = np.sqrt(sz ** 2 + (steig * sx) ** 2)
    sigma = np.maximum(sigma, 1e-4)
    param, kov, chi2, fg = _wls(A, zz, sigma)
    modell = Modell(grad_x, grad_theta, param, kov, chi2, fg)

    for z, s, mz in zip(zeilen, sigma, A @ param):
        z["sigma_eff"] = float(s)
        z["residuum"] = float(z["z"] - mz)
    return modell, zeilen


# ---------------------------------------------------------------------------
# Auswertung: Text, Tabellen, Plots
# ---------------------------------------------------------------------------

def zusammenfassung(modell, clips, zeilen):
    t = []
    t.append(f"Modell: Polynom Grad {modell.grad_x} in x, Koeffizienten Grad {modell.grad_theta} in θ "
             f"(t = θ/{SKALA_GRAD:g}°)")
    t.append(f"Punkte: {len(zeilen)}, Parameter: {len(modell.param)}, Freiheitsgrade: {modell.freiheitsgrade}")
    t.append(f"chi²_red = {modell.chi2_red:.2f}"
             + ("  (> 1: Streuung grösser als angenommen, Unsicherheiten entsprechend hochskaliert)"
                if modell.chi2_red > 1 else ""))
    t.append("")
    t.append("Parameter p_ij (Koeffizient von t^i x^j):")
    s = np.sqrt(np.diag(modell.kovarianz))
    k = 0
    for j in range(modell.grad_x + 1):
        for i in range(modell.grad_theta + 1):
            t.append(f"  p_{i}{j} = {modell.param[k]:+.5f} ± {s[k]:.5f}")
            k += 1

    t.append("")
    im_fit = {z["art"] for z in zeilen}
    quellen = [n for a, n in (("aufprall", "geklickt (Bild)"), ("aufprall_log", "Protokoll")) if a in im_fit]
    t.append("Aufprall im Fit: " + (", ".join(quellen) if quellen else "keiner"))
    t.append("")
    t.append("Pro Clip:  θ [°]   Aufprall Bild [m]   Protokoll [m]   Modell [m]       "
             "RMS Strahl [cm]   Referenzen RMS [cm]")
    for c in sorted(clips, key=lambda c: (c["theta"], c["clip"])):
        xl, sl = modell.aufprall(c["theta"])
        res = [z["residuum"] for z in zeilen if z["clip"] == c["clip"] and z["art"] in ("duese", "strahl")]
        rms = np.sqrt(np.mean(np.square(res))) * 100 if res else np.nan
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        bild = "     -      " if not video else f"{video[0]['x']:5.2f} ± {np.hypot(video[0]['sx'], c['sigma_aufprall']):.2f}"
        log = "  -  " if c["aufprall_log"] is None else f"{c['aufprall_log']:5.2f}"
        t.append(f"  {c['clip']:<8} {c['theta']:6.1f}   {bild:>15}     {log:>8}       {xl:6.2f} ± {sl:.2f}"
                 f"      {rms:6.1f}            {c['ebene_rms_cm']:5.1f}")

    t.append("")
    t.append("Streuung zwischen den Läufen (gleicher Soll-Winkel):")
    gruppen = {}
    for c in clips:
        gruppen.setdefault(c["winkel_soll"], []).append(c)
    for soll in sorted(gruppen, key=lambda w: (w is None, w)):
        g = gruppen[soll]
        th = np.array([c["theta"] for c in g])
        log = np.array([c["aufprall_log"] for c in g if c["aufprall_log"] is not None])
        text = f"  Soll {soll}°: n = {len(g)}, θ gemessen {th.mean():.1f} (± {th.std(ddof=1) if len(th) > 1 else 0:.1f})°"
        if len(log) > 1:
            text += f", Aufprall Protokoll {log.mean():.2f} ± {log.std(ddof=1):.2f} m (Std der Läufe)"
        t.append(text)
    return "\n".join(t)


def speichere_tabelle(zeilen, pfad):
    felder = ["clip", "winkel_soll", "theta", "art", "x", "z", "sx", "sz", "sigma_eff", "residuum"]
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=felder, extrasaction="ignore")
        w.writeheader()
        for z in zeilen:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in z.items()})


def plots(modell, clips, zeilen, ordner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ordner, exist_ok=True)
    sollwinkel = sorted({c["winkel_soll"] for c in clips}, key=lambda w: (w is None, w))
    farben = {w: plt.cm.viridis(i / max(1, len(sollwinkel) - 1)) for i, w in enumerate(sollwinkel)}

    # 1. Bahnen
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for c in clips:
        f = farben[c["winkel_soll"]]
        pts = [z for z in zeilen if z["clip"] == c["clip"] and z["art"] in ("duese", "strahl")]
        if pts:
            ax.errorbar([p["x"] for p in pts], [p["z"] for p in pts],
                        xerr=[p["sx"] for p in pts], yerr=[p["sz"] for p in pts],
                        fmt="o", ms=3, color=f, elinewidth=0.8, capsize=0)
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        if video:
            ax.errorbar([video[0]["x"]], [video[0]["z"]], xerr=[np.hypot(video[0]["sx"], c["sigma_aufprall"])],
                        fmt="v", ms=6, color=f, mec="k", mew=0.5, capsize=3)
        if c["aufprall_log"] is not None:   # nur Vergleich (ausser --mit-protokoll-aufprall)
            ax.plot([c["aufprall_log"]], [0], "|", ms=12, mew=1.5, color=f)
        pts = pts + [z for z in zeilen if z["clip"] == c["clip"] and z["art"] == "aufprall"]
        xl, _ = modell.aufprall(c["theta"])
        x_ende = xl if np.isfinite(xl) else max([p["x"] for p in pts] + [1.0])
        xs = np.linspace(0, x_ende, 200)
        zs, ss = modell.z(xs, c["theta"]), modell.sigma_z(xs, c["theta"])
        ax.plot(xs, zs, "-", color=f, lw=1)
        ax.fill_between(xs, zs - ss, zs + ss, color=f, alpha=0.15, lw=0)
    for w in sollwinkel:
        ax.plot([], [], "-o", color=farben[w], ms=3, label=f"Soll {w}°")
    ax.plot([], [], "v", color="gray", mec="k", label="Aufprall geklickt (Bild)")
    ax.plot([], [], "|", ms=12, mew=1.5, color="gray", label="Aufprall Protokoll (Vergleich)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("horizontaler Abstand x [m]")
    ax.set_ylabel("Höhe über Boden z [m]")
    ax.set_title("Strahl-Mittellinie: Messpunkte und gemeinsamer Fit (Band = ±1σ)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for endung in ("png", "pdf"):
        fig.savefig(os.path.join(ordner, f"bahnen.{endung}"), dpi=150)
    plt.close(fig)

    # 2. Residuen
    fig, ax = plt.subplots(figsize=(10, 4))
    for z in zeilen:
        marker = "v" if z["art"] == "aufprall_log" else "o"
        ax.errorbar(z["x"], z["residuum"] * 100, yerr=z["sigma_eff"] * 100, fmt=marker, ms=3,
                    color=farben[z["winkel_soll"]], elinewidth=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Residuum z - Modell [cm]")
    ax.set_title("Residuen (Fehlerbalken = σ_eff)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for endung in ("png", "pdf"):
        fig.savefig(os.path.join(ordner, f"residuen.{endung}"), dpi=150)
    plt.close(fig)

    # 3. Aufprallweite über dem Winkel
    fig, ax = plt.subplots(figsize=(7, 4.5))
    th_alle = [c["theta"] for c in clips]
    ths = np.linspace(min(th_alle), max(th_alle), 100)
    xl = np.array([modell.aufprall(t) for t in ths])
    ax.plot(ths, xl[:, 0], "-", color="C0", label="Modell")
    ax.fill_between(ths, xl[:, 0] - xl[:, 1], xl[:, 0] + xl[:, 1], color="C0", alpha=0.2, lw=0)
    for c in clips:
        if c["aufprall_log"] is not None:
            ax.plot(c["theta"], c["aufprall_log"], "s", mfc="none", color=farben[c["winkel_soll"]])
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        if video:
            ax.errorbar(c["theta"], video[0]["x"], yerr=np.hypot(video[0]["sx"], c["sigma_aufprall"]),
                        fmt="v", color=farben[c["winkel_soll"]], mec="k", mew=0.5, capsize=3)
    ax.plot([], [], "v", color="gray", mec="k", label="geklickt (Bild)")
    ax.plot([], [], "s", mfc="none", color="gray", label="Protokoll (Vergleich)")
    ax.set_xlabel("Düsenwinkel θ, absolut [°]")
    ax.set_ylabel("Aufprallweite [m]")
    ax.set_title("Aufprallweite (Band = ±1σ Modell)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for endung in ("png", "pdf"):
        fig.savefig(os.path.join(ordner, f"aufprall.{endung}"), dpi=150)
    plt.close(fig)
