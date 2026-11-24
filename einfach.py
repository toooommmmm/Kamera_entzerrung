"""Einfaches Strahlmodell E: kubisches Polynom ab der Düse, Startrichtung = Düsenrichtung.

Düse (L, h aus der Drohnengeometrie, nur der Drehpunkt (x_p, z_p) im Versuch wird gefittet):

    x_d(θ) = x_p + L·cos θ − h·sin θ
    z_d(θ) = z_p + L·sin θ + h·cos θ
    L = Drehpunkt -> Düsenaustritt entlang der Düsenachse, h = Versatz quer dazu
    (h > 0: bei θ = 0 liegt der Austritt über dem Drehpunkt)

Strahl relativ zur Düse, Δx = x − x_d(θ), Δz = z − z_d(θ), t = θ/30°:

    Δz = (tan θ + d(t))·Δx + c2(t)·Δx² + c3(t)·Δx³

    z(x_d) = z_d          (kein konstanter Term)
    z'(x_d) = tan θ + d   (Ableitung an der Düse = Düsenrichtung, d optional)
    c2, c3, d Polynome in t. Alles linear -> ein gewichteter Kleinste-Quadrate-Fit.

Streuung von Schuss zu Schuss, nachträglich aus den Residuen r zur mittleren Kurve:

    r² ≈ C + s_ε²·Δx² + s_α²·Δx⁴      (nicht-negative kleinste Quadrate, iterativ mit
                                        1/vorhergesagte Varianz gewichtet)
    C = Klickrauschen, s_ε = Winkelstreuung [rad], s_α = Krümmungsstreuung [1/m]
    σ_Schuss² = σ_Kurve² + s_ε²·Δx² + s_α²·Δx⁴
"""

import json

import numpy as np
from scipy.optimize import nnls

import modell as basis

SKALA_GRAD = basis.SKALA_GRAD


# ---------------------------------------------------------------------------
# Düse
# ---------------------------------------------------------------------------

def duese_mit_massen(clips, L, h):
    """L, h fest -> (x_p, z_p, L, h), Residuen der geklickten Düse pro Clip [m]."""
    namen, rx, rz, wx, wz = [], [], [], [], []
    for c in clips:
        d = [p for p in c["punkte"] if p["typ"] == "duese"]
        if not d:
            continue
        d, th = d[0], np.radians(c["theta"])
        namen.append(c["clip"])
        rx.append(d["x"] - L * np.cos(th) + h * np.sin(th))
        rz.append(d["z"] - L * np.sin(th) - h * np.cos(th))
        wx.append(1.0 / d["sx"] ** 2)
        wz.append(1.0 / d["sz"] ** 2)
    if not namen:
        raise ValueError("kein Clip mit geklickter Düse, Drehpunkt nicht bestimmbar")
    rx, rz = np.array(rx), np.array(rz)
    xp, zp = np.average(rx, weights=wx), np.average(rz, weights=wz)
    return np.array([xp, zp, L, h]), dict(zip(namen, np.column_stack([rx - xp, rz - zp])))


def duese_fit_L(clips, h=0.0):
    """h fest, x_p, z_p, L aus den geklickten Düsenspitzen (linear, gewichtete kleinste Quadrate).

    x + h·sin θ = x_p + L·cos θ,   z − h·cos θ = z_p + L·sin θ
    -> (x_p, z_p, L, h), Residuen pro Clip [m]."""
    namen, A, b, w = [], [], [], []
    for c in clips:
        d = [p for p in c["punkte"] if p["typ"] == "duese"]
        if not d:
            continue
        d, th = d[0], np.radians(c["theta"])
        namen.append((c["clip"], th, d["x"], d["z"]))
        A += [[1.0, 0.0, np.cos(th)], [0.0, 1.0, np.sin(th)]]
        b += [d["x"] + h * np.sin(th), d["z"] - h * np.cos(th)]
        w += [1.0 / d["sx"], 1.0 / d["sz"]]
    if len(namen) < 2:
        raise ValueError("zu wenige Clips mit geklickter Düse, Drehpunkt und L nicht bestimmbar")
    A, b, w = np.array(A), np.array(b), np.array(w)
    (xp, zp, L), *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    p = np.array([xp, zp, L, h])
    res = {}
    for name, th, x, z in namen:
        xd, zd = duese_position(p, np.degrees(th))
        res[name] = np.array([x - xd, z - zd])
    return p, res


def duese_position(p, theta):
    th = np.radians(theta)
    return (p[0] + p[2] * np.cos(th) - p[3] * np.sin(th),
            p[1] + p[2] * np.sin(th) + p[3] * np.cos(th))


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------

class EinfachModell:
    """Gleiche Schnittstelle wie geometrie.GeometrieModell (z, steigung, sigma_z, sigma_schuss,
    aufprall, x_start), damit Vergleich und Plots unverändert funktionieren."""

    def __init__(self, grad_theta, grad_d, param, kovarianz, duese, s_eps=0.0, s_alpha=0.0, c_klick=0.0,
                 chi2_red=1.0, theta_bereich=(-90.0, 90.0), quelle_duese=""):
        self.grad_theta, self.grad_d = int(grad_theta), int(grad_d)   # grad_d = -1: kein d
        self.param = np.asarray(param, np.float64)
        self.kovarianz = np.asarray(kovarianz, np.float64)
        self.duese_param = np.asarray(duese, np.float64)
        self.s_eps, self.s_alpha, self.c_klick = float(s_eps), float(s_alpha), float(c_klick)
        self.chi2_red = float(chi2_red)
        self.theta_bereich = tuple(float(v) for v in theta_bereich)
        self.quelle_duese = quelle_duese

    def _nd(self):
        return self.grad_d + 1

    def _n(self):
        return self.grad_theta + 1

    def duese(self, theta):
        return duese_position(self.duese_param, theta)

    def x_start(self, theta):
        return float(self.duese(theta)[0])

    def koeffizienten(self, theta):
        """(k1, k2, k3) mit Δz = k1·Δx + k2·Δx² + k3·Δx³."""
        th = np.asarray(theta, np.float64)
        t = th / SKALA_GRAD
        pot = lambda n: np.stack([t ** i for i in range(n)]) if n else np.zeros((0,) + t.shape)
        nd, n = self._nd(), self._n()
        d = np.tensordot(self.param[:nd], pot(nd), 1) if nd else 0.0 * t
        k2 = np.tensordot(self.param[nd:nd + n], pot(n), 1)
        k3 = np.tensordot(self.param[nd + n:nd + 2 * n], pot(n), 1)
        return np.tan(np.radians(th)) + d, k2, k3

    def _teile(self, x, theta):
        x = np.atleast_1d(np.asarray(x, np.float64))
        th = np.broadcast_to(np.asarray(theta, np.float64), x.shape)
        xd, zd = self.duese(th)
        return th, zd, x - xd

    def z(self, x, theta):
        th, zd, dx = self._teile(x, theta)
        k1, k2, k3 = self.koeffizienten(th)
        return zd + k1 * dx + k2 * dx ** 2 + k3 * dx ** 3

    def steigung(self, x, theta):
        th, zd, dx = self._teile(x, theta)
        k1, k2, k3 = self.koeffizienten(th)
        return k1 + 2 * k2 * dx + 3 * k3 * dx ** 2

    def gradient(self, x, theta):
        th, zd, dx = self._teile(x, theta)
        return designmatrix(th, dx, self.grad_theta, self.grad_d)

    def sigma_z(self, x, theta):
        g = self.gradient(x, theta)
        return np.sqrt(np.einsum("ij,jk,ik->i", g, self.kovarianz, g))

    def sigma_schuss(self, x, theta):
        th, zd, dx = self._teile(x, theta)
        return np.sqrt(self.sigma_z(x, theta) ** 2 + self.s_eps ** 2 * dx ** 2 + self.s_alpha ** 2 * dx ** 4)

    def aufprall(self, theta, schuss=False):
        theta = float(theta)
        xd, zd = self.duese(theta)
        k1, k2, k3 = (float(k) for k in self.koeffizienten(theta))
        kandidaten = [w.real for w in np.roots([k3, k2, k1, float(zd)])
                      if abs(w.imag) < 1e-9 and 0 < w.real < 40 and self.steigung(xd + w.real, theta)[0] < 0]
        if not kandidaten:
            return np.nan, np.nan
        xl = float(xd + min(kandidaten))
        s = abs(self.steigung(xl, theta)[0])
        sig = self.sigma_schuss(xl, theta) if schuss else self.sigma_z(xl, theta)
        return xl, float(sig[0] / s)

    def als_dict(self):
        return {"art": "einfach", "grad_theta": self.grad_theta, "grad_d": self.grad_d,
                "param": self.param.tolist(), "kovarianz": self.kovarianz.tolist(),
                "duese": self.duese_param.tolist(), "s_eps_rad": self.s_eps, "s_alpha": self.s_alpha,
                "theta_bereich_grad": list(self.theta_bereich), "skala_grad": SKALA_GRAD,
                "formel": ("x_d = x_p + L cos θ − h sin θ, z_d = z_p + L sin θ + h cos θ, duese = [x_p, z_p, L, h]; "
                           "Δx = x − x_d; z = z_d + (tan θ + Σ d_i t^i)·Δx + Σ a_i t^i·Δx² + Σ b_i t^i·Δx³; "
                           "t = θ/skala_grad; param = [d_0..d_gd, a_0..a_g, b_0..b_g]; θ absolut in Grad")}


def designmatrix(th, dx, grad_theta, grad_d):
    t = np.asarray(th, np.float64) / SKALA_GRAD
    spalten = [dx * t ** i for i in range(grad_d + 1)]
    spalten += [dx ** 2 * t ** i for i in range(grad_theta + 1)]
    spalten += [dx ** 3 * t ** i for i in range(grad_theta + 1)]
    return np.column_stack(spalten)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _streuung(dx, r, iterationen=50):
    """r² ≈ C + s_ε²·Δx² + s_α²·Δx⁴, iterativ gewichtete NNLS -> (√C, s_ε, s_α).

    r ~ N(0, v): E[r²] = v, Var[r²] = 2v². Jede Zeile wird daher mit 1/v (vorhergesagte Varianz
    aus der letzten Iteration) gewichtet, sonst bestimmen die wenigen weiten Punkte den ganzen Fit."""
    A = np.column_stack([np.ones_like(dx), dx ** 2, dx ** 4])
    v = np.full_like(r, np.mean(r ** 2))
    for _ in range(iterationen):
        koef, _ = nnls(A / v[:, None], r ** 2 / v)
        v_neu = np.maximum(A @ koef, 1e-6)   # Untergrenze 1 mm²
        if np.allclose(v_neu, v, rtol=1e-4):
            break
        v = v_neu
    return np.sqrt(koef)


def fit_einfach(clips, grad_theta=2, grad_d=-1, duese_masse=None, mit_video_aufprall=True,
                mit_log_aufprall=False):
    """-> (EinfachModell, Datenzeilen mit Residuen, Info-dict).

    duese_masse = (L, h) [m] aus der Geometrie; (None, h): L aus den Klicks, h fest;
    None: L, h beide aus den Klicks (4 Parameter)."""
    import geometrie
    if duese_masse is None:
        duese_p, duese_res = geometrie.duese_fit(clips)
        quelle = "Klicks (x_p, z_p, L, h gefittet)"
    elif duese_masse[0] is None:
        duese_p, duese_res = duese_fit_L(clips, duese_masse[1])
        quelle = f"h = {duese_masse[1]:g} m fest, x_p, z_p, L aus den Klicks"
    else:
        duese_p, duese_res = duese_mit_massen(clips, *duese_masse)
        quelle = "L, h vorgegeben, Drehpunkt gefittet"

    zeilen = []
    for k, c in enumerate(clips):
        xd, zd = duese_position(duese_p, c["theta"])
        for p in c["punkte"]:
            if p["typ"] == "strahl" or (mit_video_aufprall and p["typ"] == "aufprall"):
                sx = float(np.hypot(p["sx"], c["sigma_aufprall"])) if p["typ"] == "aufprall" else p["sx"]
                zeilen.append({"clip": c["clip"], "winkel_soll": c["winkel_soll"], "theta": c["theta"],
                               "art": p["typ"], "x": p["x"], "z": p["z"], "sx": sx, "sz": p["sz"],
                               "xd": float(xd), "zd": float(zd)})
        if mit_log_aufprall and c["aufprall_log"] is not None:
            zeilen.append({"clip": c["clip"], "winkel_soll": c["winkel_soll"], "theta": c["theta"],
                           "art": "aufprall_log", "x": c["aufprall_log"], "z": 0.0,
                           "sx": c["sigma_aufprall"], "sz": 0.0, "xd": float(xd), "zd": float(zd)})
    th = np.array([z["theta"] for z in zeilen])
    x = np.array([z["x"] for z in zeilen])
    dx = x - np.array([z["xd"] for z in zeilen])
    dz = np.array([z["z"] for z in zeilen]) - np.array([z["zd"] for z in zeilen])
    sx = np.array([z["sx"] for z in zeilen])
    sz = np.array([z["sz"] for z in zeilen])
    A = designmatrix(th, dx, grad_theta, grad_d)
    y = dz - np.tan(np.radians(th)) * dx
    if len(y) <= A.shape[1]:
        raise ValueError(f"zu wenige Punkte ({len(y)}) für {A.shape[1]} Parameter")
    bereich = (float(th.min()), float(th.max()))

    # 2 Durchgänge: x-Klickfehler über die Steigung in z umrechnen (zuerst Steigung der Düse)
    steig = np.tan(np.radians(th))
    for _ in range(2):
        s = np.maximum(np.sqrt(sz ** 2 + (steig * sx) ** 2), 1e-4)
        param, kov, chi2, fg = basis._wls(A, y, s)
        m = EinfachModell(grad_theta, grad_d, param, kov * max(1.0, chi2), duese_p, chi2_red=chi2,
                          theta_bereich=bereich, quelle_duese=quelle)
        steig = m.steigung(x, th)

    # Streuung von Schuss zu Schuss aus den Residuen
    r = y - A @ param
    m.c_klick, m.s_eps, m.s_alpha = _streuung(dx, r)

    for i, zl in enumerate(zeilen):
        zl["sigma_eff"] = float(s[i] * np.sqrt(max(1.0, chi2)))
        zl["residuum"] = float(r[i])
    return m, zeilen, {"duese_res": duese_res, "quelle_duese": quelle}


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def duese_vergleich(clips, duese_masse):
    """Text: L, h aus den Klicks gegen die vorgegebenen Masse (Kontrolle der Klick->Meter-Umrechnung)."""
    import geometrie
    p, res = geometrie.duese_fit(clips)
    r = np.array(list(res.values()))
    t = [f"Düse frei aus den Klicks: x_p = {p[0]:+.3f} m, z_p = {p[1]:.3f} m, L = {p[2]:.3f} m, h = {p[3]:+.3f} m "
         f"(Residuen RMS x {np.sqrt(np.mean(r[:, 0] ** 2)) * 100:.1f} cm, z {np.sqrt(np.mean(r[:, 1] ** 2)) * 100:.1f} cm)"]
    if duese_masse is not None and duese_masse[0] is None:
        q, res2 = duese_fit_L(clips, duese_masse[1])
        r2 = np.array(list(res2.values()))
        t.append(f"Düse h fest, L Klick:     x_p = {q[0]:+.3f} m, z_p = {q[1]:.3f} m, L = {q[2]:.3f} m, h = {q[3]:+.3f} m "
                 f"(Residuen RMS x {np.sqrt(np.mean(r2[:, 0] ** 2)) * 100:.1f} cm, "
                 f"z {np.sqrt(np.mean(r2[:, 1] ** 2)) * 100:.1f} cm)")
    elif duese_masse is not None:
        L, h = duese_masse
        q, res2 = duese_mit_massen(clips, L, h)
        r2 = np.array(list(res2.values()))
        t.append(f"Düse mit Massen:          x_p = {q[0]:+.3f} m, z_p = {q[1]:.3f} m, L = {L:.3f} m, h = {h:+.3f} m "
                 f"(Residuen RMS x {np.sqrt(np.mean(r2[:, 0] ** 2)) * 100:.1f} cm, "
                 f"z {np.sqrt(np.mean(r2[:, 1] ** 2)) * 100:.1f} cm)")
        t.append(f"  Differenz Klick − Mass: L {(p[2] - L) * 100:+.1f} cm, h {(p[3] - h) * 100:+.1f} cm")
    return "\n".join(t)


def zusammenfassung(m, clips, zeilen, info):
    nd, n = m._nd(), m._n()
    d_text = {0: "", 1: " + d"}.get(nd, " + d(t)")
    t = ["Modell E (einfach): Δz = (tan θ" + d_text + ")·Δx + c2(t)·Δx² + c3(t)·Δx³,  "
         f"c2, c3 Grad {m.grad_theta} in t = θ/{SKALA_GRAD:g}°",
         f"Düse: {info['quelle_duese']}",
         f"Punkte: {len(zeilen)}, Parameter: {len(m.param)}, χ²_red = {m.chi2_red:.2f}", ""]
    xp, zp, L, h = m.duese_param
    t.append(f"Düse: Drehpunkt x_p = {xp:+.3f} m, z_p = {zp:.3f} m, L = {L:.3f} m, h = {h:+.3f} m")
    s = np.sqrt(np.diag(m.kovarianz))
    namen = [f"d_{i}" for i in range(nd)] + [f"a_{i}" for i in range(n)] + [f"b_{i}" for i in range(n)]
    t.append("Parameter (σ mit √χ²_red skaliert, ohne Korrelation innerhalb eines Schusses -> eher zu klein):")
    for i, name in enumerate(namen):
        extra = f"   ({np.degrees(m.param[i]):+.2f}° bei kleinem θ)" if i < nd else ""
        t.append(f"  {name:<4}= {m.param[i]:+.5f} ± {s[i]:.5f}{extra}")
    t.append("")
    t.append("Streuung aus den Residuen (r² ≈ C + s_ε²Δx² + s_α²Δx⁴):")
    t.append(f"  √C = {m.c_klick * 100:.1f} cm (Klickrauschen), s_ε = {np.degrees(m.s_eps):.2f}°, "
             f"s_α = {m.s_alpha:.4f} 1/m (bei 8 m: ±{m.s_alpha * 64 * 100:.0f} cm)")
    t.append("")
    t.append("Pro Clip: θ [°]  Aufprall Bild [m]  Modell [m] ± σ_Kurve ± σ_Schuss   RMS Strahl [cm]")
    for c in sorted(clips, key=lambda c: (c["theta"], c["clip"])):
        xl, sl = m.aufprall(c["theta"])
        _, ss = m.aufprall(c["theta"], schuss=True)
        r = [z["residuum"] for z in zeilen if z["clip"] == c["clip"] and z["art"] == "strahl"]
        rms = np.sqrt(np.mean(np.square(r))) * 100 if r else np.nan
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        bild = f"{video[0]['x']:6.2f}" if video else "    - "
        t.append(f"  {c['clip']:<6} {c['theta']:5.1f}   {bild}           {xl:6.2f} ± {sl:.2f} ± {ss:.2f}     {rms:5.1f}")
    return "\n".join(t)


def speichere(m, pfad_json, zusatz=None):
    """zusatz: weitere Blöcke im JSON, z.B. {"ballistik": ...} (von lade_einfach ignoriert)."""
    d = m.als_dict()
    d.update(zusatz or {})
    with open(pfad_json, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def lade_einfach(pfad_json):
    """fit_modell_einfach.json -> EinfachModell (Gegenstück zu speichere)."""
    with open(pfad_json, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("art") != "einfach":
        raise ValueError(f"{pfad_json}: Modelltyp {d.get('art')!r} statt 'einfach'")
    if float(d["skala_grad"]) != SKALA_GRAD:
        raise ValueError(f"{pfad_json}: Winkelskala {float(d['skala_grad'])} statt {SKALA_GRAD}")
    return EinfachModell(int(d["grad_theta"]), int(d["grad_d"]), d["param"], d["kovarianz"], d["duese"],
                         float(d["s_eps_rad"]), float(d["s_alpha"]), theta_bereich=d["theta_bereich_grad"])
