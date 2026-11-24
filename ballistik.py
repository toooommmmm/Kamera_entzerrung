"""Ballistisches Strahlmodell B: Massenpunkt mit quadratischem Luftwiderstand.

Nur für die Extrapolation zu flacheren Winkeln (θ < θ_min von Modell E0, bis −20°) gedacht.
Innerhalb des gemessenen Bereichs bleibt E0 (einfach.py) das Modell.

Start am Düsenaustritt aus der E0-Düsengeometrie (L, h fest, Drehpunkt aus den Klicks):

    x(0) = x_d(θ),  z(0) = z_d(θ),  v(0) = v0·(cos θ, sin θ)
    ẍ = −k·|v|·vx,  z̈ = −g − k·|v|·vz

Zwei Parameter für alle Winkel: v0 [m/s] (Austrittsgeschwindigkeit) und k [1/m] (Widerstand).
RK4 mit festem dt = 1 ms, z(x) linear zwischen den Stützstellen interpoliert.

Fit wie bei E0: alle geklickten Strahlpunkte + geklickter Aufprall, Residuum z_Modell(x) − z_Klick,
σ² = σ_z² + (Steigung·σ_x)², zwei Durchgänge, Kovarianz mit χ²_red skaliert (falls > 1).
Streuung von Schuss zu Schuss wie bei E0 (einfach._streuung): r² ≈ C + s_ε²Δx² + s_α²Δx⁴.
"""

import numpy as np
from scipy.optimize import least_squares, nnls

import einfach

G = 9.81
DT = 1e-3


def streuung(dx, r, iterationen=200):
    """Wie einfach._streuung (r² ≈ C + s_ε²Δx² + s_α²Δx⁴, iterativ 1/v-gewichtete NNLS), aber gedämpft:
    v ← √(v·v_neu). Ohne Dämpfung springt die Iteration bei grossen systematischen Residuen zwischen
    "nur C" und "nur s_α" hin und her. Gleicher Fixpunkt, bei E0-Residuen gleiches Ergebnis."""
    A = np.column_stack([np.ones_like(dx), dx ** 2, dx ** 4])
    v = np.full_like(r, np.mean(r ** 2))
    for _ in range(iterationen):
        koef, _ = nnls(A / v[:, None], r ** 2 / v)
        v_neu = np.sqrt(v * np.maximum(A @ koef, 1e-6))
        if np.allclose(v_neu, v, rtol=1e-6):
            break
        v = v_neu
    return np.sqrt(koef)


class BallistikModell:
    """Gleiche Schnittstelle wie einfach.EinfachModell (z, steigung, sigma_z, sigma_schuss, aufprall,
    x_start), damit die Vergleichs- und Plotfunktionen unverändert funktionieren.

    Eine Austrittsgeschwindigkeit v0 für alle Winkel. Widerstand k(θ) = Σ kp_i·t^i, t = θ/30°
    (grad_k = len(kp) − 1; grad_k = 0: ein k für alle Winkel). k wird bei 0 abgeschnitten."""

    def __init__(self, v0, kp, kovarianz, duese, g=G, s_eps=0.0, s_alpha=0.0, c_klick=0.0, chi2_red=1.0,
                 dt=DT, theta_bereich=(-90.0, 90.0), ks=None):
        self.v0 = float(v0)
        self.ks = None if ks is None else float(ks)   # Zunahme des Widerstands mit der Flugstrecke [1/m²]
        self.kp = np.atleast_1d(np.asarray(kp, np.float64)).copy()
        self.kovarianz = np.asarray(kovarianz, np.float64)
        self.duese_param = np.asarray(duese, np.float64)
        self.g, self.dt = float(g), float(dt)
        self.s_eps, self.s_alpha, self.c_klick = float(s_eps), float(s_alpha), float(c_klick)
        self.chi2_red = float(chi2_red)
        self.theta_bereich = tuple(float(v) for v in theta_bereich)
        self._cache = {}

    @property
    def grad_k(self):
        return len(self.kp) - 1

    @property
    def param(self):
        return np.concatenate([[self.v0], self.kp] + ([[self.ks]] if self.ks is not None else []))

    @param.setter
    def param(self, p):
        if self.ks is None:
            self.v0, self.kp = float(p[0]), np.asarray(p[1:], np.float64).copy()
        else:
            self.v0, self.kp, self.ks = float(p[0]), np.asarray(p[1:-1], np.float64).copy(), float(p[-1])

    def k_von(self, theta, kp=None):
        kp = self.kp if kp is None else np.asarray(kp, np.float64)
        t = np.asarray(theta, np.float64) / einfach.SKALA_GRAD
        return np.maximum(sum(c * t ** i for i, c in enumerate(kp)), 0.0)

    def duese(self, theta):
        return einfach.duese_position(self.duese_param, theta)

    def x_start(self, theta):
        return float(self.duese(theta)[0])

    # -- Integration --------------------------------------------------------

    def integriere(self, thetas, x_bis, p=None, z_min=-30.0, t_max=8.0):
        """RK4 für mehrere Winkel gleichzeitig, bis jeder Winkel x ≥ x_bis oder z < z_min erreicht.
        p = (v0, kp...) oder None (eigene Parameter). -> Array (Schritte, 4, n) mit (x, z, vx, vz)."""
        p = self.param if p is None else np.asarray(p, np.float64)
        v0 = p[0]
        thg = np.atleast_1d(np.asarray(thetas, np.float64))
        if self.ks is None:
            k, ks = self.k_von(thg, p[1:]), 0.0
        else:
            k, ks = self.k_von(thg, p[1:-1]), p[-1]
        th = np.radians(thg)
        x_bis = np.broadcast_to(np.asarray(x_bis, np.float64), th.shape)
        xd, zd = self.duese(thg)
        s = np.array([xd, zd, v0 * np.cos(th), v0 * np.sin(th), np.zeros_like(th)], np.float64)   # + Flugstrecke
        g, dt = self.g, self.dt

        def f(s):
            v = np.hypot(s[2], s[3])
            kk = np.maximum(k + ks * s[4], 0.0)   # Strahl zerfällt mit der Flugstrecke: k(s) = k + k_s·s
            return np.array([s[2], s[3], -kk * v * s[2], -g - kk * v * s[3], v])

        verlauf = [s]
        for _ in range(int(t_max / dt)):
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            verlauf.append(s)
            if np.all((s[0] >= x_bis) | (s[1] < z_min)):
                break
        return np.stack(verlauf)

    def _bahnen(self, thetas, x_bis, p):
        """Bahnen pro Winkel aus dem Cache, fehlende (oder zu kurze) gemeinsam integrieren.
        -> dict θ -> (x, z, vz/vx)."""
        if len(self._cache) > 5000:
            self._cache.clear()
        schl = tuple(float(v) for v in p)
        fehlt, bis = [], []
        for th, xb in zip(thetas, x_bis):
            b = self._cache.get((th, schl))
            if b is None or (b[0][-1] < xb and b[1][-1] >= -30.0 + 1e-9):
                fehlt.append(th)
                bis.append(xb + 0.05)
        if fehlt:
            H = self.integriere(fehlt, bis, p)
            for i, th in enumerate(fehlt):
                self._cache[(th, schl)] = (H[:, 0, i], H[:, 1, i], H[:, 3, i] / H[:, 2, i])
        return {th: self._cache[(th, schl)] for th in thetas}

    def _auswerten(self, x, theta, p=None):
        """-> (z, Steigung) an den Stellen x für Winkel θ (θ skalar oder gleich lang wie x)."""
        p = self.param if p is None else np.asarray(p, np.float64)
        x = np.atleast_1d(np.asarray(x, np.float64))
        th = np.broadcast_to(np.asarray(theta, np.float64), x.shape)
        eindeutig, inv = np.unique(np.round(th, 9), return_inverse=True)
        inv = inv.reshape(x.shape)
        x_bis = [float(x[inv == i].max()) for i in range(len(eindeutig))]
        bahnen = self._bahnen([float(t) for t in eindeutig], x_bis, p)
        z = np.empty_like(x)
        st = np.empty_like(x)
        for i, t in enumerate(eindeutig):
            bx, bz, bs = bahnen[float(t)]
            m = inv == i
            xi = x[m]
            zi = np.interp(xi, bx, bz, right=np.nan)
            si = np.interp(xi, bx, bs, right=np.nan)
            vor = xi < bx[0]   # vor der Düse: Gerade in Düsenrichtung
            zi[vor] = bz[0] + bs[0] * (xi[vor] - bx[0])
            si[vor] = bs[0]
            z[m], st[m] = zi, si
        return z, st

    # -- Schnittstelle wie EinfachModell -------------------------------------

    def z(self, x, theta):
        return self._auswerten(x, theta)[0]

    def steigung(self, x, theta):
        return self._auswerten(x, theta)[1]

    def gradient(self, x, theta):
        """∂z/∂(v0, kp...), zentrale Differenzen."""
        p0 = self.param
        spalten = []
        for i in range(len(p0)):
            h = 1e-3 * abs(p0[i]) if i == 0 else max(1e-3 * abs(p0[i]), 1e-5)
            pp, pm = p0.copy(), p0.copy()
            pp[i] += h
            pm[i] -= h
            spalten.append((self._auswerten(x, theta, pp)[0] - self._auswerten(x, theta, pm)[0]) / (2 * h))
        return np.column_stack(spalten)

    def sigma_z(self, x, theta):
        g = self.gradient(x, theta)
        return np.sqrt(np.einsum("ij,jk,ik->i", g, self.kovarianz, g))

    def _dx(self, x, theta):
        x = np.atleast_1d(np.asarray(x, np.float64))
        return x - self.duese(np.broadcast_to(np.asarray(theta, np.float64), x.shape))[0]

    def sigma_schuss(self, x, theta):
        dx = self._dx(x, theta)
        return np.sqrt(self.sigma_z(x, theta) ** 2 + self.s_eps ** 2 * dx ** 2 + self.s_alpha ** 2 * dx ** 4)

    def aufprall(self, theta, schuss=False, z_boden=0.0):
        """Erster Durchgang durch z = z_boden (absteigend) und sein horizontales σ."""
        theta = float(theta)
        H = self.integriere([theta], [np.inf], z_min=z_boden)
        x, z = H[:, 0, 0], H[:, 1, 0]
        if z[-1] >= z_boden:
            return np.nan, np.nan
        i = int(np.argmax(z < z_boden))
        xl = float(x[i - 1] + (z_boden - z[i - 1]) * (x[i] - x[i - 1]) / (z[i] - z[i - 1]))
        s = abs(float(self.steigung(xl, theta)[0]))
        sig = self.sigma_schuss(xl, theta) if schuss else self.sigma_z(xl, theta)
        return xl, float(sig[0] / s)

    def korrelation(self):
        s = np.sqrt(np.diag(self.kovarianz))
        return self.kovarianz / np.outer(s, s)

    def als_dict(self, theta_min_grad=-20.0, modellfehler=None):
        d = {"v0": self.v0, "kp": self.kp.tolist(), "kovarianz": self.kovarianz.tolist(), "g": self.g,
             "dt": self.dt, "skala_grad": einfach.SKALA_GRAD,
             "theta_min_grad": float(theta_min_grad), "s_eps_rad": self.s_eps, "s_alpha": self.s_alpha,
             "duese": self.duese_param.tolist(),
             "formel": ("Start x_d, z_d wie Modell E; v(0) = v0 (cos θ, sin θ); k(θ) = max(0, Σ kp_i t^i), "
                        "t = θ/skala_grad; ẍ = −k|v|vx, z̈ = −g − k|v|vz; RK4 dt, z(x) linear interpoliert")}
        if self.ks is not None:
            d["ks"] = self.ks
        if modellfehler is not None:
            d["modellfehler"] = modellfehler
        return d


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def datenzeilen(clips, duese_p, mit_video_aufprall=True):
    """Gleiche Datenpunkte und σ wie einfach.fit_einfach."""
    zeilen = []
    for c in clips:
        xd, zd = einfach.duese_position(duese_p, c["theta"])
        for p in c["punkte"]:
            if p["typ"] == "strahl" or (mit_video_aufprall and p["typ"] == "aufprall"):
                sx = float(np.hypot(p["sx"], c["sigma_aufprall"])) if p["typ"] == "aufprall" else p["sx"]
                zeilen.append({"clip": c["clip"], "theta": c["theta"], "art": p["typ"], "x": p["x"], "z": p["z"],
                               "sx": sx, "sz": p["sz"], "xd": float(xd), "zd": float(zd)})
    return zeilen


def fit_ballistik(clips, duese_masse=(0.81, 0.0), grad_k=0, start=(15.0, 0.1), mit_video_aufprall=True,
                  g=G, mit_weg=False, duese_p=None):
    """Eine Austrittsgeschwindigkeit v0, Widerstand k(θ) polynomial vom Grad grad_k in t = θ/30°.
    -> (BallistikModell, Datenzeilen mit Residuen, Info-dict)."""
    if duese_p is None:
        duese_p, duese_res = einfach.duese_mit_massen(clips, *duese_masse)
    else:   # Drehpunkt vorgegeben (z.B. aus dem E0-Fit), damit beide Modelle an derselben Düse starten
        duese_p, duese_res = np.asarray(duese_p, np.float64), {}
    zeilen = datenzeilen(clips, duese_p, mit_video_aufprall)
    th = np.array([z["theta"] for z in zeilen])
    x = np.array([z["x"] for z in zeilen])
    zk = np.array([z["z"] for z in zeilen])
    sx = np.array([z["sx"] for z in zeilen])
    sz = np.array([z["sz"] for z in zeilen])
    dx = x - np.array([z["xd"] for z in zeilen])
    bereich = (float(th.min()), float(th.max()))
    n_par = 2 + grad_k + int(mit_weg)

    p = np.zeros(n_par)
    p[0], p[1] = start
    m = BallistikModell(p[0], p[1:2 + grad_k], np.zeros((n_par, n_par)), duese_p, g=g, theta_bereich=bereich,
                        ks=0.0 if mit_weg else None)
    unten = [1.0, 0.0] + [-1.0] * grad_k + [-1.0] * int(mit_weg)     # k(0°) ≥ 0, Steigungen frei
    oben = [60.0, 1.0] + [1.0] * grad_k + [1.0] * int(mit_weg)
    steig = np.tan(np.radians(th))
    for _ in range(2):
        s = np.maximum(np.sqrt(sz ** 2 + (steig * sx) ** 2), 1e-4)

        def res(q):
            r = (m._auswerten(x, th, q)[0] - zk) / s
            return np.where(np.isfinite(r), r, 1e3)   # Bahn reicht nicht bis x: grosse Strafe

        e = least_squares(res, p, x_scale=[1.0] + [0.01] * (n_par - 1), diff_step=1e-4,
                          bounds=(unten, oben))
        p = e.x
        m.param = p
        steig = m.steigung(x, th)
    fg = len(zk) - n_par
    chi2 = float(np.sum(res(p) ** 2) / fg)
    J = m.gradient(x, th) / s[:, None]
    m.kovarianz = np.linalg.inv(J.T @ J) * max(1.0, chi2)
    m.chi2_red = chi2

    r = zk - m.z(x, th)
    m.c_klick, m.s_eps, m.s_alpha = streuung(dx, r)
    for i, zl in enumerate(zeilen):
        zl["sigma_eff"] = float(s[i] * np.sqrt(max(1.0, chi2)))
        zl["residuum"] = float(r[i])
    return m, zeilen, {"duese_res": duese_res, "erfolg": bool(e.success), "nachricht": e.message}


def parameter_namen(m):
    return ["v0"] + [f"k{i}" for i in range(len(m.kp))] + (["k_s"] if m.ks is not None else [])


def zusammenfassung(m, n_punkte=None, winkel=(-20, -10, -3, 2, 6, 11, 15, 20, 25)):
    s = np.sqrt(np.diag(m.kovarianz))
    namen = parameter_namen(m)
    einheit = {"v0": "m/s"}
    teile = [f"{n} = {v:.5g} ± {e:.2g}{' ' + einheit[n] if n in einheit else ''}"
             for n, v, e in zip(namen, m.param, s)]
    kor = m.korrelation()
    kt = ", ".join(f"{namen[i]}/{namen[j]} {kor[i, j]:+.2f}" for i in range(len(namen)) for j in range(i + 1, len(namen)))
    weg = " + k_s·Flugstrecke" if m.ks is not None else ""
    t = [f"Ballistik (k Grad {m.grad_k} in θ/{einfach.SKALA_GRAD:g}°{weg}): " + ", ".join(teile)
         + f", χ²_red = {m.chi2_red:.2f}" + (f", {n_punkte} Punkte" if n_punkte else ""),
         f"  Korrelationen: {kt}",
         ("  k(θ) [1/m]: " + ", ".join(f"{w:g}°: {float(m.k_von(w)):.4f}" for w in winkel) if m.grad_k else
          f"  k bei Flugstrecke 0/3/6/10 m [1/m]: " + ", ".join(
              f"{max(0.0, m.kp[0] + (m.ks or 0.0) * sw):.4f}" for sw in (0, 3, 6, 10))),
         f"  Streuung: √C = {m.c_klick * 100:.1f} cm, s_ε = {np.degrees(m.s_eps):.2f}°, s_α = {m.s_alpha:.4f} 1/m",
         f"  Düse: x_p = {m.duese_param[0]:+.3f} m, z_p = {m.duese_param[1]:.3f} m, L = {m.duese_param[2]:.3f} m, "
         f"h = {m.duese_param[3]:+.3f} m"]
    return "\n".join(t)


# ---------------------------------------------------------------------------
# Validierung: Fit auf steileren Winkeln, Vorhersage der flacheren (strahl.py ballistik-test)
# ---------------------------------------------------------------------------

BINS = ((0.0, 2.0), (2.0, 6.0), (6.0, np.inf))

# Abnahmekriterien (vor dem Ergebnis festgelegt, siehe Plan)
KRIT_AUFPRALL_M = 0.30      # Aufprallfehler RMS ≲ Landestreuung
KRIT_WINKEL_GRAD = 0.80     # Winkelfehler RMS ≲ E0 leave-one-run-out
KRIT_IN_1SIGMA = 0.50       # Anteil der Validierungspunkte in ±1σ ("mostly")
KRIT_SCHLECHTER = 1.10      # "nicht deutlich schlechter" als E0 extrapoliert: höchstens 10 % grössere RMS
KRIT_ANSTIEG_M = 0.15       # Aufprall-RMS Gruppe 0 − Gruppe 5 (9° vs. 4° Lücke) höchstens 15 cm


def gruppe(clip):
    return clip.split("_")[0]


def schuss_kennzahlen(m, c, lo, hi):
    """Residuen eines Schusses gegen ein Modell beim gemessenen θ."""
    import geometrie
    th = c["theta"]
    pts = [p for p in c["punkte"] if p["typ"] in ("strahl", "aufprall")]
    x = np.array([p["x"] for p in pts])
    z = np.array([p["z"] for p in pts])
    sx = np.array([np.hypot(p["sx"], c["sigma_aufprall"]) if p["typ"] == "aufprall" else p["sx"] for p in pts])
    sz = np.array([p["sz"] for p in pts])
    dx = x - m.x_start(th)
    zm = m.z(x, th)
    r = z - zm
    st = np.atleast_1d(m.steigung(x, th))
    s_klick = np.sqrt(sz ** 2 + (st * sx) ** 2)
    s_ges = np.sqrt(m.sigma_schuss(x, th) ** 2 + s_klick ** 2)   # neuer Schuss + Klickfehler dieses Punkts
    video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
    xl, sl = m.aufprall(th, schuss=True)
    return {"clip": c["clip"], "gruppe": gruppe(c["clip"]), "theta": th, "dx": dx, "r": r, "norm": r / s_ges,
            "aufprall_fehler": (xl - video[0]["x"]) if video else np.nan, "sigma_aufprall": sl,
            "winkel_fehler": geometrie.winkel_schaetzen(m, c, lo, hi) - th}


def _rms(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan


def zusammenfassen(liste):
    dx = np.concatenate([e["dx"] for e in liste])
    r = np.concatenate([e["r"] for e in liste])
    n = np.concatenate([e["norm"] for e in liste])
    af = np.array([e["aufprall_fehler"] for e in liste], float)
    wf = np.array([e["winkel_fehler"] for e in liste], float)
    k = {"n": len(liste), "aufprall_rms": _rms(af), "aufprall_mittel": float(np.nanmean(af)),
         "winkel_rms": _rms(wf), "winkel_mittel": float(np.nanmean(wf)),
         "in_1sigma": float(np.mean(np.abs(n) <= 1.0)), "norm_rms": _rms(n)}
    for lo, hi in BINS:
        s = (dx >= lo) & (dx < hi)
        k[f"bin_{lo:g}"] = (_rms(r[s]), int(s.sum()))
    return k


def validierung(clips, duese_masse=(0.81, 0.0), train_gruppen=("10", "15", "20"), val_gruppen=("5", "0"),
                durchfluss_l_min=None, duesen_d_mm=None, grad_k=0, final_gruppen=None, mit_weg=False):
    """-> (Text, Modelle, Ergebnisse, bestanden)."""
    train = [c for c in clips if gruppe(c["clip"]) in train_gruppen]
    val = sorted([c for c in clips if gruppe(c["clip"]) in val_gruppen], key=lambda c: (-c["theta"], c["clip"]))
    if not train or not val:
        raise SystemExit("Trainings- oder Validierungsclips fehlen")

    mb, zb, info = fit_ballistik(train, duese_masse, grad_k, mit_weg=mit_weg)
    me, _, _ = einfach.fit_einfach(train, 2, -1, duese_masse=duese_masse)
    me_alle, _, _ = einfach.fit_einfach(clips, 2, -1, duese_masse=duese_masse)
    mb_alle, zb_alle, _ = fit_ballistik(clips, duese_masse, grad_k, mit_weg=mit_weg)
    modelle = {"B": mb, "E0_train": me, "E0_alle": me_alle}
    namen = {"B": "Ballistik k(s) (Tr.)" if mit_weg else f"Ballistik k Grad {grad_k} (Tr.)", "E0_train": "E0 extrapoliert (Training)", "E0_alle": "E0 alle Clips (Referenz)"}

    th_tr = sorted({c["theta"] for c in train})
    th_val = sorted({c["theta"] for c in val})
    t = [f"Ballistik-Validierung: Fit auf Clips {', '.join(train_gruppen)} ({len(train)} Schüsse, "
         f"θ = {', '.join(f'{v:g}' for v in th_tr)}°), Vorhersage Clips {', '.join(val_gruppen)} "
         f"({len(val)} Schüsse, θ = {', '.join(f'{v:g}' for v in th_val)}°)",
         f"Trainingsclips: {', '.join(c['clip'] for c in train)}",
         f"Validierungsclips: {', '.join(c['clip'] for c in val)}",
         f"Extrapolation unter den tiefsten Trainingswinkel: "
         + ", ".join(f"Gruppe {g}: {min(th_tr) - np.mean([c['theta'] for c in val if gruppe(c['clip']) == g]):.1f}°"
                     for g in val_gruppen), ""]
    t.append("Fit Training:   " + zusammenfassung(mb, len(zb)).replace("\n", "\n                "))
    t.append(f"                least_squares: {info['nachricht']}")
    t.append("Fit alle Clips: " + zusammenfassung(mb_alle, len(zb_alle)).replace("\n", "\n                "))
    sb = np.sqrt(np.diag(mb.kovarianz))
    t.append("Stabilität alle − Training: " + ", ".join(
        f"Δ{n} = {a - b:+.4g} ({(a - b) / e:+.1f} σ_train)"
        for n, a, b, e in zip(parameter_namen(mb), mb_alle.param, mb.param, sb)))
    r_alle = np.array([z["residuum"] for z in zb_alle])
    dx_alle = np.array([z["x"] - z["xd"] for z in zb_alle])
    t.append("  Ballistik auf allen Clips, Residuen-RMS pro Δx: " + ", ".join(
        f"{lo:g}–{hi:g} m: {_rms(r_alle[(dx_alle >= lo) & (dx_alle < hi)]) * 100:.0f} cm"
        for lo, hi in ((0, 2), (2, 6), (6, 8), (8, 13))))
    if final_gruppen:
        final = [c for c in clips if gruppe(c["clip"]) in final_gruppen]
        mf, zf, _ = fit_ballistik(final, duese_masse, grad_k, mit_weg=mit_weg)
        t.append(f"Fit Endsatz (Clips {', '.join(final_gruppen)}, {len(final)} Schüsse): "
                 + zusammenfassung(mf, len(zf)).replace("\n", "\n                "))
        t.append("Stabilität Endsatz − Training: " + ", ".join(
            f"Δ{n} = {a - b:+.4g} ({(a - b) / e:+.1f} σ_train)"
            for n, a, b, e in zip(parameter_namen(mb), mf.param, mb.param, sb)))
        rf = np.array([z["residuum"] for z in zf])
        dxf = np.array([z["x"] - z["xd"] for z in zf])
        t.append("  Endsatz-Fit, Residuen-RMS pro Δx: " + ", ".join(
            f"{lo:g}–{hi:g} m: {_rms(rf[(dxf >= lo) & (dxf < hi)]) * 100:.0f} cm"
            for lo, hi in ((0, 2), (2, 6), (6, 8), (8, 13))))
        for c in sorted(final, key=lambda c: c["theta"]):
            video = [q for q in c["punkte"] if q["typ"] == "aufprall"]
            if video:
                xl, _ = mf.aufprall(c["theta"])
                t.append(f"    {c['clip']:<5} θ = {c['theta']:3.0f}°  Aufprall Modell − Klick = {(xl - video[0]['x']) * 100:+4.0f} cm")
    if durchfluss_l_min and duesen_d_mm:
        a = np.pi * (duesen_d_mm / 2000.0) ** 2
        t.append(f"Plausibilität: Q/A = {durchfluss_l_min / 60000.0 / a:.2f} m/s (Q = {durchfluss_l_min:g} l/min, "
                 f"d = {duesen_d_mm:g} mm) gegen v0 = {mb.v0:.2f} m/s")
    else:
        t.append("Plausibilität v0 gegen Q/A: Durchfluss / Düsendurchmesser nicht angegeben "
                 "(--durchfluss-l-min, --duesen-d-mm)")
    t.append("")

    ergebnisse = {}
    for name, m in modelle.items():
        ergebnisse[name] = [schuss_kennzahlen(m, c, c["theta"] - 10.0, c["theta"] + 10.0) for c in val]

    t.append("Pro Validierungsschuss: Aufprallfehler = Modell − geklickt [cm], Winkelfehler = θ_bester − θ_gemessen [°],")
    t.append("Anteil Punkte in ±1σ (σ² = σ_Schuss² + Klick-σ² des Punkts), RMS Residuen pro Δx [cm]")
    kopf = f"  {'Clip':<5} {'θ':>4}  {'Modell':<27} {'Aufpr.':>7} {'±σ':>4} {'Winkel':>7} {'in1σ':>5}  " + \
           "  ".join(f"{lo:g}–{hi:g} m" if np.isfinite(hi) else f">{lo:g} m" for lo, hi in BINS)
    t.append(kopf)
    for i, c in enumerate(val):
        for name in modelle:
            e = ergebnisse[name][i]
            k = zusammenfassen([e])
            bins = "  ".join(f"{k[f'bin_{lo:g}'][0] * 100:5.1f}" for lo, _ in BINS)
            t.append(f"  {c['clip']:<5} {c['theta']:4.0f}  {namen[name]:<27} {e['aufprall_fehler'] * 100:+7.0f} "
                     f"{e['sigma_aufprall'] * 100:4.0f} {e['winkel_fehler']:+7.2f} {k['in_1sigma']:5.0%}  {bins}")
    t.append("")

    t.append("Zusammenfassung pro Gruppe (RMS, in Klammern Mittelwert = systematischer Fehler):")
    t.append(f"  {'Gruppe':<10} {'Modell':<27} {'Aufprall cm':>14} {'Winkel °':>14} {'in1σ':>5} {'RMS r/σ':>7}  "
             + "  ".join(f"{lo:g}–{hi:g} m (n)" if np.isfinite(hi) else f">{lo:g} m (n)" for lo, hi in BINS))
    kz = {}
    for g in list(val_gruppen) + ["alle"]:
        for name in modelle:
            liste = [e for e in ergebnisse[name] if g == "alle" or e["gruppe"] == g]
            k = zusammenfassen(liste)
            kz[(g, name)] = k
            bins = "  ".join(f"{k[f'bin_{lo:g}'][0] * 100:6.1f} ({k[f'bin_{lo:g}'][1]:2d})" for lo, _ in BINS)
            t.append(f"  {g:<10} {namen[name]:<27} {k['aufprall_rms'] * 100:5.0f} ({k['aufprall_mittel'] * 100:+5.0f}) "
                     f"{k['winkel_rms']:6.2f} ({k['winkel_mittel']:+5.2f}) {k['in_1sigma']:5.0%} {k['norm_rms']:7.2f}  {bins}")
    t.append("")

    b, e0 = kz[("alle", "B")], kz[("alle", "E0_train")]
    kriterien = [
        (f"Aufprallfehler RMS ≤ {KRIT_AUFPRALL_M * 100:.0f} cm", b["aufprall_rms"] <= KRIT_AUFPRALL_M,
         f"{b['aufprall_rms'] * 100:.0f} cm"),
        (f"Winkelfehler RMS ≤ {KRIT_WINKEL_GRAD:.1f}°", b["winkel_rms"] <= KRIT_WINKEL_GRAD, f"{b['winkel_rms']:.2f}°"),
        (f"nicht schlechter als E0 extrapoliert (≤ {KRIT_SCHLECHTER:.1f}× Aufprall- und Winkel-RMS)",
         b["aufprall_rms"] <= KRIT_SCHLECHTER * e0["aufprall_rms"] and b["winkel_rms"] <= KRIT_SCHLECHTER * e0["winkel_rms"],
         f"{b['aufprall_rms'] * 100:.0f} vs {e0['aufprall_rms'] * 100:.0f} cm, {b['winkel_rms']:.2f} vs {e0['winkel_rms']:.2f}°"),
        (f"Validierungspunkte in ±1σ ≥ {KRIT_IN_1SIGMA:.0%}", b["in_1sigma"] >= KRIT_IN_1SIGMA, f"{b['in_1sigma']:.0%}"),
    ]
    if len(val_gruppen) >= 2:   # wächst der Fehler mit dem Abstand zum tiefsten Trainingswinkel?
        g5, g0 = kz[(val_gruppen[0], "B")], kz[(val_gruppen[-1], "B")]
        kriterien.append((f"Aufprall-RMS Gruppe {val_gruppen[-1]} − Gruppe {val_gruppen[0]} ≤ {KRIT_ANSTIEG_M * 100:.0f} cm",
                          g0["aufprall_rms"] - g5["aufprall_rms"] <= KRIT_ANSTIEG_M,
                          f"{g0['aufprall_rms'] * 100:.0f} − {g5['aufprall_rms'] * 100:.0f} cm"))
    bestanden = all(ok for _, ok, _ in kriterien)
    t.append("Abnahmekriterien (Ballistik, Training -> Validierung):")
    for text, ok, wert in kriterien:
        t.append(f"  [{'ok' if ok else 'NICHT ERFÜLLT'}] {text}: {wert}")
    t.append(f"=> {'BESTANDEN' if bestanden else 'NICHT BESTANDEN: nicht integrieren'}")
    return "\n".join(t), {**modelle, "B_alle": mb_alle}, ergebnisse, bestanden, val


FARBEN = {"B": "#2a78d6", "E0_train": "#eb6834", "E0_alle": "#1baf7a"}   # dataviz-Standardpalette, Slots 1–3
STILE = {"B": "-", "E0_train": "--", "E0_alle": ":"}
LABEL = {"B": "Ballistik, Fit Training (±1σ Schuss)", "E0_train": "E0 extrapoliert, Fit Training",
         "E0_alle": "E0, Fit alle Clips"}


def plot_validierung(modelle, val, pfad, untertitel="",
                     titel="Ballistik-Validierung: Fit auf Clips 10/15/20, Vorhersage der flacheren Clips 5 und 0"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(val)
    spalten = 3
    zeilen = int(np.ceil(n / spalten))
    fig, achsen = plt.subplots(zeilen * 2, spalten, figsize=(13, 3.6 * zeilen * 2 * 0.75),
                               gridspec_kw={"height_ratios": [2.2, 1] * zeilen}, sharex=True)
    achsen = np.atleast_2d(achsen)
    for i, c in enumerate(val):
        ax = achsen[2 * (i // spalten), i % spalten]
        axr = achsen[2 * (i // spalten) + 1, i % spalten]
        th = c["theta"]
        pts = [p for p in c["punkte"] if p["typ"] in ("strahl", "aufprall")]
        x = np.array([p["x"] for p in pts])
        z = np.array([p["z"] for p in pts])
        x_ende = max(x.max() + 0.5, 8.5)
        for name in ("E0_alle", "E0_train", "B"):
            m = modelle[name]
            xs = np.linspace(m.x_start(th), x_ende, 300)
            zs = m.z(xs, th)
            ax.plot(xs, zs, STILE[name], color=FARBEN[name], lw=2 if name == "B" else 1.5, label=LABEL[name])
            if name == "B":
                ss = m.sigma_schuss(xs, th)
                ax.fill_between(xs, zs - ss, zs + ss, color=FARBEN[name], alpha=0.15, lw=0)
                axr.fill_between(xs, -ss * 100, ss * 100, color=FARBEN[name], alpha=0.15, lw=0)
            axr.plot(x, (z - m.z(x, th)) * 100, "o", ms=4, color=FARBEN[name], mfc="none" if name != "B" else FARBEN[name])
        strahl = [p for p in pts if p["typ"] == "strahl"]
        ax.plot([p["x"] for p in strahl], [p["z"] for p in strahl], "o", ms=4, color="#333333",
                label="geklickte Strahlpunkte")
        auf = [p for p in pts if p["typ"] == "aufprall"]
        if auf:
            ax.plot([auf[0]["x"]], [auf[0]["z"]], "v", ms=8, color="#333333", label="geklickter Aufprall")
        ax.axhline(0, color="k", lw=0.6)
        axr.axhline(0, color="k", lw=0.6)
        ax.set_title(f"Clip {c['clip']}, θ = {th:g}°", fontsize=10, loc="left")
        ax.set_ylim(-0.3, max(2.0, z.max() + 0.3))
        ax.grid(alpha=0.3)
        axr.grid(alpha=0.3)
        if i % spalten == 0:
            ax.set_ylabel("z [m]")
            axr.set_ylabel("z − Modell [cm]")
        if i // spalten == zeilen - 1:
            axr.set_xlabel("x [m]")
    for j in range(n, zeilen * spalten):
        achsen[2 * (j // spalten), j % spalten].axis("off")
        achsen[2 * (j // spalten) + 1, j % spalten].axis("off")
    h, l = achsen[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(titel + (f"\n{untertitel}" if untertitel else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(pfad, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Erweiterung für zielmodell.py (θ unter dem gemessenen Bereich)
# ---------------------------------------------------------------------------

ZIEL_GRUPPEN = ("0", "5", "10", "15")       # flache Schüsse (θ ≤ 11°): Fit für die Extrapolation nach unten
TEST_TRAIN, TEST_VAL = ("5", "10", "15"), ("0",)
THETA_MIN_ZIEL = -20.0


def fit_fuer_ziel(clips, duese_p, theta_min_grad=THETA_MIN_ZIEL):
    """Konstantes v0, k auf den flachen Clips, Düse wie E0. Modellfehler-Koeffizient aus dem Hold-out-Test
    (Fit 5/10/15, Vorhersage 0): c = RMS über die Validierungspunkte mit Δx > 1 m von r / (Winkellücke · Δx),
    Residuen inkl. Streuung, also eher grosszügig. -> (Modell, JSON-Block, Text)."""
    flach = [c for c in clips if gruppe(c["clip"]) in ZIEL_GRUPPEN]
    m, zeilen, _ = fit_ballistik(flach, duese_p=duese_p)
    train = [c for c in clips if gruppe(c["clip"]) in TEST_TRAIN]
    val = [c for c in clips if gruppe(c["clip"]) in TEST_VAL]
    mt, _, _ = fit_ballistik(train, duese_p=duese_p)
    th_lo = min(c["theta"] for c in train)
    q, landung = [], []
    for c in val:
        pts = [p for p in c["punkte"] if p["typ"] in ("strahl", "aufprall")]
        x = np.array([p["x"] for p in pts])
        dx = x - mt.x_start(c["theta"])
        r = np.array([p["z"] for p in pts]) - mt.z(x, c["theta"])
        luecke = th_lo - c["theta"]
        sel = dx > 1.0
        q += list(r[sel] / (luecke * dx[sel]))
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        if video:
            landung.append(mt.aufprall(c["theta"])[0] - video[0]["x"])
    c_mf = float(np.sqrt(np.mean(np.square(q))))
    block = m.als_dict(theta_min_grad, {"m_pro_grad_pro_m": c_mf, "test": f"Fit {'/'.join(TEST_TRAIN)}, Vorhersage "
                                        f"{'/'.join(TEST_VAL)}", "aufprall_rms_m": float(np.sqrt(np.mean(np.square(landung))))})
    block["clips"] = sorted({c["clip"] for c in flach})
    block["theta_fit_bereich_grad"] = [float(min(c["theta"] for c in flach)), float(max(c["theta"] for c in flach))]
    text = (f"Ballistische Erweiterung für θ < θ_min (Clips {', '.join(ZIEL_GRUPPEN)}, Düse wie E0):\n"
            + zusammenfassung(m, len(zeilen)) + "\n"
            f"  Modellfehler: σ_Modell = {c_mf * 100:.3f} cm/(°·m) · (θ_min − θ) · Δx "
            f"(Hold-out {block['modellfehler']['test']}, Aufprall-RMS {block['modellfehler']['aufprall_rms_m'] * 100:.0f} cm)")
    return m, block, text

