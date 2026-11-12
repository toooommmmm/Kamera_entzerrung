"""Strahlmodell mit Geometrie: Kurve beginnt an der Düse, mit der Düsenrichtung.

Modell (alle Längen in m, θ = gemessener absoluter Düsenwinkel, t = θ/30°):

    Düsenspitze:  x_d(θ) = x_p + L·cos θ − h·sin θ
                  z_d(θ) = z_p + L·sin θ + h·cos θ
                  (Drehpunkt (x_p, z_p) des Servos, Spitze im Abstand L entlang und h quer
                  zur Düsenachse; gemeinsam aus den geklickten Düsenpunkten aller Clips gefittet)

    Strahl relativ zur Düse, Δx = x − x_d, Δz = z − z_d:

        Δz = tan(θ + δ)·Δx + a(θ)·Δx² + b(θ)·Δx³
        a(θ) = sec²θ · Σ a_i t^i        b(θ) = sec³θ · Σ b_i t^i      (i = 0 .. grad_theta)

    sec²θ und sec³θ: so hängen die Koeffizienten bei fester Austrittsgeschwindigkeit
    (Pumpe konstant) vom Winkel ab (Wurf mit Luftwiderstand, führende Ordnung).
    b ≡ 0 (grad_x = 2) ergibt die an die Geometrie gebundene Parabel.

Streuung von Schuss zu Schuss (Zufallseffekte pro Clip):
    ε_k  ~ N(0, σ_ε²)   zusätzlicher Winkel  (Neigungsmesser, Austrittsrichtung)
    α_k  ~ N(0, σ_α²)   zusätzliche Krümmung (Pumpe, Wind)  -> + α_k·sec²θ·Δx²
    Messfehler: f·σ_Klick (f = Skalierung der angenommenen Klick-σ)
σ_ε, σ_α, f werden aus den Daten geschätzt (REML, restringierte Maximum-Likelihood),
die festen Parameter mit verallgemeinerten kleinsten Quadraten. Alles linear bis auf δ:
tan(θ+δ) wird um das aktuelle δ linearisiert und wiederholt, bis δ konvergiert.
"""

import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize, minimize_scalar

import modell as basis

SKALA_GRAD = basis.SKALA_GRAD


# ---------------------------------------------------------------------------
# Düsenposition (Drehpunkt + Arm)
# ---------------------------------------------------------------------------

def duese_fit(clips):
    """Geklickte Düsenspitzen aller Clips -> (x_p, z_p, L, h), Residuen pro Clip [m]."""
    zeilen, b, s, namen = [], [], [], []
    for c in clips:
        d = [p for p in c["punkte"] if p["typ"] == "duese"]
        if not d:
            continue
        d, th = d[0], np.radians(c["theta"])
        zeilen += [[1, 0, np.cos(th), -np.sin(th)], [0, 1, np.sin(th), np.cos(th)]]
        b += [d["x"], d["z"]]
        s += [d["sx"], d["sz"]]
        namen.append(c["clip"])
    if len(namen) < 3:
        raise ValueError(f"nur {len(namen)} Clips mit Düsenpunkt, mindestens 3 nötig für das Düsenmodell")
    A, b, s = np.array(zeilen), np.array(b), np.array(s)
    p, *_ = np.linalg.lstsq(A / s[:, None], b / s, rcond=None)
    r = (b - A @ p).reshape(-1, 2)
    return p, dict(zip(namen, r))


def duese_position(p, theta):
    th = np.radians(theta)
    return (p[0] + p[2] * np.cos(th) - p[3] * np.sin(th),
            p[1] + p[2] * np.sin(th) + p[3] * np.cos(th))


# ---------------------------------------------------------------------------
# Modell
# ---------------------------------------------------------------------------

class GeometrieModell:
    """Gleiche Schnittstelle wie modell.Modell (z, sigma_z, steigung, aufprall) plus
    sigma_schuss (Streuung eines neuen Schusses) und winkel_fuer_ziel (Umkehrung)."""

    def __init__(self, grad_x, grad_theta, param, kovarianz, duese, sigma_eps=0.0, sigma_alpha=0.0,
                 f_mess=1.0, theta_bereich=(-90.0, 90.0), freiheitsgrade=0, grad_delta=0):
        self.grad_x, self.grad_theta, self.grad_delta = int(grad_x), int(grad_theta), int(grad_delta)
        self.param = np.asarray(param, np.float64)
        self.kovarianz = np.asarray(kovarianz, np.float64)
        self.duese_param = np.asarray(duese, np.float64)
        self.sigma_eps, self.sigma_alpha, self.f_mess = float(sigma_eps), float(sigma_alpha), float(f_mess)
        self.theta_bereich = (float(theta_bereich[0]), float(theta_bereich[1]))
        self.freiheitsgrade = freiheitsgrade
        self.chi2_red = self.f_mess ** 2     # für Ausgaben, die modell.Modell erwarten

    # --- Hilfen ---
    @property
    def delta(self):
        """δ_0 [rad] (bei grad_delta = 0 der einzige Winkeloffset)."""
        return float(self.param[0])

    def _nd(self):
        return self.grad_delta + 1

    def _n(self):
        return self.grad_theta + 1

    def delta_von(self, theta):
        """δ(θ) = Σ δ_i t^i [rad]: Strahlrichtung − gemessener Düsenwinkel."""
        t = np.asarray(theta, np.float64) / SKALA_GRAD
        return sum(self.param[i] * t ** i for i in range(self._nd()))

    def _teile(self, x, theta):
        x = np.atleast_1d(np.asarray(x, np.float64))
        th = np.broadcast_to(np.asarray(theta, np.float64), x.shape)
        xd, zd = duese_position(self.duese_param, th)
        return x, th, np.radians(th), xd, zd, x - xd

    def ab(self, theta):
        """(a(θ), b(θ)) inklusive sec-Faktoren."""
        th = np.asarray(theta, np.float64)
        t, sec = th / SKALA_GRAD, 1.0 / np.cos(np.radians(th))
        pot = np.stack([t ** i for i in range(self._n())])
        n, o = self._n(), self._nd()
        a = sec ** 2 * np.tensordot(self.param[o:o + n], pot, 1)
        b = sec ** 3 * np.tensordot(self.param[o + n:o + 2 * n], pot, 1) if self.grad_x == 3 else 0.0 * a
        return a, b

    def duese(self, theta):
        return duese_position(self.duese_param, theta)

    def x_start(self, theta):
        return float(self.duese(theta)[0])

    # --- Kurve ---
    def z(self, x, theta):
        x, th, thr, xd, zd, dx = self._teile(x, theta)
        a, b = self.ab(th)
        return zd + np.tan(thr + self.delta_von(th)) * dx + a * dx ** 2 + b * dx ** 3

    def steigung(self, x, theta):
        x, th, thr, xd, zd, dx = self._teile(x, theta)
        a, b = self.ab(th)
        return np.tan(thr + self.delta_von(th)) + 2 * a * dx + 3 * b * dx ** 2

    def gradient(self, x, theta):
        """∂z/∂param, eine Zeile pro Punkt (Reihenfolge wie param)."""
        x, th, thr, xd, zd, dx = self._teile(x, theta)
        t, sec = th / SKALA_GRAD, 1.0 / np.cos(thr)
        c2 = dx / np.cos(thr + self.delta_von(th)) ** 2
        spalten = [c2 * t ** i for i in range(self._nd())]
        spalten += [sec ** 2 * dx ** 2 * t ** i for i in range(self._n())]
        if self.grad_x == 3:
            spalten += [sec ** 3 * dx ** 3 * t ** i for i in range(self._n())]
        return np.column_stack(spalten)

    def sigma_z(self, x, theta):
        """Unsicherheit der mittleren Kurve (feste Parameter)."""
        g = self.gradient(x, theta)
        return np.sqrt(np.einsum("ij,jk,ik->i", g, self.kovarianz, g))

    def sigma_schuss(self, x, theta):
        """Erwartete Abweichung eines neuen Schusses: mittlere Kurve + Streuung von Schuss zu Schuss."""
        x, th, thr, xd, zd, dx = self._teile(x, theta)
        s_eps = dx / np.cos(thr + self.delta_von(th)) ** 2 * self.sigma_eps
        s_alpha = dx ** 2 / np.cos(thr) ** 2 * self.sigma_alpha
        return np.sqrt(self.sigma_z(x, theta) ** 2 + s_eps ** 2 + s_alpha ** 2)

    def aufprall(self, theta, schuss=False):
        """Landepunkt (z = 0) und sein σ_x. schuss=True: σ für einen neuen Schuss."""
        theta = float(theta)
        xd, zd = self.duese(theta)
        a, b = self.ab(theta)
        k = [float(b), float(a), float(np.tan(np.radians(theta) + self.delta_von(theta))), float(zd)]
        if self.grad_x == 2:
            k = k[1:]
        kandidaten = []
        for w in np.roots(k):
            if abs(w.imag) < 1e-9 and 0 < w.real < 40 and self.steigung(xd + w.real, theta)[0] < 0:
                kandidaten.append(w.real)
        if not kandidaten:
            return np.nan, np.nan
        xl = float(xd + min(kandidaten))
        s = abs(self.steigung(xl, theta)[0])
        sig = self.sigma_schuss(xl, theta) if schuss else self.sigma_z(xl, theta)
        return xl, float(sig[0] / s)

    # --- Umkehrung für das Zielen ---
    def winkel_fuer_ziel(self, x, z, bezug="drehpunkt", schuss=True, n=300):
        """Ziel (x, z) -> nötiger absoluter Düsenwinkel θ [°] und σ_θ [°].

        bezug="drehpunkt": x, z relativ zum Servo-Drehpunkt (für die Drohne)
        bezug="boden":     x, z in den Koordinaten des Versuchs (Pfosten, Boden)
        Nur innerhalb der gemessenen Winkel und zwischen Düse und Landepunkt, keine Extrapolation.
        """
        if bezug == "drehpunkt":
            X, Z = self.duese_param[0] + x, self.duese_param[1] + z
        elif bezug == "boden":
            X, Z = float(x), float(z)
        else:
            raise ValueError(f"unbekannter Bezug {bezug!r}")
        ths = np.linspace(*self.theta_bereich, n)

        def gueltig(th):
            # Toleranz 5 cm hinter dem Landepunkt: Ziele auf Bodenhöhe (z = 0) liegen genau am Rand
            xl, _ = self.aufprall(th)
            return np.isfinite(xl) and self.x_start(th) < X <= xl + 0.05

        f = np.array([self.z(X, th)[0] - Z if gueltig(th) else np.nan for th in ths])
        ergebnis = {"x_boden": float(X), "z_boden": float(Z), "theta": np.nan, "sigma_theta": np.nan}
        ok = np.isfinite(f)
        if not ok.any():
            ergebnis["grund"] = "x ausserhalb des gemessenen Bereichs (vor der Düse oder hinter dem Landepunkt)"
            return ergebnis
        for i in range(n - 1):
            if ok[i] and ok[i + 1] and f[i] * f[i + 1] <= 0:
                th = brentq(lambda t: self.z(X, t)[0] - Z, ths[i], ths[i + 1]) if f[i] != f[i + 1] else ths[i]
                h = 0.01
                dzdth = (self.z(X, th + h)[0] - self.z(X, th - h)[0]) / (2 * h)
                sz = (self.sigma_schuss(X, th) if schuss else self.sigma_z(X, th))[0]
                ergebnis.update(theta=float(th), sigma_theta=float(sz / abs(dzdth)), sigma_z=float(sz),
                                dz_dtheta=float(dzdth), grund="ok")
                return ergebnis
        ergebnis["grund"] = ("zu hoch / zu weit: über dem steilsten gemessenen Strahl" if np.nanmax(f) < 0
                             else "zu tief: unter dem flachsten gemessenen Strahl")
        return ergebnis

    def monotonie(self, n_theta=60, n_x=60):
        """Kleinstes ∂z/∂θ [m/°] im Gültigkeitsbereich (muss > 0 sein, damit die Umkehrung eindeutig ist)."""
        werte = []
        for th in np.linspace(*self.theta_bereich, n_theta):
            xl, _ = self.aufprall(th)
            if not np.isfinite(xl):
                continue
            xs = np.linspace(self.x_start(th) + 0.05, xl, n_x)
            werte.append(np.min((self.z(xs, th + 0.01) - self.z(xs, th - 0.01)) / 0.02))
        return float(np.min(werte)) if werte else np.nan

    # --- Speichern ---
    def als_dict(self):
        return {"art": "geometrie", "grad_x": self.grad_x, "grad_theta": self.grad_theta,
                "grad_delta": self.grad_delta,
                "param": self.param.tolist(), "kovarianz": self.kovarianz.tolist(),
                "duese": self.duese_param.tolist(), "sigma_eps_rad": self.sigma_eps,
                "sigma_alpha": self.sigma_alpha, "f_mess": self.f_mess,
                "theta_bereich_grad": list(self.theta_bereich), "skala_grad": SKALA_GRAD}

    def speichere(self, pfad_npz, pfad_json=None):
        d = self.als_dict()
        np.savez(pfad_npz, **{k: np.asarray(v) for k, v in d.items()})
        if pfad_json:
            d["formel"] = ("Düse: x_d = x_p + L cos θ − h sin θ, z_d = z_p + L sin θ + h cos θ, duese = [x_p, z_p, L, h]; "
                           "Δx = x − x_d; z = z_d + tan(θ + δ)·Δx + sec²θ·Σ a_i t^i·Δx² + sec³θ·Σ b_i t^i·Δx³; "
                           "t = θ/skala_grad; δ(θ) = Σ δ_i t^i (rad); param = [δ_0..δ_gd, a_0..a_g, b_0..b_g]; θ absolut in Grad; "
                           "nur für θ in theta_bereich_grad und zwischen Düse und Landepunkt gültig")
            with open(pfad_json, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)


def lade_geometrie(pfad):
    if pfad.endswith(".json"):
        with open(pfad, encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = {k: v for k, v in np.load(pfad).items()}
    if float(d["skala_grad"]) != SKALA_GRAD:
        raise ValueError(f"{pfad}: Winkelskala {float(d['skala_grad'])} statt {SKALA_GRAD}")
    return GeometrieModell(int(d["grad_x"]), int(d["grad_theta"]), d["param"], d["kovarianz"], d["duese"],
                           float(d["sigma_eps_rad"]), float(d["sigma_alpha"]), float(d["f_mess"]),
                           tuple(np.asarray(d["theta_bereich_grad"], float)), 0, int(d.get("grad_delta", 0)))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _gls(X, y, V):
    c = cho_factor(V)
    ViX, Viy = cho_solve(c, X), cho_solve(c, y)
    cov = np.linalg.inv(X.T @ ViX)
    beta = cov @ (X.T @ Viy)
    return beta, cov, c


def _reml(X, Z, y, s, K, frei, start):
    """Varianzkomponenten (σ_ε, σ_α, f) per REML. frei: welche geschätzt werden, sonst start."""
    start = np.asarray(start, float)

    def werte(lp):
        v = start.copy()
        v[frei] = np.exp(lp)
        return v

    def nll(lp):
        se, sa, f = werte(lp)
        G = np.r_[np.full(K, se ** 2), np.full(K, sa ** 2)]
        V = np.diag((f * s) ** 2) + (Z * G) @ Z.T
        try:
            c = cho_factor(V)
        except np.linalg.LinAlgError:
            return 1e12
        ViX, Viy = cho_solve(c, X), cho_solve(c, y)
        A = X.T @ ViX
        beta = np.linalg.solve(A, X.T @ Viy)
        r = y - X @ beta
        logdet_V = 2 * np.sum(np.log(np.diag(c[0])))
        return 0.5 * (logdet_V + np.linalg.slogdet(A)[1] + r @ cho_solve(c, r))

    if not np.any(frei):
        return start
    lp0 = np.log(np.maximum(start[frei], 1e-6))
    res = minimize(nll, lp0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 4000})
    return werte(res.x)


def _datenzeilen(clips, duese_p, ursprung, mit_video_aufprall, mit_log_aufprall):
    zeilen = []
    for k, c in enumerate(clips):
        d = [p for p in c["punkte"] if p["typ"] == "duese"]
        if ursprung == "klick" and d:
            xd, zd = d[0]["x"], d[0]["z"]
        else:
            xd, zd = duese_position(duese_p, c["theta"])
        basis_ = {"clip": c["clip"], "winkel_soll": c["winkel_soll"], "theta": c["theta"], "messung": c["messung"],
                  "k": k, "xd": float(xd), "zd": float(zd)}
        for p in c["punkte"]:
            if p["typ"] == "strahl" or (mit_video_aufprall and p["typ"] == "aufprall"):
                sx = float(np.hypot(p["sx"], c["sigma_aufprall"])) if p["typ"] == "aufprall" else p["sx"]
                zeilen.append(dict(basis_, art=p["typ"], x=p["x"], z=p["z"], sx=sx, sz=p["sz"]))
        if mit_log_aufprall and c["aufprall_log"] is not None:
            zeilen.append(dict(basis_, art="aufprall_log", x=c["aufprall_log"], z=0.0,
                               sx=c["sigma_aufprall"], sz=0.0))
    return zeilen


def fit_geometrie(clips, grad_x=3, grad_theta=0, ursprung="fit", mit_video_aufprall=True,
                  mit_log_aufprall=False, zufall=True, sigma_eps_grad=None, sigma_alpha=None, f_mess=None,
                  grad_delta=0):
    """-> (GeometrieModell, Datenzeilen mit Residuen, Info-dict).

    grad_delta > 0: Winkeloffset δ(θ) = Σ δ_i t^i statt eines festen δ."""
    if grad_x not in (2, 3):
        raise ValueError("grad_x muss 2 oder 3 sein")
    duese_p, duese_res = duese_fit(clips)
    zeilen = _datenzeilen(clips, duese_p, ursprung, mit_video_aufprall, mit_log_aufprall)
    if not zeilen:
        raise ValueError("keine Datenpunkte für den Fit")
    th = np.array([z["theta"] for z in zeilen])
    thr, t = np.radians(th), th / SKALA_GRAD
    sec = 1.0 / np.cos(thr)
    x = np.array([z["x"] for z in zeilen])
    dx = x - np.array([z["xd"] for z in zeilen])
    dz = np.array([z["z"] for z in zeilen]) - np.array([z["zd"] for z in zeilen])
    sx = np.array([z["sx"] for z in zeilen])
    sz = np.array([z["sz"] for z in zeilen])
    kid = np.array([z["k"] for z in zeilen])
    K = len(clips)
    n = grad_theta + 1
    nd = grad_delta + 1
    n_param = nd + n * (grad_x - 1)
    if len(zeilen) <= n_param:
        raise ValueError(f"zu wenige Punkte ({len(zeilen)}) für {n_param} Parameter")
    theta_bereich = (float(th.min()), float(th.max()))

    frei = np.array([sigma_eps_grad is None, sigma_alpha is None, f_mess is None]) & zufall
    vk = np.array([np.radians(sigma_eps_grad) if sigma_eps_grad is not None else np.radians(1.0),
                   sigma_alpha if sigma_alpha is not None else 0.003,
                   f_mess if f_mess is not None else 1.0])
    if not zufall:
        vk[:2] = 0.0

    dkoef, steig = np.zeros(nd), np.tan(thr)
    for _ in range(20):
        delta = sum(dkoef[i] * t ** i for i in range(nd))
        c2 = 1.0 / np.cos(thr + delta) ** 2
        y = dz - np.tan(thr + delta) * dx
        X = np.column_stack([c2 * dx * t ** i for i in range(nd)] + [sec ** 2 * dx ** 2 * t ** i for i in range(n)]
                            + ([sec ** 3 * dx ** 3 * t ** i for i in range(n)] if grad_x == 3 else []))
        Z = np.zeros((len(y), 2 * K))
        Z[np.arange(len(y)), kid] = c2 * dx
        Z[np.arange(len(y)), K + kid] = sec ** 2 * dx ** 2
        s = np.maximum(np.sqrt(sz ** 2 + (steig * sx) ** 2), 1e-4)
        if zufall:
            vk = _reml(X, Z, y, s, K, frei, vk)
            G = np.r_[np.full(K, vk[0] ** 2), np.full(K, vk[1] ** 2)]
            V = np.diag((vk[2] * s) ** 2) + (Z * G) @ Z.T
            beta, cov, c = _gls(X, y, V)
            u = G * (Z.T @ cho_solve(c, y - X @ beta))
        else:
            beta, cov, _ = _gls(X, y, np.diag(s ** 2))
            r = (y - X @ beta) / s
            chi2 = float(r @ r / (len(y) - len(beta)))
            vk[2] = np.sqrt(chi2) if f_mess is None else vk[2]
            cov = cov * max(1.0, vk[2] ** 2) if f_mess is None else cov * vk[2] ** 2
            u = np.zeros(2 * K)
        schritt = beta[:nd]
        dkoef = dkoef + schritt
        param = beta.copy()
        param[:nd] = dkoef
        m = GeometrieModell(grad_x, grad_theta, param, cov, duese_p, vk[0], vk[1], vk[2],
                            theta_bereich, len(y) - len(beta), grad_delta)
        steig = m.steigung(x, th)
        if np.max(np.abs(schritt)) < 1e-8:
            break

    zufall_z = Z @ u
    mz = m.z(x, th)
    for i, zl in enumerate(zeilen):
        zl["sigma_eff"] = float(vk[2] * s[i])
        zl["residuum"] = float(zl["z"] - mz[i])
        zl["residuum_clip"] = float(zl["z"] - mz[i] - zufall_z[i])
    info = {"duese_res": duese_res, "eps": {c["clip"]: float(u[k]) for k, c in enumerate(clips)},
            "alpha": {c["clip"]: float(u[K + k]) for k, c in enumerate(clips)},
            "ursprung": ursprung, "zufall": zufall, "frei": frei.tolist()}
    return m, zeilen, info


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

def parameter_namen(m):
    namen = [f"δ_{i} [°]" for i in range(m._nd())] + [f"a_{i}" for i in range(m._n())]
    if m.grad_x == 3:
        namen += [f"b_{i}" for i in range(m._n())]
    return namen


def zusammenfassung(m, clips, zeilen, info):
    t = []
    t.append("Modell: Geometrie, Kurve ab Düse mit Düsenrichtung")
    t.append("  z = z_d(θ) + tan(θ+δ" + ("(θ)" if m.grad_delta else "") + ")·Δx + sec²θ·a(t)·Δx²" + (" + sec³θ·b(t)·Δx³" if m.grad_x == 3 else "")
             + f",  Δx = x − x_d(θ),  a, b Grad {m.grad_theta} in t = θ/{SKALA_GRAD:g}°")
    t.append(f"Ursprung der Daten: {'geklickte Düse pro Clip' if info['ursprung'] == 'klick' else 'Düsenmodell (Drehpunkt + Arm)'}")
    t.append(f"Punkte: {len(zeilen)}, feste Parameter: {len(m.param)}, Clips: {len(clips)}")
    t.append("")
    xp, zp, L, h = m.duese_param
    res = np.array(list(info["duese_res"].values()))
    schlimmster = max(info["duese_res"], key=lambda k: np.hypot(*info["duese_res"][k]))
    t.append(f"Düse: Drehpunkt x_p = {xp:+.3f} m, z_p = {zp:.3f} m, Arm L = {L:.3f} m, quer h = {h:+.3f} m")
    t.append(f"  Residuen der geklickten Düse: RMS x {np.sqrt(np.mean(res[:, 0] ** 2)) * 100:.1f} cm, "
             f"z {np.sqrt(np.mean(res[:, 1] ** 2)) * 100:.1f} cm; grösste: {schlimmster} "
             f"({info['duese_res'][schlimmster][0] * 100:+.1f}, {info['duese_res'][schlimmster][1] * 100:+.1f}) cm")
    t.append("")
    s = np.sqrt(np.diag(m.kovarianz))
    t.append("Feste Parameter:")
    for i, name in enumerate(parameter_namen(m)):
        if i < m._nd():
            t.append(f"  δ_{i} = {np.degrees(m.param[i]):+.2f} ± {np.degrees(s[i]):.2f} °"
                     + ("   (Strahlrichtung − gemessener Düsenwinkel)" if i == 0 else ""))
        else:
            t.append(f"  {name:<4}= {m.param[i]:+.5f} ± {s[i]:.5f}")
    if m.grad_delta:
        g = lambda th: m.gradient(1.0, th)[0, :m._nd()] / (1.0 / np.cos(np.radians(th) + m.delta_von(th)) ** 2)
        teile = []
        for th in np.linspace(*m.theta_bereich, 5):
            gv = g(th)
            sd = np.sqrt(gv @ m.kovarianz[:m._nd(), :m._nd()] @ gv)
            teile.append(f"{th:.0f}°: {np.degrees(m.delta_von(th)):+.1f}±{np.degrees(sd):.1f}")
        t.append("  δ(θ) [°]: " + ",  ".join(teile))
    t.append("")
    if info["zufall"]:
        geschaetzt = info["frei"]
        t.append("Streuung von Schuss zu Schuss (" + ", ".join(
            n for n, g in zip(("σ_ε", "σ_α", "f"), geschaetzt) if g) + " per REML geschätzt):")
        t.append(f"  σ_ε = {np.degrees(m.sigma_eps):.2f} °     zusätzlicher Winkel pro Schuss "
                 "(Neigungsmesser + Austrittsrichtung)")
        t.append(f"  σ_α = {m.sigma_alpha:.4f} 1/m  zusätzliche Krümmung pro Schuss (Pumpe, Wind); "
                 f"bei 8 m: ±{m.sigma_alpha * 64 * 100:.0f} cm in z")
    t.append(f"  f   = {m.f_mess:.2f}       Skalierung der angenommenen Klick-σ"
             + ("  (< 1: Klicks genauer als angenommen)" if m.f_mess < 1 else ""))
    mono = m.monotonie()
    t.append(f"Monotonie: kleinstes ∂z/∂θ im Gültigkeitsbereich = {mono * 100:.2f} cm/° "
             + ("(> 0: Umkehrung eindeutig)" if mono > 0 else "(ACHTUNG ≤ 0: Umkehrung nicht eindeutig)"))
    t.append(f"Gültig für θ = {m.theta_bereich[0]:g} … {m.theta_bereich[1]:g}°, x zwischen Düse und Landepunkt")
    t.append("")
    t.append("Pro Clip: θ [°]  Aufprall Bild [m]  Modell [m] ± σ_Kurve ± σ_Schuss   ε_k [°]  α_k [1/m]  "
             "RMS zur Kurve / zur Clip-Kurve [cm]")
    for c in sorted(clips, key=lambda c: (c["theta"], c["clip"])):
        xl, sl = m.aufprall(c["theta"])
        _, ss = m.aufprall(c["theta"], schuss=True)
        r = [z for z in zeilen if z["clip"] == c["clip"] and z["art"] == "strahl"]
        rms = np.sqrt(np.mean([z["residuum"] ** 2 for z in r])) * 100 if r else np.nan
        rmsk = np.sqrt(np.mean([z["residuum_clip"] ** 2 for z in r])) * 100 if r else np.nan
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        bild = f"{video[0]['x']:6.2f}" if video else "    - "
        t.append(f"  {c['clip']:<6} {c['theta']:5.1f}   {bild}           {xl:6.2f} ± {sl:.2f} ± {ss:.2f}     "
                 f"{np.degrees(info['eps'][c['clip']]):+5.2f}   {info['alpha'][c['clip']]:+.4f}     "
                 f"{rms:5.1f} / {rmsk:4.1f}")
    return "\n".join(t)


def speichere_kurven(m, clips, pfad):
    """Eine Zeile pro gemessenem Winkel: Düse und z = z_d + k1·Δx + k2·Δx² + k3·Δx³."""
    import csv
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["theta_gemessen_grad", "clips", "x_duese", "z_duese", "k1", "k2", "k3",
                    "aufprall_modell_m", "sigma_kurve_m", "sigma_schuss_m"])
        for th, gruppe in basis.winkel_gruppen(clips).items():
            xd, zd = m.duese(th)
            a, b = m.ab(th)
            xl, sl = m.aufprall(th)
            _, ss = m.aufprall(th, schuss=True)
            w.writerow([f"{th:g}", " ".join(c["clip"] for c in gruppe), f"{xd:.4f}", f"{zd:.4f}",
                        f"{np.tan(np.radians(th) + m.delta_von(th)):.6g}", f"{float(a):.6g}", f"{float(b):.6g}",
                        f"{xl:.4f}", f"{sl:.4f}", f"{ss:.4f}"])


# ---------------------------------------------------------------------------
# Modellvergleich: leave-one-run-out
# ---------------------------------------------------------------------------

def modell_varianten(mit_video_aufprall=True, mit_log_aufprall=False):
    """Name -> (Kurzname, Fit-Funktion(clips) -> Modell)."""
    kw = {"mit_video_aufprall": mit_video_aufprall, "mit_log_aufprall": mit_log_aufprall}
    return {
        "A": ("frei, Parabel (9 Param.)",
              lambda cl: basis.fit(cl, 2, 2, mit_log_aufprall=mit_log_aufprall,
                                   mit_video_aufprall=mit_video_aufprall)[0]),
        "A3": ("frei, kubisch, θ² (12 Param.)",
               lambda cl: basis.fit(cl, 3, 2, mit_log_aufprall=mit_log_aufprall,
                                    mit_video_aufprall=mit_video_aufprall)[0]),
        "A3'": ("frei, kubisch, θ¹ (8 Param.)",
                lambda cl: basis.fit(cl, 3, 1, mit_log_aufprall=mit_log_aufprall,
                                     mit_video_aufprall=mit_video_aufprall)[0]),
        "B": ("Geometrie, Parabel", lambda cl: fit_geometrie(cl, 2, 0, **kw)[0]),
        "C": ("Geometrie, kubisch", lambda cl: fit_geometrie(cl, 3, 0, **kw)[0]),
        "C1": ("Geometrie, kubisch + t-Korrektur", lambda cl: fit_geometrie(cl, 3, 1, **kw)[0]),
        "C2": ("Geometrie, kubisch + t, t²", lambda cl: fit_geometrie(cl, 3, 2, **kw)[0]),
        "D": ("Geometrie, kubisch, δ(θ) quadr.", lambda cl: fit_geometrie(cl, 3, 0, grad_delta=2, **kw)[0]),
        "D2": ("Geometrie, kub. + t, t², δ(θ) quadr.",
               lambda cl: fit_geometrie(cl, 3, 2, grad_delta=2, **kw)[0]),
    }


def winkel_schaetzen(m, c, lo, hi):
    """Welchen Winkel hätte man dem Modell geben müssen, um diesen Schuss zu treffen?
    Punkte ab 0,5 m vor der Düse (nahe Punkte tragen keine Winkelinformation) + Aufprall."""
    d = [p for p in c["punkte"] if p["typ"] == "duese"]
    x0 = d[0]["x"] if d else 0.0
    pts = [p for p in c["punkte"] if (p["typ"] == "strahl" and p["x"] > x0 + 0.5) or p["typ"] == "aufprall"]
    if not pts:
        return np.nan
    x = np.array([p["x"] for p in pts])
    z = np.array([p["z"] for p in pts])
    sx = np.array([np.hypot(p["sx"], c["sigma_aufprall"]) if p["typ"] == "aufprall" else p["sx"] for p in pts])
    sz = np.array([p["sz"] for p in pts])
    st = np.atleast_1d(m.steigung(x, c["theta"]))
    s = np.sqrt(sz ** 2 + (st * sx) ** 2)
    r = minimize_scalar(lambda th: np.sum(((z - m.z(x, th)) / s) ** 2), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-3})
    return float(r.x)


def leave_one_run_out(clips, fitter):
    """Für jeden Lauf: Fit ohne ihn, Vorhersage für seine Clips. -> Liste von dicts."""
    ergebnis = []
    for lauf in sorted({c["messung"] for c in clips}):
        train = [c for c in clips if c["messung"] != lauf]
        test = [c for c in clips if c["messung"] == lauf]
        m = fitter(train)
        ths = [c["theta"] for c in train]
        lo, hi = min(ths) - 5, max(ths) + 5
        for c in test:
            video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
            xl, _ = m.aufprall(c["theta"])
            th_hat = winkel_schaetzen(m, c, lo, hi)
            ergebnis.append({"clip": c["clip"], "lauf": lauf, "theta": c["theta"],
                             "aufprall_fehler": (xl - video[0]["x"]) if video else np.nan,
                             "winkel_fehler": c["theta"] - th_hat})
    return ergebnis


def kennzahlen(liste):
    lf = np.array([e["aufprall_fehler"] for e in liste], float)
    wf = np.array([e["winkel_fehler"] for e in liste], float)
    lf, wf = lf[np.isfinite(lf)], wf[np.isfinite(wf)]
    return {"aufprall_rms_cm": np.sqrt(np.mean(lf ** 2)) * 100, "aufprall_mittel_cm": np.mean(lf) * 100,
            "aufprall_max_cm": np.max(np.abs(lf)) * 100,
            "winkel_rms": np.sqrt(np.mean(wf ** 2)), "winkel_mittel": np.mean(wf), "winkel_max": np.max(np.abs(wf))}


def in_sample(clips, m):
    """Gleiche Kennzahlen, aber mit dem Fit auf allen Daten (zum Vergleich mit leave-one-run-out)."""
    ths = [c["theta"] for c in clips]
    out = []
    for c in clips:
        video = [p for p in c["punkte"] if p["typ"] == "aufprall"]
        xl, _ = m.aufprall(c["theta"])
        out.append({"aufprall_fehler": (xl - video[0]["x"]) if video else np.nan,
                    "winkel_fehler": c["theta"] - winkel_schaetzen(m, c, min(ths) - 5, max(ths) + 5)})
    return out


def vergleich(clips, varianten):
    """-> (Text, {Name: Modell auf allen Daten}, {Name: (zeilen)})"""
    t, modelle, loro = [], {}, {}
    t.append("Modellvergleich. Hauptkriterium: leave-one-run-out (Fit auf 2 Läufen, Vorhersage des 3.)")
    t.append("  Aufprall: Modell-Landepunkt beim gemessenen θ − geklickter Landepunkt")
    t.append("  Winkel:   gemessenes θ − θ, mit dem das Modell die geklickten Punkte des Schusses am besten trifft")
    t.append("            (= Zielfehler in Grad, wenn man mit diesem Modell auf diese Punkte gezielt hätte)")
    t.append("")
    kopf = (f"{'Modell':<38} {'Param':>5}  {'δ [°]':>6}  {'σ_ε [°]':>7}  {'σ_α':>6}  {'f':>5} | "
            f"{'LORO Aufprall cm':>17}  {'LORO Winkel °':>14} | {'in-sample Aufpr.':>16}  {'Winkel':>6}")
    t.append(kopf)
    t.append(f"{'':<38} {'':>5}  {'':>6}  {'':>7}  {'':>6}  {'':>5} | {'RMS  mittel  max':>17}  "
             f"{'RMS  mittel':>14} | {'RMS cm':>16}  {'RMS °':>6}")
    t.append("-" * len(kopf))
    for name, (kurz, fitter) in varianten.items():
        m = fitter(clips)
        modelle[name] = m
        lo = leave_one_run_out(clips, fitter)
        loro[name] = lo
        k = kennzahlen(lo)
        ki = kennzahlen(in_sample(clips, m))
        if isinstance(m, GeometrieModell):
            extra = (f"{np.degrees(m.delta_von(10.0)):+6.2f}  {np.degrees(m.sigma_eps):7.2f}  {m.sigma_alpha:6.4f}  "
                     f"{m.f_mess:5.2f}")
        else:
            extra = f"{'-':>6}  {'-':>7}  {'-':>6}  {np.sqrt(m.chi2_red):5.2f}"
        t.append(f"{name + ' ' + kurz:<38} {len(m.param):>5}  {extra} | "
                 f"{k['aufprall_rms_cm']:5.0f} {k['aufprall_mittel_cm']:+6.0f} {k['aufprall_max_cm']:4.0f}  "
                 f"{k['winkel_rms']:6.2f} {k['winkel_mittel']:+6.2f} | {ki['aufprall_rms_cm']:16.0f}  "
                 f"{ki['winkel_rms']:6.2f}")
    t.append("")
    t.append("f: Skalierung der Klick-σ (Modelle A: √χ²_red, ohne Streuung von Schuss zu Schuss).")
    t.append("δ-Spalte: bei θ = 10° (bei D/D2 hängt δ von θ ab).")
    t.append("")
    t.append("Strahlbeginn (Fit auf allen Daten): RMS der Punkte bis 1,5 m nach der Düse, und")
    t.append("Strahlrichtung an der Düse − gemessenes θ [°] (Düse aus dem Düsenmodell):")
    winkel = [-3, 2, 6, 11, 15, 20, 24]
    duese_p, _ = duese_fit(clips)
    t.append(f"  {'Modell':<6} {'RMS ≤1,5 m [cm]':>15}   " + "  ".join(f"{w:>5}°" for w in winkel))
    for name, m in modelle.items():
        res = []
        for c in clips:
            d = [p for p in c["punkte"] if p["typ"] == "duese"]
            x0 = d[0]["x"] if d else 0.0
            for p in c["punkte"]:
                if p["typ"] in ("duese", "strahl") and p["x"] <= x0 + 1.5:
                    res.append(p["z"] - m.z(p["x"], c["theta"])[0])
        rms = np.sqrt(np.mean(np.square(res))) * 100
        aus = []
        for w in winkel:
            xd, _ = duese_position(duese_p, w)
            aus.append(np.degrees(np.arctan(np.atleast_1d(m.steigung(xd, w))[0])) - w)
        t.append(f"  {name:<6} {rms:15.1f}   " + "  ".join(f"{v:+6.1f}" for v in aus))
    t.append("Pro Lauf (LORO, RMS Aufprall cm / RMS Winkel °):")
    laeufe = sorted({e["lauf"] for e in next(iter(loro.values()))})
    for name in varianten:
        teile = []
        for l in laeufe:
            k = kennzahlen([e for e in loro[name] if e["lauf"] == l])
            teile.append(f"Lauf {l}: {k['aufprall_rms_cm']:4.0f} / {k['winkel_rms']:4.2f}")
        t.append(f"  {name:<3} " + "   ".join(teile))
    t.append("")
    t.append("Pro Clip (LORO Winkelfehler °):")
    t.append("  Clip    θ   " + "  ".join(f"{n:>6}" for n in varianten))
    for i, e in enumerate(loro[next(iter(varianten))]):
        t.append(f"  {e['clip']:<6} {e['theta']:4.0f}  " + "  ".join(
            f"{loro[n][i]['winkel_fehler']:+6.2f}" for n in varianten))
    return "\n".join(t), modelle, loro


def vergleich_plots(clips, modelle, ordner):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stile = {"A": ("--", "A frei Parabel"), "B": (":", "B Geometrie Parabel"),
             "C": ("-", "C Geometrie kubisch"), "C1": ("-.", "C1 kubisch + t"),
             "C2": ((0, (5, 1, 1, 1, 1, 1)), "C2 kubisch + t, t²"),
             "A3": ((0, (8, 2)), "A3 frei kubisch"), "A3'": ((0, (3, 3)), "A3' frei kubisch θ¹"),
             "D": ((0, (1, 1)), "D Geometrie kubisch, δ(θ)"), "D2": ((0, (6, 2, 1, 2)), "D2 kub. + t, t², δ(θ)")}
    gruppen = basis.winkel_gruppen(clips)
    # Übersicht nur mit den wichtigsten Modellen (alle in vergleich_residuen)
    auswahl_m = {n: modelle[n] for n in ("A", "A3", "C", "D2") if n in modelle} or modelle
    ths = list(gruppen)
    auswahl = [ths[i] for i in np.unique(np.linspace(0, len(ths) - 1, 4).round().astype(int))]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, th in enumerate(auswahl):
        farbe = plt.cm.viridis(j / max(1, len(auswahl) - 1))
        for c in gruppen[th]:
            pts = [p for p in c["punkte"] if p["typ"] in ("duese", "strahl", "aufprall")]
            ax.plot([p["x"] for p in pts], [p["z"] for p in pts], "o", ms=3.5, color=farbe)
        for name, m in auswahl_m.items():
            ls, _ = stile.get(name, ("-", name))
            xl, _ = m.aufprall(th)
            x0 = m.x_start(th) if hasattr(m, "x_start") else 0.0
            xs = np.linspace(x0, xl if np.isfinite(xl) else 11, 200)
            ax.plot(xs, m.z(xs, th), linestyle=ls, color=farbe, lw=1.2)
        ax.plot([], [], "o-", color=farbe, label=f"θ = {th:g}° ({', '.join(c['clip'] for c in gruppen[th])})")
    for name in auswahl_m:
        ls, label = stile.get(name, ("-", name))
        ax.plot([], [], linestyle=ls, color="gray", label=label)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("horizontaler Abstand x [m]")
    ax.set_ylabel("Höhe über Boden z [m]")
    ax.set_title("Modellvergleich (Auswahl von Winkeln)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for endung in ("png",):
        fig.savefig(os.path.join(ordner, f"vergleich_bahnen.{endung}"), dpi=150)
    plt.close(fig)

    th_min, th_max = min(ths), max(ths)
    fig, achsen = plt.subplots(len(modelle), 1, figsize=(10, 2.6 * len(modelle)), sharex=True, sharey=True)
    for ax, (name, m) in zip(np.atleast_1d(achsen), modelle.items()):
        for c in clips:
            farbe = plt.cm.viridis((c["theta"] - th_min) / max(1e-9, th_max - th_min))
            pts = [p for p in c["punkte"] if p["typ"] in ("strahl", "aufprall")]
            x = np.array([p["x"] for p in pts])
            r = (np.array([p["z"] for p in pts]) - m.z(x, c["theta"])) * 100
            ax.plot(x, r, "o", ms=3, color=farbe)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("z − Modell [cm]")
        ax.set_title(stile.get(name, ("", name))[1], fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    np.atleast_1d(achsen)[-1].set_xlabel("x [m]  (Farbe = θ)")
    fig.tight_layout()
    for endung in ("png",):
        fig.savefig(os.path.join(ordner, f"vergleich_residuen.{endung}"), dpi=150)
    plt.close(fig)
