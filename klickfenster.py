"""Schritt 3: Klickfenster für Strahl-Mittellinie und Referenzen.

Die Kompositbilder (aus kompositbild.py, Originalpixel) werden beim Laden
entzerrt. Alles wird im ENTZERRTEN Bild geklickt und gespeichert, darum sind
gerade Linien (Bodenlinie, Pfosten) auch im Bild gerade.

Tasten:
  d  Düsenspitze (1 Punkt)          s  Strahlpunkte (beliebig viele)
  a  Aufprall im Video (1 Punkt)    g  Bodenlinie (>= 2 Punkte)
  r  Pfosten-Marken: Pfosten aus referenzen.csv der Reihe nach,
     Enter = nächster Pfosten, n = Pfosten nicht sichtbar / überspringen
  m  Bodenmarke (Klebeband, Pylone ...): klicken, dann x in m (Abstand von
     der Düse, horizontal) eingeben
  1/2/3/4  Bild Overlay / Std / Hintergrund / Hell
  z oder rechte Maustaste  letzten Punkt löschen
  x oder Entf  Punkt unter der Maus löschen (egal welcher)
  w  speichern (passiert auch beim Schliessen)
Zoomen:
  - Mausrad: rein/raus um die Mausposition (immer möglich)
  - Knöpfe 'Zoom +' / 'Zoom −' in der Werkzeugleiste: Klick ins Bild zoomt um
    Faktor 2 (rechte Maus = Gegenrichtung); Knopf wieder aus = Punkte setzen
  - Hand der Werkzeugleiste: verschieben; Haus: ganzes Bild
Solange ein Zoom-Knopf, die Lupe oder die Hand aktiv ist, werden Klicks nicht
als Punkte gezählt.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import daten
import ebene
from entzerren import entzerre_punkte, entzerrungs_tabellen, lade_kalibrierung, pruefe_groesse
from kompositbild import lade_zeitfenster

BILDER = [("overlay", "1"), ("std", "2"), ("hintergrund", "3"), ("hell", "4")]
LOESCH_TASTEN = {"x", "delete", "backspace"}
EIGENE_TASTEN = {"d", "s", "a", "g", "r", "m", "n", "z", "w", "1", "2", "3", "4", "enter"} | LOESCH_TASTEN
FARBEN = {"duese": "cyan", "strahl": "yellow", "aufprall": "magenta",
          "boden": "lime", "pfosten_marke": "red", "bodenmarke": "orange"}
MODI = {"d": "duese", "s": "strahl", "a": "aufprall", "g": "boden", "r": "pfosten_marke", "m": "bodenmarke"}
MODUS_TEXT = {None: "keiner (Taste wählen)", "duese": "Düsenspitze", "strahl": "Strahlpunkte",
              "aufprall": "Aufprall", "boden": "Bodenlinie", "pfosten_marke": "Pfosten-Marken",
              "bodenmarke": "Bodenmarken (Klick, dann x eingeben)"}
HILFE = ("d Düse | s Strahl | a Aufprall | g Boden | r Pfosten (Enter weiter, n überspringen) | "
         "m Bodenmarke\n"
         "1/2/3/4 Bild | z / rechte Maus: rückgängig | x / Entf: Punkt unter der Maus löschen | "
         "w speichern | Zoom: Mausrad, oder Knöpfe 'Zoom +/−' oben")
LOESCH_RADIUS_PX = 15      # so nah (Bildschirmpixel) muss die Maus am Punkt sein


def _matplotlib_tasten_freigeben():
    """Standard-Tastenkürzel von matplotlib entfernen, die mit unseren kollidieren."""
    for schluessel in list(plt.rcParams.keys()):
        if schluessel.startswith("keymap."):
            plt.rcParams[schluessel] = [t for t in plt.rcParams[schluessel] if t not in EIGENE_TASTEN]


class Klickfenster:
    def __init__(self, clip, komposite, punkte_ordner, kalibrierung, referenzen_pfad, zeitfenster_pfad):
        self.clip = clip
        self.pfad_json = os.path.join(punkte_ordner, f"{clip}.json")
        self.kalibrierung = kalibrierung
        self.referenzen = daten.lade_referenzen(referenzen_pfad)
        self.zeitfenster = lade_zeitfenster(zeitfenster_pfad).get(f"{clip}.mov")
        self.winkel_soll, self.variante = daten.clip_info(clip)

        K, D, groesse, huelle = lade_kalibrierung(kalibrierung)
        self.bilder = self._lade_bilder(komposite, K, D, groesse)
        self.huelle = self._entzerrte_huelle(huelle, K, D)

        self.punkte = []          # chronologisch: {"typ", "x", "y", ["pfosten"], ["x_m"]}
        self.modus = None
        self.pfosten_index = 0
        self.ebene = None
        self.ebene_fehler = ""
        self.meldung = ""
        self.kuenstler = []
        if os.path.exists(self.pfad_json):
            alt = daten.lade_punkte(self.pfad_json)
            self.punkte = [{k: p[k] for k in ("typ", "x", "y", "pfosten", "x_m") if k in p} for p in alt["punkte"]]
            self.meldung = f"{len(self.punkte)} Punkte aus {self.pfad_json} geladen"

    # ------------------------------------------------------------------ Laden

    def _lade_bilder(self, ordner, K, D, groesse):
        karte1, karte2 = entzerrungs_tabellen(K, D, groesse)
        bilder = {}
        for name, _ in BILDER:
            datei = f"{self.clip}_{name}.png"
            pfad = os.path.join(ordner, self.clip, datei)       # komposite/<clip>/<clip>_<name>.png
            if not os.path.exists(pfad):
                pfad = os.path.join(ordner, datei)              # alte, flache Ablage
            if not os.path.exists(pfad):
                continue
            bild = cv2.imread(pfad)
            pruefe_groesse(bild.shape[1], bild.shape[0], groesse)
            entzerrt = cv2.remap(bild, karte1, karte2, cv2.INTER_CUBIC)
            bilder[name] = cv2.cvtColor(entzerrt, cv2.COLOR_BGR2RGB)
        if not bilder:
            raise SystemExit(f"Keine Kompositbilder für '{self.clip}' in {ordner} gefunden "
                             f"(zuerst kompositbild.py laufen lassen).")
        return bilder

    @staticmethod
    def _entzerrte_huelle(huelle, K, D):
        """Kalibrierten Bereich ins entzerrte Bild übertragen (Kanten verdichtet, da gekrümmt)."""
        if huelle is None:
            return None
        h = huelle.reshape(-1, 2).astype(np.float64)
        dicht = [h[i] + t * (h[(i + 1) % len(h)] - h[i])
                 for i in range(len(h)) for t in np.linspace(0, 1, 40, endpoint=False)]
        return entzerre_punkte(np.array(dicht), K, D).astype(np.float32).reshape(-1, 1, 2)

    def im_bereich(self, x, y):
        return self.huelle is None or cv2.pointPolygonTest(self.huelle, (float(x), float(y)), False) >= 0

    # ------------------------------------------------------------------ Fenster

    def zeigen(self):
        _matplotlib_tasten_freigeben()
        self.fig = plt.figure(figsize=(15, 8.5))
        self.ax = self.fig.add_axes([0.0, 0.0, 0.74, 0.94])
        self.info = self.fig.text(0.75, 0.97, "", va="top", ha="left", family="monospace", fontsize=8.5)
        self.titel = self.fig.text(0.01, 0.985, "", va="top", ha="left", fontsize=10, weight="bold")
        self.hilfe = self.fig.text(0.01, 0.955, HILFE, va="top", ha="left", fontsize=8)

        self.bild_name = next(iter(self.bilder))
        self.im = self.ax.imshow(self.bilder[self.bild_name], interpolation="nearest")
        self.ax.set_axis_off()
        if self.huelle is not None:
            h = self.huelle.reshape(-1, 2)
            self.ax.plot(np.r_[h[:, 0], h[0, 0]], np.r_[h[:, 1], h[0, 1]], "w--", lw=0.8, alpha=0.7)
        self.ax.format_coord = self._koordinaten_text

        self.zoom_werkzeug = None          # None, "rein" oder "raus"
        self._zoom_knoepfe = {}
        self._zoom_knoepfe_anlegen()

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._klick)
        c.mpl_connect("key_press_event", self._taste)
        c.mpl_connect("scroll_event", self._mausrad)
        c.mpl_connect("close_event", lambda ev: self.speichern(nur_wenn_inhalt=True))
        self._aktualisieren()
        plt.show()

    # ------------------------------------------------------------------ Zoom

    def _zoom_knoepfe_anlegen(self):
        """Knöpfe 'Zoom +' / 'Zoom −' in die Werkzeugleiste (Qt), neben die Lupe."""
        leiste = self.fig.canvas.toolbar
        if leiste is None or not hasattr(leiste, "addAction"):
            print("Hinweis: Zoom-Knöpfe nur mit Qt-Oberfläche; Mausrad-Zoom geht trotzdem.")
            return
        aktionen = leiste.actions()
        lupe = getattr(leiste, "_actions", {}).get("zoom")
        davor = None
        if lupe in aktionen and aktionen.index(lupe) + 1 < len(aktionen):
            davor = aktionen[aktionen.index(lupe) + 1]
        for name, text, tipp in (("rein", "Zoom +", "Klick ins Bild vergrössert (rechte Maus: verkleinert)"),
                                 ("raus", "Zoom −", "Klick ins Bild verkleinert (rechte Maus: vergrössert)")):
            aktion = leiste.addAction(text)
            if davor is not None:
                leiste.removeAction(aktion)
                leiste.insertAction(davor, aktion)
            aktion.setCheckable(True)
            aktion.setToolTip(tipp)
            aktion.triggered.connect(lambda an, n=name: self._zoom_werkzeug_setzen(n if an else None))
            self._zoom_knoepfe[name] = aktion
        # eingebaute Lupe/Hand eingeschaltet -> unsere Zoom-Werkzeuge aus
        for name in ("zoom", "pan"):
            eingebaut = getattr(leiste, "_actions", {}).get(name)
            if eingebaut is not None:
                eingebaut.toggled.connect(lambda an: an and self._zoom_werkzeug_setzen(None))

    def _zoom_werkzeug_setzen(self, name):
        self.zoom_werkzeug = name
        for n, aktion in self._zoom_knoepfe.items():
            aktion.setChecked(n == name)
        leiste = self.fig.canvas.toolbar
        modus = getattr(leiste, "mode", "")
        if name is not None and getattr(modus, "value", modus):
            # eingebaute Lupe/Hand ausschalten (gleicher Aufruf schaltet um)
            if str(getattr(modus, "value", modus)).startswith("zoom"):
                leiste.zoom()
            else:
                leiste.pan()
        self.meldung = {None: "", "rein": "Zoom +: ins Bild klicken (rechte Maus = verkleinern). "
                                          "Zum Punkte setzen Knopf wieder aus.",
                        "raus": "Zoom −: ins Bild klicken (rechte Maus = vergrössern). "
                                "Zum Punkte setzen Knopf wieder aus."}[name]
        self._info_text()
        self.fig.canvas.draw_idle()

    def _zoomen(self, x, y, faktor):
        """Um (x, y) zoomen; Faktor > 1 vergrössert. Der Punkt unter der Maus bleibt stehen."""
        leiste = self.fig.canvas.toolbar
        if leiste is not None and hasattr(leiste, "push_current"):
            leiste.push_current()          # damit Zurück/Vor der Werkzeugleiste funktionieren
        hoehe, breite = self.bilder[self.bild_name].shape[:2]
        neue = []
        for (a, b), mitte, (lo, hi) in ((self.ax.get_xlim(), x, (-0.5, breite - 0.5)),
                                        (self.ax.get_ylim(), y, (-0.5, hoehe - 0.5))):
            na, nb = mitte - (mitte - a) / faktor, mitte + (b - mitte) / faktor
            klein, gross = min(na, nb), max(na, nb)
            if gross - klein >= hi - lo:              # nicht weiter raus als das ganze Bild
                klein, gross = lo, hi
            elif klein < lo:
                klein, gross = lo, gross + (lo - klein)
            elif gross > hi:
                klein, gross = klein - (gross - hi), hi
            neue.append((klein, gross) if na < nb else (gross, klein))   # Richtung (y umgekehrt) behalten
        self.ax.set_xlim(neue[0])
        self.ax.set_ylim(neue[1])
        self.fig.canvas.draw_idle()

    def _mausrad(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None:
            self._zoomen(ev.xdata, ev.ydata, 1.5 if ev.button == "up" else 1 / 1.5)

    def _koordinaten_text(self, x, y):
        text = f"entzerrt ({x:.1f}, {y:.1f}) px"
        if self.ebene is not None:
            xm, zm = self.ebene.nach_metern([[x, y]])[0]
            text += f"   x = {xm:.3f} m, z = {zm:.3f} m"
        if not self.im_bereich(x, y):
            text += "   (ausserhalb kalibriert)"
        return text

    # ------------------------------------------------------------------ Eingaben

    def _klick(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        werkzeug = getattr(self.fig.canvas.toolbar, "mode", "")
        if getattr(werkzeug, "value", werkzeug):      # eingebaute Lupe oder Hand aktiv
            return
        if self.zoom_werkzeug is not None:            # unsere Zoom-Knöpfe
            rein = (self.zoom_werkzeug == "rein") == (ev.button == 1)
            self._zoomen(ev.xdata, ev.ydata, 2.0 if rein else 0.5)
            return
        if ev.button == 3:
            self._rueckgaengig()
        elif ev.button == 1:
            self._punkt_hinzufuegen(float(ev.xdata), float(ev.ydata))
        self._aktualisieren()

    def _taste(self, ev):
        t = ev.key
        if t in LOESCH_TASTEN:
            self._punkt_loeschen_bei(ev)
        elif t in MODI:
            self.modus = MODI[t]
            self.meldung = ""
            if self.modus == "pfosten_marke":
                self._pfosten_start()
        elif t in ("enter", "n") and self.modus == "pfosten_marke":
            self._naechster_pfosten()
        elif t in ("1", "2", "3", "4"):
            name = BILDER[int(t) - 1][0]
            if name in self.bilder:
                self.bild_name = name
                self.im.set_data(self.bilder[name])
            else:
                self.meldung = f"Bild '{name}' nicht vorhanden"
        elif t == "z":
            self._rueckgaengig()
        elif t == "w":
            self.speichern()
        else:
            return
        self._aktualisieren()

    def _punkt_hinzufuegen(self, x, y):
        if self.modus is None:
            self.meldung = "Zuerst einen Modus wählen (d, s, a, g, r)"
            return
        punkt = {"typ": self.modus, "x": round(x, 2), "y": round(y, 2)}
        if self.modus in ("duese", "aufprall"):        # nur ein Punkt erlaubt
            self.punkte = [p for p in self.punkte if p["typ"] != self.modus]
        if self.modus == "pfosten_marke":
            pfosten = self._aktueller_pfosten()
            if pfosten is None:
                self.meldung = "Kein Pfosten aktiv (r drücken, referenzen.csv prüfen)"
                return
            punkt["pfosten"] = pfosten["name"]
        if self.modus == "bodenmarke":
            x_m = self._frage_x_m(x, y)
            if x_m is None:
                self.meldung = "Bodenmarke abgebrochen"
                return
            punkt["x_m"] = x_m
        self.punkte.append(punkt)
        self.meldung = "" if self.im_bereich(x, y) else "ACHTUNG: Punkt ausserhalb des kalibrierten Bereichs"

    @staticmethod
    def _beschreibung(p):
        text = MODUS_TEXT.get(p["typ"], p["typ"]).split(" (")[0]
        if "pfosten" in p:
            text += f" ({p['pfosten']})"
        if "x_m" in p:
            text += f" (x = {p['x_m']:g} m)"
        return text

    def _rueckgaengig(self):
        if self.punkte:
            p = self.punkte.pop()
            self.meldung = f"gelöscht: {self._beschreibung(p)}"

    def _punkt_loeschen_bei(self, ev):
        """Den Punkt löschen, der der Maus am nächsten ist (auf dem Bildschirm gemessen)."""
        if ev.inaxes is not self.ax or ev.x is None or not self.punkte:
            self.meldung = "Zum Löschen die Maus auf einen Punkt im Bild halten"
            return
        bildschirm = self.ax.transData.transform([(p["x"], p["y"]) for p in self.punkte])
        abstand = np.hypot(bildschirm[:, 0] - ev.x, bildschirm[:, 1] - ev.y)
        i = int(np.argmin(abstand))
        if abstand[i] > LOESCH_RADIUS_PX:
            self.meldung = "Kein Punkt unter der Maus (näher ranzoomen oder genauer zeigen)"
            return
        p = self.punkte.pop(i)
        self.meldung = f"gelöscht: {self._beschreibung(p)}"

    def _frage_x_m(self, u, v):
        """x der Bodenmarke abfragen (Qt-Dialog, sonst Konsole). Vorschlag aus der Abbildung."""
        bisher = [p["x_m"] for p in self.punkte if p["typ"] == "bodenmarke"]
        if self.ebene is not None:
            vorschlag = float(round(self.ebene.nach_metern([[u, v]])[0][0]))
        else:
            vorschlag = max(bisher) + 1.0 if bisher else 1.0
        frage = "Horizontaler Abstand dieser Bodenmarke von der Düse x [m]:"
        try:
            from matplotlib.backends.qt_compat import QtWidgets
            wert, ok = QtWidgets.QInputDialog.getDouble(self.fig.canvas, "Bodenmarke", frage,
                                                        vorschlag, -100.0, 100.0, 2)
            return float(wert) if ok else None
        except Exception:
            antwort = input(f"{frage} [{vorschlag:g}] ").strip().replace(",", ".")
            try:
                return float(antwort) if antwort else vorschlag
            except ValueError:
                return None

    # ------------------------------------------------------------------ Pfosten

    def _gueltige_pfosten(self):
        return [i for i, r in enumerate(self.referenzen) if r["x_m"] is not None]

    def _aktueller_pfosten(self):
        if 0 <= self.pfosten_index < len(self.referenzen) and self.referenzen[self.pfosten_index]["x_m"] is not None:
            return self.referenzen[self.pfosten_index]
        return None

    def _pfosten_start(self):
        gueltig = self._gueltige_pfosten()
        if not gueltig:
            self.meldung = "referenzen.csv: keine Pfosten mit x_m eingetragen"
            self.modus = None
            return
        fertig = {p.get("pfosten") for p in self.punkte if p["typ"] == "pfosten_marke"}
        offen = [i for i in gueltig if self.referenzen[i]["name"] not in fertig]
        self.pfosten_index = offen[0] if offen else gueltig[0]

    def _naechster_pfosten(self):
        weitere = [i for i in self._gueltige_pfosten() if i > self.pfosten_index]
        if weitere:
            self.pfosten_index = weitere[0]
        else:
            self.meldung = "Alle Pfosten durch."
            self.modus = None

    # ------------------------------------------------------------------ Anzeige

    def _aktualisieren(self):
        try:
            self.ebene = ebene.aus_klickpunkten(self.punkte, self.referenzen)
            self.ebene_fehler = ""
        except (ValueError, np.linalg.LinAlgError) as fehler:
            self.ebene, self.ebene_fehler = None, str(fehler)
        self._zeichnen()
        self._info_text()
        self.fig.canvas.draw_idle()

    def _punkte_vom_typ(self, typ):
        return np.array([(p["x"], p["y"]) for p in self.punkte if p["typ"] == typ]).reshape(-1, 2)

    def _strahl_linie(self):
        """Düse -> Strahlpunkte (nach Abstand von der Düse sortiert) -> Aufprall."""
        duese, strahl, aufprall = (self._punkte_vom_typ(t) for t in ("duese", "strahl", "aufprall"))
        if len(strahl):
            bezug = duese[0] if len(duese) else strahl[np.argmax(strahl[:, 0])]
            strahl = strahl[np.argsort(np.abs(strahl[:, 0] - bezug[0]))]
        return np.vstack([duese, strahl, aufprall])

    def _zeichnen(self):
        for k in self.kuenstler:
            k.remove()
        self.kuenstler = []
        ax, neu = self.ax, self.kuenstler.append
        breite = self.bilder[self.bild_name].shape[1]

        linie = self._strahl_linie()
        if len(linie) >= 2:
            neu(ax.plot(linie[:, 0], linie[:, 1], "-", color="yellow", lw=0.8, alpha=0.8)[0])

        boden = self._punkte_vom_typ("boden")
        bodenlinie = None
        if len(boden) >= 2:
            bodenlinie = ebene.gerade_fit(boden)
            p, d = bodenlinie
            if abs(d[0]) > 1e-6:
                t = (np.array([0, breite]) - p[0]) / d[0]
                neu(ax.plot(p[0] + t * d[0], p[1] + t * d[1], "-", color="lime", lw=0.8)[0])

        for ref in self.referenzen:
            marken = np.array([(p["x"], p["y"]) for p in self.punkte
                               if p["typ"] == "pfosten_marke" and p.get("pfosten") == ref["name"]]).reshape(-1, 2)
            if len(marken) == 0:
                continue
            oben = marken[np.argmin(marken[:, 1])]
            neu(ax.annotate(ref["name"], oben, xytext=(6, -6), textcoords="offset points",
                            color="red", fontsize=8, weight="bold"))
            if len(marken) >= 2 and bodenlinie is not None:
                fuss = ebene.schnittpunkt(ebene.gerade_fit(marken), bodenlinie)
                if fuss is not None:
                    neu(ax.plot([fuss[0], oben[0]], [fuss[1], oben[1]], "-", color="red", lw=0.7)[0])
                    neu(ax.plot(*fuss, "s", mfc="none", mec="lime", ms=7)[0])

        for p in self.punkte:
            if p["typ"] != "bodenmarke":
                continue
            neu(ax.annotate(f"{p.get('x_m', float('nan')):g} m", (p["x"], p["y"]), xytext=(4, 6),
                            textcoords="offset points", color="orange", fontsize=8, weight="bold"))
            if bodenlinie is not None:
                fuss = ebene.projiziere_auf_gerade((p["x"], p["y"]), bodenlinie)
                neu(ax.plot(*fuss, "o", mfc="none", mec="orange", ms=6)[0])

        for typ, farbe in FARBEN.items():
            pts = self._punkte_vom_typ(typ)
            if len(pts):
                marker = "+" if typ in ("strahl", "pfosten_marke", "boden", "bodenmarke") else "x"
                neu(ax.plot(pts[:, 0], pts[:, 1], marker, color=farbe, ms=10, mew=1.3)[0])

        if self.ebene is not None:
            for typ in ("duese", "strahl", "aufprall"):
                for u, v in self._punkte_vom_typ(typ):
                    xm, zm = self.ebene.nach_metern([[u, v]])[0]
                    neu(ax.annotate(f"{xm:.2f}/{zm:.2f}", (u, v), xytext=(5, 5), textcoords="offset points",
                                    color=FARBEN[typ], fontsize=7))

    def _info_text(self):
        modus = MODUS_TEXT[self.modus]
        if self.modus == "pfosten_marke" and self._aktueller_pfosten() is not None:
            r = self._aktueller_pfosten()
            modus += f": {r['name']} (x = {r['x_m']} m) - Marken klicken"
        if self.zoom_werkzeug is not None:
            modus = f"ZOOM {'+' if self.zoom_werkzeug == 'rein' else '−'} aktiv (keine Punkte) - danach: {modus}"
        self.titel.set_text(f"Clip {self.clip}   |   Bild: {self.bild_name}   |   Modus: {modus}")

        anzahl = {t: len(self._punkte_vom_typ(t)) for t in FARBEN}
        zeilen = [f"Clip {self.clip}  (Soll {self.winkel_soll}°, Variante '{self.variante}')",
                  f"Zeitfenster: {self.zeitfenster}", "",
                  f"Düse      {anzahl['duese']}", f"Strahl    {anzahl['strahl']}",
                  f"Aufprall  {anzahl['aufprall']}", f"Boden     {anzahl['boden']}",
                  f"Marken    {anzahl['pfosten_marke']}", f"Bodenmarken {anzahl['bodenmarke']}",
                  "", "Pfosten:"]
        for r in self.referenzen:
            n = sum(1 for p in self.punkte if p.get("pfosten") == r["name"])
            x = "x fehlt!" if r["x_m"] is None else f"x={r['x_m']:g} m"
            zeilen.append(f"  {r['name']:<8} {x:<10} {n} Marken")
        if not self.referenzen:
            zeilen.append("  (referenzen.csv leer)")

        zeilen += ["", "Schritt 4 (Meter):"]
        if self.ebene is None:
            zeilen.append(f"  noch nicht: {self.ebene_fehler}")
        else:
            zeilen.append(f"  Referenz-Residuen RMS {self.ebene.rms_cm():.1f} cm, max {self.ebene.max_cm():.1f} cm")
            for name, h in self.ebene.offsets.items():
                zeilen.append(f"  {name}: unterste Marke {h:.3f} m über Boden")
            for r in self.ebene.residuen:
                if r["art"] == "bodenmarke":
                    zeilen.append(f"  Bodenmarke x={r['x_m']:g} m: dx {r['dx_m'] * 100:+.1f} cm")
            for typ, text in (("duese", "Düse"), ("aufprall", "Aufprall")):
                pts = self._punkte_vom_typ(typ)
                if len(pts):
                    xm, zm = self.ebene.nach_metern(pts[:1])[0]
                    zeilen.append(f"  {text}: x = {xm:.2f} m, z = {zm:.2f} m")

        aussen = sum(1 for p in self.punkte if not self.im_bereich(p["x"], p["y"]))
        if aussen:
            zeilen += ["", f"ACHTUNG: {aussen} Punkt(e) ausserhalb kalibriert"]
        if self.meldung:
            zeilen += ["", self.meldung]
        self.info.set_text("\n".join(zeilen))

    # ------------------------------------------------------------------ Speichern

    def speichern(self, nur_wenn_inhalt=False):
        if nur_wenn_inhalt and not self.punkte and not os.path.exists(self.pfad_json):
            return
        boden = self._punkte_vom_typ("boden")
        bodenlinie = ebene.gerade_fit(boden) if len(boden) >= 2 else None
        pfosten = []
        for r in self.referenzen:
            marken = [(p["x"], p["y"]) for p in self.punkte if p.get("pfosten") == r["name"]]
            eintrag = {"pfosten": r["name"], "x_m": r["x_m"], "bodenpunkt": None}
            if len(marken) >= 2 and bodenlinie is not None:
                fuss = ebene.schnittpunkt(ebene.gerade_fit(marken), bodenlinie)
                eintrag["bodenpunkt"] = None if fuss is None else [round(float(v), 2) for v in fuss]
            if marken:
                pfosten.append(eintrag)

        # Marken-Index k (0 = unterste) je Pfosten
        k_von = {}
        for r in self.referenzen:
            idx = [i for i, p in enumerate(self.punkte) if p.get("pfosten") == r["name"]]
            for k, i in enumerate(sorted(idx, key=lambda i: -self.punkte[i]["y"])):
                k_von[i] = k

        punkte = []
        for i, p in enumerate(self.punkte):
            e = dict(p)
            if i in k_von:
                e["marke_k"] = k_von[i]
            e["im_kalibrierten_bereich"] = bool(self.im_bereich(p["x"], p["y"]))
            punkte.append(e)

        daten.speichere_punkte(self.pfad_json, {
            "clip": self.clip, "winkel_soll": self.winkel_soll, "variante": self.variante,
            "kalibrierung": os.path.basename(self.kalibrierung), "zeitfenster_s": self.zeitfenster,
            "koordinaten": "entzerrte Pixel (Kameramatrix K der Kalibrierung)",
            "boden_linie": None if bodenlinie is None else
            {"punkt": [round(float(v), 2) for v in bodenlinie[0]],
             "richtung": [round(float(v), 6) for v in bodenlinie[1]]},
            "pfosten": pfosten,
            "punkte": punkte,
        })
        self.meldung = f"gespeichert: {self.pfad_json}"
        print(self.meldung)
