"""Gemeinsame Datendateien der Strahl-Auswertung lesen und schreiben.

  referenzen.csv     pfosten,x_m,bemerkung          (Pfosten und ihr horizontaler Abstand)
  messungen.csv      clip,messung,winkel_gemessen_grad,aufprall_m,bemerkung
  punkte/<clip>.json Klickpunkte eines Clips (entzerrte Pixel)
"""

import csv
import glob
import json
import os


def _zahl(wert):
    """'12.5' oder '12,5' -> 12.5, leer -> None."""
    if wert is None or not str(wert).strip():
        return None
    return float(str(wert).strip().replace(",", "."))


def lade_referenzen(pfad):
    """-> Liste von {"name", "x_m", "bemerkung"} in der Reihenfolge der Datei."""
    if not os.path.exists(pfad):
        return []
    ergebnis = []
    with open(pfad, newline="", encoding="utf-8") as f:
        for zeile in csv.DictReader(f):
            name = (zeile.get("pfosten") or "").strip()
            if name:
                ergebnis.append({"name": name, "x_m": _zahl(zeile.get("x_m")),
                                 "bemerkung": (zeile.get("bemerkung") or "").strip()})
    return ergebnis


def lade_messungen(pfad):
    """-> {clip_name_ohne_endung: {"messung", "winkel", "aufprall_m", "bemerkung"}}."""
    if not os.path.exists(pfad):
        return {}
    ergebnis = {}
    with open(pfad, newline="", encoding="utf-8") as f:
        for zeile in csv.DictReader(f):
            clip = os.path.splitext((zeile.get("clip") or "").strip())[0]
            if clip:
                ergebnis[clip] = {"messung": (zeile.get("messung") or "").strip(),
                                  "winkel": _zahl(zeile.get("winkel_gemessen_grad")),
                                  "aufprall_m": _zahl(zeile.get("aufprall_m")),
                                  "bemerkung": (zeile.get("bemerkung") or "").strip()}
    return ergebnis


def clip_info(clip):
    """'15_a' -> (15.0, 'a'); '30' -> (30.0, ''). Soll-Winkel aus dem Dateinamen."""
    teile = clip.split("_", 1)
    try:
        winkel = float(teile[0])
    except ValueError:
        winkel = None
    return winkel, teile[1] if len(teile) > 1 else ""


def lade_punkte(pfad):
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def speichere_punkte(pfad, daten):
    os.makedirs(os.path.dirname(pfad) or ".", exist_ok=True)
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(daten, f, indent=1, ensure_ascii=False)


def alle_punktdateien(ordner):
    return sorted(glob.glob(os.path.join(ordner, "*.json")))
