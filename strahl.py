"""Auswertung der Wasserstrahl-Videos (Schritte 3-6 aus waterjet_trajectory_plan.md).

  python strahl.py klicken 0_a   Schritt 3 (+4 live): Klickfenster für einen Clip
  python strahl.py metrisch      Schritt 4: alle geklickten Clips in Meter -> Tabelle
  python strahl.py fit           Schritte 5+6: gemeinsamer Fit, Unsicherheiten, Plots

Daten:
  komposite/<clip>/<clip>_*.png  aus kompositbild.py (Originalpixel, werden beim Laden entzerrt)
  referenzen.csv           Pfosten und ihr horizontaler Abstand x von der Düse
  messungen.csv            pro Clip: Lauf, gemessener Düsenwinkel, Aufprall laut Protokoll
  punkte/<clip>.json       Klickpunkte (entzerrte Pixel), geschrieben vom Klickfenster
  auswertung/              Tabellen und Plots
"""

import argparse
import csv
import os
import sys

import numpy as np

import daten
import modell

HIER = os.path.dirname(os.path.abspath(__file__))


def befehl_klicken(args):
    from klickfenster import Klickfenster
    clip = os.path.basename(args.clip)
    for endung in ("_overlay.png", "_std.png", "_hintergrund.png", "_hell.png", "_dunkel.png", ".mov", ".json"):
        if clip.endswith(endung):
            clip = clip[: -len(endung)]
    Klickfenster(clip, args.komposite, args.punkte, args.kalibrierung,
                 args.referenzen, args.zeitfenster).zeigen()


def befehl_metrisch(args):
    referenzen = daten.lade_referenzen(args.referenzen)
    messungen = daten.lade_messungen(args.messungen)
    os.makedirs(args.ausgabe, exist_ok=True)
    zeilen = []
    for pfad in daten.alle_punktdateien(args.punkte):
        d = daten.lade_punkte(pfad)
        clip = d["clip"]
        try:
            e, punkte = modell.clip_in_metern(d, referenzen, args.sigma_klick)
        except ValueError as fehler:
            print(f"{clip}: übersprungen ({fehler})")
            continue
        text = f"{clip}: {len(punkte)} Punkte, Referenzen RMS {e.rms_cm():.1f} cm"
        duese = [p for p in punkte if p["typ"] == "duese"]
        if duese:
            text += f", Düse x={duese[0]['x']:.2f} z={duese[0]['z']:.2f} m"
        video = [p for p in punkte if p["typ"] == "aufprall"]
        log = messungen.get(clip, {}).get("aufprall_m")
        if video:
            text += f", Aufprall Video {video[0]['x']:.2f} m"
            if log is not None:
                text += f" / Protokoll {log:.2f} m (Diff {video[0]['x'] - log:+.2f} m)"
        aussen = sum(1 for p in punkte if not p["im_kalibrierten_bereich"])
        if aussen:
            text += f"  [ACHTUNG {aussen} Punkt(e) ausserhalb kalibriert]"
        print(text)
        for i, p in enumerate(punkte):
            zeilen.append({"clip": clip, "winkel_soll": d.get("winkel_soll"), "variante": d.get("variante"),
                           "typ": p["typ"], "nr": i, "u_px": p["u"], "v_px": p["v"],
                           "x_m": round(p["x"], 4), "z_m": round(p["z"], 4),
                           "sx_m": round(p["sx"], 4), "sz_m": round(p["sz"], 4),
                           "im_kalibrierten_bereich": p["im_kalibrierten_bereich"]})
    pfad = os.path.join(args.ausgabe, "strahlpunkte.csv")
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        if zeilen:
            w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
            w.writeheader()
            w.writerows(zeilen)
    print(f"Gespeichert: {pfad} ({len(zeilen)} Punkte)")


def befehl_fit(args):
    referenzen = daten.lade_referenzen(args.referenzen)
    messungen = daten.lade_messungen(args.messungen)
    clips, warnungen = modell.sammle_clips(args.punkte, referenzen, messungen, args.sigma_klick,
                                           args.winkel_offset, args.sigma_aufprall_0, args.sigma_aufprall_30)
    for w in warnungen:
        print("WARNUNG", w)
    if not clips:
        raise SystemExit("Keine Clips für den Fit (Klickpunkte, Referenzen und messungen.csv prüfen).")

    m, zeilen = modell.fit(clips, args.grad_x, args.grad_theta,
                           mit_log_aufprall=args.mit_protokoll_aufprall,
                           mit_video_aufprall=not args.ohne_bild_aufprall)
    text = modell.zusammenfassung(m, clips, zeilen)
    einstellungen = (f"Einstellungen: Klick-σ {args.sigma_klick} px, Winkel absolut "
                     f"(Offset {args.winkel_offset:+g}°), σ Sprühkegel Aufprall "
                     f"{args.sigma_aufprall_0}-{args.sigma_aufprall_30} m, "
                     f"Aufprall geklickt {'nein' if args.ohne_bild_aufprall else 'ja'}, "
                     f"Aufprall Protokoll {'ja' if args.mit_protokoll_aufprall else 'nein (nur Vergleich)'}")
    print(einstellungen)
    print(text)

    os.makedirs(args.ausgabe, exist_ok=True)
    with open(os.path.join(args.ausgabe, "fit_ergebnis.txt"), "w", encoding="utf-8") as f:
        f.write(einstellungen + "\n\n" + text + "\n")
    modell.speichere_tabelle(zeilen, os.path.join(args.ausgabe, "fit_punkte.csv"))
    modell.speichere_kurven(m, clips, os.path.join(args.ausgabe, "kurven.csv"))
    np.savez(os.path.join(args.ausgabe, "fit_modell.npz"), param=m.param, kovarianz=m.kovarianz,
             grad_x=m.grad_x, grad_theta=m.grad_theta, skala_grad=modell.SKALA_GRAD)
    modell.plots(m, clips, zeilen, args.ausgabe)
    print(f"Gespeichert in {args.ausgabe}/: fit_ergebnis.txt, fit_punkte.csv, kurven.csv, fit_modell.npz, "
          f"bahnen/residuen/aufprall.png+pdf")


def main():
    # Windows-Konsolen (cp1252) kennen σ, θ, ² nicht: ersetzen statt abbrechen
    for strom in (sys.stdout, sys.stderr):
        if hasattr(strom, "reconfigure"):
            strom.reconfigure(errors="replace")
    parser =argparse.ArgumentParser(description="Wasserstrahl-Auswertung (Schritte 3-6)")
    parser.add_argument("--punkte", default=os.path.join(HIER, "punkte"), help="Ordner der Klickdateien")
    parser.add_argument("--referenzen", default=os.path.join(HIER, "referenzen.csv"))
    parser.add_argument("--messungen", default=os.path.join(HIER, "messungen.csv"))
    parser.add_argument("--sigma-klick", type=float, default=3.0,
                        help="angenommene Klickgenauigkeit in Pixeln (Standard 3)")
    unter = parser.add_subparsers(dest="befehl", required=True)

    p = unter.add_parser("klicken", help="Klickfenster für einen Clip (z.B. 0_a)")
    p.add_argument("clip")
    p.add_argument("--komposite", default=os.path.join(HIER, "komposite"))
    p.add_argument("--kalibrierung", default=os.path.join(HIER, "kalibrierungen", "blackmagic_4k_stab_aus.json"))
    p.add_argument("--zeitfenster", default=os.path.join(HIER, "zeitfenster.csv"))
    p.set_defaults(funktion=befehl_klicken)

    p = unter.add_parser("metrisch", help="alle Klickpunkte in Meter umrechnen")
    p.add_argument("--ausgabe", default=os.path.join(HIER, "auswertung"))
    p.set_defaults(funktion=befehl_metrisch)

    p = unter.add_parser("fit", help="gemeinsamer Fit + Unsicherheiten + Plots")
    p.add_argument("--ausgabe", default=os.path.join(HIER, "auswertung"))
    p.add_argument("--grad-x", type=int, default=2, help="Polynomgrad in x (Standard 2 = Parabel)")
    p.add_argument("--grad-theta", type=int, default=2, help="Polynomgrad der Koeffizienten in θ")
    p.add_argument("--winkel-offset", type=float, default=0.0,
                   help="wird zum gemessenen Winkel addiert. Standard 0: der Winkel in messungen.csv "
                        "ist absolut (Weltkoordinaten), Drohnenneigung schon enthalten")
    p.add_argument("--sigma-aufprall-0", type=float, default=0.10,
                   help="σ Aufprall (Sprühkegel) bei 0° [m], gilt für geklickten und Protokoll-Aufprall")
    p.add_argument("--sigma-aufprall-30", type=float, default=0.25, help="σ Aufprall (Sprühkegel) bei 30° [m]")
    p.add_argument("--ohne-bild-aufprall", action="store_true",
                   help="geklickten Aufprall (Taste a) nicht im Fit verwenden")
    p.add_argument("--mit-protokoll-aufprall", action="store_true",
                   help="Aufprall aus dem Protokoll zusätzlich im Fit verwenden (Standard: nur Vergleich)")
    p.set_defaults(funktion=befehl_fit)

    args = parser.parse_args()
    args.funktion(args)


if __name__ == "__main__":
    main()
