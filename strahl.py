"""Auswertung der Wasserstrahl-Videos (Schritte 3-6 aus waterjet_trajectory_plan.md).

  python strahl.py klicken 0_a   Schritt 3 (+4 live): Klickfenster für einen Clip
  python strahl.py metrisch      Schritt 4: alle geklickten Clips in Meter -> Tabelle
  python strahl.py fit           Schritte 5+6: gemeinsamer Fit (Standard: Modell E0, einfach.py),
                                 Unsicherheiten, Plots -> auswertung/*_einfach.*
  python strahl.py vergleich     alle Modelle, leave-one-run-out

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

# Düse der Drohne (Modell E). Überschreibbar mit --duese-L / --duese-h.
DUESE_L = 0.81   # [m] Drehpunkt -> Düsenaustritt entlang der Düsenachse. None: L aus den Klicks fitten
DUESE_H = 0.0    # [m] Versatz des Austritts quer zur Achse (reale Drohne: 0)


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


def _clips(args):
    referenzen = daten.lade_referenzen(args.referenzen)
    messungen = daten.lade_messungen(args.messungen)
    clips, warnungen = modell.sammle_clips(args.punkte, referenzen, messungen, args.sigma_klick,
                                           args.winkel_offset, args.sigma_aufprall_0, args.sigma_aufprall_30)
    for w in warnungen:
        print("WARNUNG", w)
    if not clips:
        raise SystemExit("Keine Clips für den Fit (Klickpunkte, Referenzen und messungen.csv prüfen).")
    return clips


def _einstellungen(args):
    return (f"Einstellungen: Klick-σ {args.sigma_klick} px, Winkel absolut "
            f"(Offset {args.winkel_offset:+g}°), σ Sprühkegel Aufprall "
            f"{args.sigma_aufprall_0}-{args.sigma_aufprall_30} m, "
            f"Aufprall geklickt {'nein' if args.ohne_bild_aufprall else 'ja'}, "
            f"Aufprall Protokoll {'ja' if args.mit_protokoll_aufprall else 'nein (nur Vergleich)'}")


def befehl_fit(args):
    if args.modell == "geometrie":
        return befehl_fit_geometrie(args)
    if args.modell == "einfach":
        return befehl_fit_einfach(args)
    clips = _clips(args)
    args.grad_x = 2 if args.grad_x is None else args.grad_x
    args.grad_theta = 2 if args.grad_theta is None else args.grad_theta
    m, zeilen = modell.fit(clips, args.grad_x, args.grad_theta,
                           mit_log_aufprall=args.mit_protokoll_aufprall,
                           mit_video_aufprall=not args.ohne_bild_aufprall)
    text = modell.zusammenfassung(m, clips, zeilen)
    einstellungen = _einstellungen(args)
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
          f"bahnen/residuen/aufprall.png")


def befehl_fit_geometrie(args):
    import geometrie
    clips = _clips(args)
    grad_x = 3 if args.grad_x is None else args.grad_x
    grad_theta = 0 if args.grad_theta is None else args.grad_theta
    m, zeilen, info = geometrie.fit_geometrie(
        clips, grad_x, grad_theta, ursprung=args.ursprung,
        mit_video_aufprall=not args.ohne_bild_aufprall, mit_log_aufprall=args.mit_protokoll_aufprall,
        zufall=not args.ohne_streuung, sigma_eps_grad=args.sigma_theta, sigma_alpha=args.sigma_kruemmung,
        grad_delta=args.grad_delta)
    einstellungen = _einstellungen(args)
    text = geometrie.zusammenfassung(m, clips, zeilen, info)
    print(einstellungen)
    print(text)
    os.makedirs(args.ausgabe, exist_ok=True)
    with open(os.path.join(args.ausgabe, "fit_ergebnis_geometrie.txt"), "w", encoding="utf-8") as f:
        f.write(einstellungen + "\n\n" + text + "\n")
    modell.speichere_tabelle(zeilen, os.path.join(args.ausgabe, "fit_punkte_geometrie.csv"))
    geometrie.speichere_kurven(m, clips, os.path.join(args.ausgabe, "kurven_geometrie.csv"))
    m.speichere(os.path.join(args.ausgabe, "fit_modell_geometrie.npz"),
                os.path.join(args.ausgabe, "fit_modell_geometrie.json"))
    modell.plots(m, clips, zeilen, args.ausgabe, "_geometrie")
    print(f"Gespeichert in {args.ausgabe}/: fit_ergebnis_geometrie.txt, fit_punkte_geometrie.csv, "
          f"kurven_geometrie.csv, fit_modell_geometrie.npz/.json, bahnen/residuen/aufprall_geometrie.png")


def _duese_masse(args):
    """(L, h); L = None: L aus den Klicks fitten (h fest)."""
    return (args.duese_L, args.duese_h)


def befehl_fit_einfach(args):
    import einfach
    clips = _clips(args)
    masse = _duese_masse(args)
    grad_theta = 2 if args.grad_theta is None else args.grad_theta
    m, zeilen, info = einfach.fit_einfach(clips, grad_theta, args.grad_d, duese_masse=masse,
                                          mit_video_aufprall=not args.ohne_bild_aufprall,
                                          mit_log_aufprall=args.mit_protokoll_aufprall)
    einstellungen = _einstellungen(args)
    text = einfach.duese_vergleich(clips, masse) + "\n\n" + einfach.zusammenfassung(m, clips, zeilen, info)
    zusatz = {}
    if not args.ohne_ballistik:   # Extrapolation unter θ_min für zielmodell.py (modell_E_einfach.md, Abschnitt 10)
        import ballistik
        _, block, text_b = ballistik.fit_fuer_ziel(clips, m.duese_param)
        zusatz["ballistik"] = block
        text += "\n\n" + text_b
    print(einstellungen)
    print(text)
    os.makedirs(args.ausgabe, exist_ok=True)
    with open(os.path.join(args.ausgabe, "fit_ergebnis_einfach.txt"), "w", encoding="utf-8") as f:
        f.write(einstellungen + "\n\n" + text + "\n")
    modell.speichere_tabelle(zeilen, os.path.join(args.ausgabe, "fit_punkte_einfach.csv"))
    einfach.speichere(m, os.path.join(args.ausgabe, "fit_modell_einfach.json"), zusatz)
    modell.plots(m, clips, zeilen, args.ausgabe, "_einfach")
    print(f"Gespeichert in {args.ausgabe}/: fit_ergebnis_einfach.txt, fit_punkte_einfach.csv, "
          f"fit_modell_einfach.json, bahnen/residuen/aufprall_einfach.png")


def befehl_vergleich(args):
    import einfach
    import geometrie
    clips = _clips(args)
    masse = _duese_masse(args)
    varianten = geometrie.modell_varianten(mit_video_aufprall=not args.ohne_bild_aufprall,
                                           mit_log_aufprall=args.mit_protokoll_aufprall, duese_masse=masse)
    text, modelle, _ = geometrie.vergleich(clips, varianten)
    text = einfach.duese_vergleich(clips, masse) + "\n\n" + text
    einstellungen = _einstellungen(args)
    print(einstellungen)
    print(text)
    os.makedirs(args.ausgabe, exist_ok=True)
    with open(os.path.join(args.ausgabe, "vergleich.txt"), "w", encoding="utf-8") as f:
        f.write(einstellungen + "\n\n" + text + "\n")
    geometrie.vergleich_plots(clips, modelle, args.ausgabe)
    print(f"Gespeichert in {args.ausgabe}/: vergleich.txt, vergleich_bahnen/vergleich_residuen.png")


def befehl_ballistik_test(args):
    import ballistik
    clips = _clips(args)
    text, modelle, _, bestanden, val = ballistik.validierung(
        clips, _duese_masse(args), durchfluss_l_min=args.durchfluss_l_min, duesen_d_mm=args.duesen_d_mm,
        grad_k=args.grad_k, train_gruppen=tuple(args.train.split(",")), val_gruppen=tuple(args.val.split(",")),
        final_gruppen=tuple(args.endsatz.split(",")) if args.endsatz else None, mit_weg=args.mit_weg)
    name = "ballistik_test" + (f"_k{args.grad_k}" if args.grad_k else "") + ("_weg" if args.mit_weg else "")
    if (args.train, args.val) != ("10,15,20", "5,0"):
        name += f"_train{args.train.replace(',', '-')}_val{args.val.replace(',', '-')}"
    einstellungen = _einstellungen(args)
    print(einstellungen)
    print(text)
    os.makedirs(args.ausgabe, exist_ok=True)
    with open(os.path.join(args.ausgabe, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(einstellungen + "\n\n" + text + "\n")
    untertitel = ("ein v0 und ein k für alle Winkel" if args.grad_k == 0 else
                  f"ein v0 für alle Winkel, Widerstand k(θ) Polynom Grad {args.grad_k} in θ")
    if args.mit_weg:
        untertitel += ", Widerstand wächst mit der Flugstrecke: k = k0 + k_s·s"
    ballistik.plot_validierung(modelle, val, os.path.join(args.ausgabe, name + ".png"), untertitel,
                               f"Ballistik-Validierung: Fit auf Clips {args.train.replace(',', '/')}, "
                               f"Vorhersage der flacheren Clips {args.val.replace(',', ' und ')}")
    print(f"Gespeichert in {args.ausgabe}/: {name}.txt, {name}.png")


def befehl_winkel(args):
    import geometrie
    m = geometrie.lade_geometrie(args.modell_datei)
    e = m.winkel_fuer_ziel(args.x, args.z, bezug=args.bezug)
    print(f"Ziel x = {args.x} m, z = {args.z} m (Bezug {args.bezug}) -> im Versuch x = {e['x_boden']:.2f} m, "
          f"z = {e['z_boden']:.2f} m")
    if e["grund"] != "ok":
        print(f"nicht erreichbar: {e['grund']}")
        return
    print(f"θ absolut = {e['theta']:.2f}° ± {e['sigma_theta']:.2f}° "
          f"(σ_z neuer Schuss {e['sigma_z'] * 100:.0f} cm, ∂z/∂θ = {e['dz_dtheta'] * 100:.1f} cm/°)")


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

    def daten_optionen(p):
        p.add_argument("--ausgabe", default=os.path.join(HIER, "auswertung"))
        p.add_argument("--winkel-offset", type=float, default=0.0,
                       help="wird zum gemessenen Winkel addiert. Standard 0: der Winkel in messungen.csv "
                            "ist absolut (Weltkoordinaten), Drohnenneigung schon enthalten")
        p.add_argument("--sigma-aufprall-0", type=float, default=0.10,
                       help="σ Aufprall (Sprühkegel) bei 0° [m], gilt für geklickten und Protokoll-Aufprall")
        p.add_argument("--sigma-aufprall-30", type=float, default=0.25,
                       help="σ Aufprall (Sprühkegel) bei 30° [m]")
        p.add_argument("--ohne-bild-aufprall", action="store_true",
                       help="geklickten Aufprall (Taste a) nicht im Fit verwenden")
        p.add_argument("--mit-protokoll-aufprall", action="store_true",
                       help="Aufprall aus dem Protokoll zusätzlich im Fit verwenden (Standard: nur Vergleich)")
        p.add_argument("--duese-L", type=float, default=DUESE_L,
                       help="Modell E: Drehpunkt -> Düsenaustritt entlang der Düsenachse [m] "
                            f"(Standard DUESE_L = {DUESE_L}; None: aus den Klicks gefittet)")
        p.add_argument("--duese-h", type=float, default=DUESE_H,
                       help="Modell E: Versatz des Austritts quer zur Achse [m], > 0 = bei θ=0 über dem Drehpunkt "
                            f"(Standard DUESE_H = {DUESE_H})")

    p = unter.add_parser("fit", help="gemeinsamer Fit + Unsicherheiten + Plots")
    daten_optionen(p)
    p.add_argument("--ohne-ballistik", action="store_true",
                   help="einfach: keine ballistische Erweiterung unter θ_min ins JSON schreiben")
    p.add_argument("--grad-d", type=int, default=-1,
                   help="einfach: Richtungskorrektur d(t), Grad in t (Standard -1 = keine, exakt tan θ)")
    p.add_argument("--modell", choices=("einfach", "frei", "geometrie"), default="einfach",
                   help="einfach (Standard, Modell E0): kubisches Polynom ab der Düse, Startrichtung tan θ, "
                        "siehe einfach.py und modell_E_einfach.md. "
                        "frei: Polynom in x mit Koeffizienten polynomial in θ (Modelle A/A3). "
                        "geometrie: Kurve ab der Düse mit tan(θ+δ) (Modell D2), siehe geometrie.py")
    p.add_argument("--grad-x", type=int, default=None,
                   help="Polynomgrad in x (Standard: frei 2 = Parabel, geometrie 3 = kubisch)")
    p.add_argument("--grad-theta", type=int, default=None,
                   help="Grad in θ (Standard: einfach 2 = c2, c3 quadratisch in t; frei 2; "
                        "geometrie 0 = nur sec²θ/sec³θ, 1 = plus Korrektur in t)")
    p.add_argument("--grad-delta", type=int, default=0,
                   help="geometrie: Winkeloffset δ(θ) polynomial in t (Standard 0 = ein festes δ, 2 = Modell D)")
    p.add_argument("--ursprung", choices=("fit", "klick"), default="fit",
                   help="geometrie: Düsenposition aus dem Modell Drehpunkt+Arm (Standard) oder geklickt pro Clip")
    p.add_argument("--ohne-streuung", action="store_true",
                   help="geometrie: keine Streuung von Schuss zu Schuss (Zufallseffekte) modellieren")
    p.add_argument("--sigma-theta", type=float, default=None,
                   help="geometrie: σ_ε [°] fest vorgeben statt schätzen")
    p.add_argument("--sigma-kruemmung", type=float, default=None,
                   help="geometrie: σ_α [1/m] fest vorgeben statt schätzen")
    p.set_defaults(funktion=befehl_fit)

    p = unter.add_parser("vergleich", help="Modelle A/B/C vergleichen, leave-one-run-out")
    daten_optionen(p)
    p.set_defaults(funktion=befehl_vergleich)

    p = unter.add_parser("ballistik-test",
                         help="Ballistikmodell (v0, k) auf Clips 10/15/20 fitten, Clips 5 und 0 vorhersagen")
    daten_optionen(p)
    p.add_argument("--durchfluss-l-min", type=float, default=None,
                   help="Pumpendurchfluss [l/min], nur für die Plausibilität von v0 (Q/A)")
    p.add_argument("--duesen-d-mm", type=float, default=None, help="Düsendurchmesser [mm], nur für Q/A")
    p.add_argument("--train", default="10,15,20", help="Trainingsclips (Clipnamen ohne _a/_b, Komma-getrennt)")
    p.add_argument("--val", default="5,0",
                   help="Validierungsclips, vom kleinsten zum grössten Abstand zum Training (Standard 5,0)")
    p.add_argument("--endsatz", default=None,
                   help="zusätzlich auf diesen Clips fitten und mit dem Training vergleichen (z.B. 0,5,10,15)")
    p.add_argument("--mit-weg", action="store_true",
                   help="Widerstand wächst linear mit der Flugstrecke s ab der Düse: k = k0 + k_s·s (Strahlzerfall)")
    p.add_argument("--grad-k", type=int, default=0,
                   help="Widerstand k(θ) als Polynom in θ/30° (Standard 0 = ein k für alle Winkel, "
                        "1 = linear). v0 ist immer eine Konstante für alle Winkel")
    p.set_defaults(funktion=befehl_ballistik_test)

    p = unter.add_parser("winkel", help="nötiger Düsenwinkel für ein Ziel (Geometriemodell)")
    p.add_argument("x", type=float, help="horizontaler Abstand des Ziels [m]")
    p.add_argument("z", type=float, help="Höhe des Ziels [m]")
    p.add_argument("--bezug", choices=("drehpunkt", "boden"), default="drehpunkt",
                   help="drehpunkt: relativ zum Servo-Drehpunkt (Standard); boden: Versuchskoordinaten")
    p.add_argument("--modell-datei", default=os.path.join(HIER, "auswertung", "fit_modell_geometrie.npz"))
    p.set_defaults(funktion=befehl_winkel)

    args = parser.parse_args()
    args.funktion(args)


if __name__ == "__main__":
    main()
