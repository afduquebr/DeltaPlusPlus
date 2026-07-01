#!/usr/bin/env python3
"""
Load and explore a compressed event JSON file (*.json.gz).

Usage:
    python open_events.py [path/to/file.json.gz]

If no path is given, defaults to the file in the same directory as this script.
"""

import gzip
import json
import sys
from pathlib import Path


def load(path) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def summary(data: dict) -> None:
    meta = data["meta"]
    events = data["events"]
    print(f"File   : {meta['input_file']}")
    print(f"Tree   : {meta['tree_name']}")
    print(f"Events : {meta['events_loaded']} / {meta['total_entries']}")
    print()

    total_particles = sum(ev["num_particles"] for ev in events)
    total_pairs     = sum(len(ev.get("pairs", [])) for ev in events)
    signal_pairs    = sum(
        sum(1 for p in ev.get("pairs", []) if p["is_signal"] == 1)
        for ev in events
    )
    print(f"Total particles : {total_particles}")
    print(f"Total pairs     : {total_pairs}  (signal: {signal_pairs}, bg: {total_pairs - signal_pairs})")
    print()

    ev0 = events[0]
    print(f"--- Event {ev0['event_number']} (first) ---")
    print(f"  b        = {ev0['impact_parameter_fm']} fm")
    print(f"  psi_RP   = {ev0['reaction_plane_angle_rad']} rad")
    print(f"  N_part   = {ev0['num_particles']}")
    gen = ev0["generator_info"]
    print(f"  generator: {gen['generator']}  {gen['projectile']} + {gen['target']}")
    print()

    print("  First particle:")
    p = ev0["particles"][0]
    mom = p["momentum"]
    print(f"    PDG={p['pdg']}  status={p['status']}")
    print(f"    p=({mom['px_GeV']:.4f}, {mom['py_GeV']:.4f}, {mom['pz_GeV']:.4f}) GeV  E={mom['energy_GeV']:.4f} GeV")
    print(f"    pt={p['pt_GeV']} GeV  eta={p['eta']}  phi={p['phi_rad']} rad")

    if ev0.get("pairs"):
        print()
        print("  First pair:")
        pair = ev0["pairs"][0]
        print(f"    proton={pair['proton_idx']}  pion={pair['pion_idx']}  signal={pair['is_signal']}")
        print(f"    M={pair['inv_mass_GeV']} GeV  pt={pair['pair_pt_GeV']} GeV  y={pair['pair_rapidity']}")


def get_event(data: dict, number: int):
    for ev in data["events"]:
        if ev["event_number"] == number:
            return ev
    return None


def get_particles(event: dict, pdg=None) -> list:
    particles = event["particles"]
    if pdg is not None:
        particles = [p for p in particles if p["pdg"] == pdg]
    return particles


def get_signal_pairs(event: dict) -> list[dict]:
    return [p for p in event.get("pairs", []) if p["is_signal"] == 1]


if __name__ == "__main__":
    default = Path(__file__).parent / "AuAu_1230MeV_1000evts_1.json.gz"
    path = sys.argv[1] if len(sys.argv) > 1 else default

    print(f"Loading {path} ...")
    data = load(path)
    print("Done.\n")
    summary(data)
