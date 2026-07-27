#!/usr/bin/env python3
"""
Diagnose whether is_signal labels (proton/pion mate_index matching, see
convert_RootDict.py:159) are inflated by a sentinel/default mate_index value
shared by particles that have no real tracked decay partner.

If most non-decay particles share one default mate_index (commonly 0 or -1
in these UrQMD/ROOT-derived formats), every proton-with-no-real-mate paired
with every pion-with-no-real-mate gets incorrectly labeled is_signal=1 —
purely because sentinel == sentinel, not because of real physics. That would
explain both an inflated signal fraction and a label set too noisy to learn.

Usage:
    python src/diagnose_labels.py --data_dir <path.json.gz> [--max_events N]
"""

import argparse
import gzip
from collections import Counter

import ijson

PDG_PROTON = 2212
PDG_PION = 211


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to .json.gz file")
    parser.add_argument("--max_events", type=int, default=20_000,
                         help="Stop after this many events (0 = no limit)")
    args = parser.parse_args()

    mate_counts_all    = Counter()
    mate_counts_proton = Counter()
    mate_counts_pion   = Counter()
    n_particles = 0
    n_protons   = 0
    n_pions     = 0

    n_pairs   = 0
    n_signal  = 0
    signal_mate_value_counts = Counter()  # mate_index value -> # signal pairs matched via it

    n_events = 0
    with gzip.open(args.data_dir, "rt", encoding="utf-8") as f:
        for event in ijson.items(f, "events.item"):
            particles = event["particles"]
            for p in particles:
                n_particles += 1
                mv = p["mate_index"]
                mate_counts_all[mv] += 1
                if p["pdg"] == PDG_PROTON:
                    n_protons += 1
                    mate_counts_proton[mv] += 1
                elif p["pdg"] == PDG_PION:
                    n_pions += 1
                    mate_counts_pion[mv] += 1

            for pair in event.get("pairs", []):
                n_pairs += 1
                if pair["is_signal"]:
                    n_signal += 1
                    p_idx, pi_idx = pair["proton_idx"], pair["pion_idx"]
                    if p_idx < len(particles) and pi_idx < len(particles):
                        mv = particles[p_idx]["mate_index"]
                        signal_mate_value_counts[mv] += 1

            n_events += 1
            if args.max_events and n_events >= args.max_events:
                break

    print(f"File               : {args.data_dir}")
    print(f"Events processed   : {n_events:,}")
    print(f"Particles seen     : {n_particles:,}  (protons {n_protons:,}, pions {n_pions:,})")
    print(f"Pairs seen         : {n_pairs:,}")
    print(f"Signal pairs       : {n_signal:,}  ({100 * n_signal / max(n_pairs, 1):.3f}%)")

    print("\nTop 5 most common mate_index values (ALL particles):")
    for val, cnt in mate_counts_all.most_common(5):
        print(f"    mate_index={val!r:>10}  count={cnt:>12,}  ({100 * cnt / max(n_particles,1):.2f}% of all particles)")

    print("\nTop 5 most common mate_index values (protons only):")
    for val, cnt in mate_counts_proton.most_common(5):
        print(f"    mate_index={val!r:>10}  count={cnt:>12,}  ({100 * cnt / max(n_protons,1):.2f}% of protons)")

    print("\nTop 5 most common mate_index values (pions only):")
    for val, cnt in mate_counts_pion.most_common(5):
        print(f"    mate_index={val!r:>10}  count={cnt:>12,}  ({100 * cnt / max(n_pions,1):.2f}% of pions)")

    print("\nTop 5 mate_index values responsible for SIGNAL-labeled pairs:")
    for val, cnt in signal_mate_value_counts.most_common(5):
        print(f"    mate_index={val!r:>10}  signal_pairs={cnt:>12,}  ({100 * cnt / max(n_signal,1):.2f}% of all signal pairs)")

    if signal_mate_value_counts:
        top_val, top_cnt = signal_mate_value_counts.most_common(1)[0]
        frac = top_cnt / max(n_signal, 1)
        print(f"\n>>> Single most common mate_index among signal pairs ({top_val!r}) "
              f"accounts for {100 * frac:.1f}% of all signal labels. <<<")
        if frac > 0.5:
            print(">>> This strongly suggests a sentinel-value collision bug, not real physics. <<<")
        else:
            print(">>> Signal labels are spread across many distinct mate_index values — "
                  "less likely to be a sentinel-collision bug. <<<")


if __name__ == "__main__":
    main()
