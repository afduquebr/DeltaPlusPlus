#!/usr/bin/env python3
"""
Diagnose whether is_signal labels (proton/pion mate_index matching, see
convert_RootDict.py:159) actually correspond to real Delta++ -> p + pi+
decays.

Two independent failure modes are checked:

1. Sentinel collision (false positives): if most non-decay particles share
   one default mate_index (e.g. -1, "no tracked partner"), every
   no-real-mate proton paired with every no-real-mate pion gets incorrectly
   labeled is_signal=1, purely because sentinel == sentinel.

2. Mate reassignment (false negatives): if a pion's mate_index sometimes
   points to something other than its true decay sibling (e.g. a radiated
   photon, if UrQMD's mate bookkeeping gets overwritten by a later
   interaction), real Delta++ pairs would fail the mate match and get
   dumped into the background class (is_signal=0) even though they are
   real signal — contaminating "background" with real-signal kinematics.

Both are checked against a more direct ground truth: two particles that
share the same parent_index, where that parent's pdg is 2224 (Delta++), are
about as directly confirmed as "same decay vertex" as this data allows,
independent of whatever mate_index is actually used for.

Usage:
    python src/diagnose_labels.py --data_dir <path.json.gz> [--max_events N]
"""

import argparse
import gzip
from collections import Counter

import ijson

PDG_PROTON   = 2212
PDG_PION     = 211
PDG_DELTA_PP = 2224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to .json.gz file")
    parser.add_argument("--max_events", type=int, default=20_000,
                         help="Stop after this many events (0 = no limit)")
    args = parser.parse_args()

    mate_counts_all    = Counter()
    mate_counts_proton = Counter()
    mate_counts_pion   = Counter()
    parent_counts_all  = Counter()
    n_particles = 0
    n_protons   = 0
    n_pions     = 0

    n_pairs  = 0
    n_signal = 0
    signal_mate_value_counts = Counter()  # mate_index value -> # signal pairs matched via it
    parent_pdg_counts = Counter()         # pdg of a shared parent, when proton/pion parent_index match

    # Confusion matrix: current mate-based is_signal vs. "shares a real
    # Delta++ parent" ground truth.
    tp = fp = fn = tn = 0

    # Invariant-mass sanity check: real Delta++ decays should cluster near
    # the resonance mass (1.232 GeV/c^2). Track sum/sumsq/count for three
    # groups so we can compare mean +/- std.
    mass_stats = {
        "all_pairs":        [0.0, 0.0, 0],
        "mate_signal":      [0.0, 0.0, 0],
        "parent_delta":     [0.0, 0.0, 0],
    }

    def _accum(key, m):
        s = mass_stats[key]
        s[0] += m
        s[1] += m * m
        s[2] += 1

    n_events = 0
    with gzip.open(args.data_dir, "rt", encoding="utf-8") as f:
        for event in ijson.items(f, "events.item"):
            particles = event["particles"]
            for p in particles:
                n_particles += 1
                mv = p["mate_index"]
                mate_counts_all[mv] += 1
                parent_counts_all[p["parent_index"]] += 1
                if p["pdg"] == PDG_PROTON:
                    n_protons += 1
                    mate_counts_proton[mv] += 1
                elif p["pdg"] == PDG_PION:
                    n_pions += 1
                    mate_counts_pion[mv] += 1

            for pair in event.get("pairs", []):
                n_pairs += 1
                p_idx, pi_idx = pair["proton_idx"], pair["pion_idx"]
                if p_idx >= len(particles) or pi_idx >= len(particles):
                    continue

                proton, pion = particles[p_idx], particles[pi_idx]
                sig = bool(pair["is_signal"])
                if sig:
                    n_signal += 1
                    signal_mate_value_counts[proton["mate_index"]] += 1

                # parent_index encodes the PDG code of the parent particle's
                # species (not an array position) -- confirmed by its value
                # distribution being dominated by 1114/2114/2214/2224, the
                # I=3/2 Delta-resonance PDG codes. So a proton/pion pair
                # both showing parent_index == 2224 both came from the same
                # kind of parent (Delta++) -- this is our ground truth.
                ppar, pipar = proton["parent_index"], pion["parent_index"]
                if ppar == pipar:
                    parent_pdg_counts[ppar] += 1
                is_delta = (ppar == pipar == PDG_DELTA_PP)

                m = pair.get("inv_mass_GeV")
                if m is not None:
                    m = float(m)
                    _accum("all_pairs", m)
                    if sig:
                        _accum("mate_signal", m)
                    if is_delta:
                        _accum("parent_delta", m)

                if sig and is_delta:
                    tp += 1
                elif sig and not is_delta:
                    fp += 1
                elif not sig and is_delta:
                    fn += 1
                else:
                    tn += 1

            n_events += 1
            if args.max_events and n_events >= args.max_events:
                break

    print(f"File               : {args.data_dir}")
    print(f"Events processed   : {n_events:,}")
    print(f"Particles seen     : {n_particles:,}  (protons {n_protons:,}, pions {n_pions:,})")
    print(f"Pairs seen         : {n_pairs:,}")
    print(f"Signal pairs (mate-based label) : {n_signal:,}  ({100 * n_signal / max(n_pairs, 1):.3f}%)")

    print("\n--- mate_index sentinel check ---")
    print("Top 5 most common mate_index values (ALL particles):")
    for val, cnt in mate_counts_all.most_common(5):
        print(f"    mate_index={val!r:>10}  count={cnt:>12,}  ({100 * cnt / max(n_particles,1):.2f}% of all particles)")

    print("Top 5 most common mate_index values (protons only):")
    for val, cnt in mate_counts_proton.most_common(5):
        print(f"    mate_index={val!r:>10}  count={cnt:>12,}  ({100 * cnt / max(n_protons,1):.2f}% of protons)")

    print("Top 5 most common mate_index values (pions only):")
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
            print(">>> Suggests a sentinel-value collision bug (false positives). <<<")
        else:
            print(">>> Signal labels spread across many distinct mate_index values — "
                  "sentinel collision unlikely to be the (main) problem. <<<")

    print("\n--- parent_index ground truth check (shared parent, parent pdg == 2224 i.e. Delta++) ---")
    print("Top 5 most common parent_index values (ALL particles, for sentinel context):")
    for val, cnt in parent_counts_all.most_common(5):
        print(f"    parent_index={val!r:>10}  count={cnt:>12,}  ({100 * cnt / max(n_particles,1):.2f}% of all particles)")

    print("\nTop parent PDG codes among proton/pion pairs sharing a parent_index:")
    for val, cnt in parent_pdg_counts.most_common(8):
        flag = "  <-- Delta++" if val == PDG_DELTA_PP else ""
        print(f"    parent_pdg={val:>8}  count={cnt:>10,}{flag}")

    print(f"\nConfusion matrix: mate-based is_signal  vs.  shares-a-confirmed-Delta++-parent")
    print(f"  is_signal=1 AND real Delta++ parent  (true positive)                : {tp:,}")
    print(f"  is_signal=1 AND NOT a real Delta++ parent (false positive)          : {fp:,}")
    print(f"  is_signal=0 AND real Delta++ parent  (false negative -- MISSED)     : {fn:,}")
    print(f"  is_signal=0 AND NOT a real Delta++ parent (true negative)          : {tn:,}")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"\n>>> Recall: of pairs that truly share a Delta++ parent, mate-matching "
              f"labels {100*recall:.1f}% of them is_signal=1. <<<")
        if recall < 0.9:
            print(">>> Real signal is being missed by mate-based labeling -- consistent with "
                  "mate_index getting reassigned away from the true sibling (e.g. to a photon). <<<")
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f">>> Precision: of pairs currently labeled is_signal=1, {100*precision:.1f}% "
              f"actually share a confirmed Delta++ parent. <<<")
        if precision < 0.9:
            print(">>> A meaningful fraction of 'signal' labels don't correspond to a real "
                  "Delta++ parent -- spurious mate matches. <<<")

    print("\n--- invariant mass sanity check (real Delta++ should peak near 1.232 GeV/c^2) ---")
    for key, label in [("all_pairs", "All pairs (mostly background)"),
                        ("mate_signal", "mate-based is_signal=1"),
                        ("parent_delta", "parent_index==2224 ground truth")]:
        s0, s1, n = mass_stats[key]
        if n == 0:
            print(f"  {label:<35}: n=0")
            continue
        mean = s0 / n
        var  = max(s1 / n - mean * mean, 0.0)
        print(f"  {label:<35}: n={n:>10,}  mean={mean:.4f} GeV  std={var**0.5:.4f} GeV")


if __name__ == "__main__":
    main()
