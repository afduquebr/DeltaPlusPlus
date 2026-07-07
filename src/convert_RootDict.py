#!/usr/bin/env python3
"""Local, ROOT-free replacement for convert_RootDict.py.

Reads the same UniGen-format ROOT file directly via uproot's generic
streamer-info decoding, so no PyROOT / libUniGen.so is required. Produces
the same nested JSON structure as the original script.

Run with the `jetflow` conda env (has uproot + numpy):
    /opt/homebrew/Caskroom/miniforge/base/envs/jetflow/bin/python3 convert_RootDict_uproot.py
"""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import uproot

INPUT_FILE = os.path.expanduser(
    "/sps/atlas.new/a/aduque/Delta++/urqmd_f15_flagEos0_1e6.root"
)
OUTPUT_DIR = os.path.expanduser("/sps/atlas.new/a/aduque/Delta++")

CHUNK_SIZE = 2000  # events per chunk; lower this if you still hit OOM

PROTON_PDG = 2212
PION_PDG = 211

PARTICLE_BRANCHES = [
    "fParticles.fIndex",
    "fParticles.fPdg",
    "fParticles.fStatus",
    "fParticles.fParent",
    "fParticles.fParentDecay",
    "fParticles.fMate",
    "fParticles.fDecay",
    "fParticles.fChild[2]",
    "fParticles.fPx",
    "fParticles.fPy",
    "fParticles.fPz",
    "fParticles.fE",
    "fParticles.fX",
    "fParticles.fY",
    "fParticles.fZ",
    "fParticles.fT",
]

EVENT_BRANCHES = ["fB", "fPhi", "fNes", "fStepNr", "fStepT", "fNpa"]


# ================ Kinematics (TLorentzVector equivalents) ================ #

def _lorentz_kinematics(px, py, pz, e):
    """Vectorized numpy equivalents of ROOT's TLorentzVector getters."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        pt = np.sqrt(px ** 2 + py ** 2)
        cos_theta = np.clip(np.divide(pz, p, out=np.zeros_like(pz), where=p != 0), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        phi = np.arctan2(py, px)
        eta = 0.5 * np.log((p + pz) / (p - pz))
        rapidity = 0.5 * np.log((e + pz) / (e - pz))
        m2 = e ** 2 - p ** 2
        mass = np.where(m2 < 0, -np.sqrt(-m2), np.sqrt(m2))
    return p, pt, eta, phi, theta, rapidity, mass


def _to_float_list(arr):
    """Elementwise float(), with non-finite values mapped to None (matches
    the original script's _to_float() isfinite guard)."""
    finite = np.isfinite(arr)
    return [float(v) if f else None for v, f in zip(arr, finite)]


def _to_int_list(arr):
    return [int(v) for v in arr]


# ================ Dictionary Builders =================================== #

def event_to_dict(ev, event_number):
    """Build one event's nested dict from the raw per-event field arrays."""
    n_particles = int(ev["fNpa"])

    pdg = np.asarray(ev["fParticles.fPdg"], dtype=np.int64)
    index = np.asarray(ev["fParticles.fIndex"], dtype=np.int64)
    status = np.asarray(ev["fParticles.fStatus"], dtype=np.int64)
    parent = np.asarray(ev["fParticles.fParent"], dtype=np.int64)
    parent_decay = np.asarray(ev["fParticles.fParentDecay"], dtype=np.int64)
    mate = np.asarray(ev["fParticles.fMate"], dtype=np.int64)
    decay = np.asarray(ev["fParticles.fDecay"], dtype=np.int64)
    child = np.asarray(ev["fParticles.fChild[2]"], dtype=np.int64).reshape(-1, 2)
    first_child = child[:, 0] if n_particles else np.array([], dtype=np.int64)
    last_child = child[:, 1] if n_particles else np.array([], dtype=np.int64)

    px = np.asarray(ev["fParticles.fPx"], dtype=np.float64)
    py = np.asarray(ev["fParticles.fPy"], dtype=np.float64)
    pz = np.asarray(ev["fParticles.fPz"], dtype=np.float64)
    e = np.asarray(ev["fParticles.fE"], dtype=np.float64)
    x = np.asarray(ev["fParticles.fX"], dtype=np.float64)
    y = np.asarray(ev["fParticles.fY"], dtype=np.float64)
    z = np.asarray(ev["fParticles.fZ"], dtype=np.float64)
    t = np.asarray(ev["fParticles.fT"], dtype=np.float64)

    p, pt, eta, phi, theta, rapidity, mass = _lorentz_kinematics(px, py, pz, e)

    px_f, py_f, pz_f, e_f = _to_float_list(px), _to_float_list(py), _to_float_list(pz), _to_float_list(e)
    x_f, y_f, z_f, t_f = _to_float_list(x), _to_float_list(y), _to_float_list(z), _to_float_list(t)
    p_f, pt_f = _to_float_list(p), _to_float_list(pt)
    eta_f, phi_f, theta_f = _to_float_list(eta), _to_float_list(phi), _to_float_list(theta)
    rapidity_f, mass_f = _to_float_list(rapidity), _to_float_list(mass)

    particles = [
        {
            "index": int(index[i]),
            "pdg": int(pdg[i]),
            "status": int(status[i]),
            "parent_index": int(parent[i]),
            "parent_decay_index": int(parent_decay[i]),
            "mate_index": int(mate[i]),
            "decay_index": int(decay[i]),
            "first_child_index": int(first_child[i]),
            "last_child_index": int(last_child[i]),
            "momentum": {
                "px_GeV": px_f[i],
                "py_GeV": py_f[i],
                "pz_GeV": pz_f[i],
                "energy_GeV": e_f[i],
            },
            "position": {
                "x_fm": x_f[i],
                "y_fm": y_f[i],
                "z_fm": z_f[i],
                "t_fm": t_f[i],
            },
            "p_GeV": p_f[i],
            "pt_GeV": pt_f[i],
            "eta": eta_f[i],
            "phi_rad": phi_f[i],
            "theta_rad": theta_f[i],
            "rapidity": rapidity_f[i],
            "mass_GeV": mass_f[i],
        }
        for i in range(n_particles)
    ]

    proton_idx = np.flatnonzero(pdg == PROTON_PDG)
    pion_idx = np.flatnonzero(pdg == PION_PDG)

    pairs = []
    if proton_idx.size and pion_idx.size:
        pp_px = px[proton_idx][:, None] + px[pion_idx][None, :]
        pp_py = py[proton_idx][:, None] + py[pion_idx][None, :]
        pp_pz = pz[proton_idx][:, None] + pz[pion_idx][None, :]
        pp_e = e[proton_idx][:, None] + e[pion_idx][None, :]

        _, pair_pt, _, _, _, pair_rapidity, pair_mass = _lorentz_kinematics(pp_px, pp_py, pp_pz, pp_e)
        delta_rapidity = rapidity[proton_idx][:, None] - rapidity[pion_idx][None, :]
        is_signal = (mate[proton_idx][:, None] == mate[pion_idx][None, :]).astype(int)

        proton_grid, pion_grid = np.meshgrid(proton_idx, pion_idx, indexing="ij")

        inv_mass_f = _to_float_list(pair_mass.ravel())
        pair_pt_f = _to_float_list(pair_pt.ravel())
        pair_rapidity_f = _to_float_list(pair_rapidity.ravel())
        delta_rapidity_f = _to_float_list(delta_rapidity.ravel())

        pairs = [
            {
                "proton_idx": int(pi),
                "pion_idx": int(ki),
                "is_signal": int(sig),
                "inv_mass_GeV": inv_mass_f[j],
                "pair_pt_GeV": pair_pt_f[j],
                "pair_rapidity": pair_rapidity_f[j],
                "delta_rapidity": delta_rapidity_f[j],
                "label_source": "mate_match",
            }
            for j, (pi, ki, sig) in enumerate(
                zip(proton_grid.ravel(), pion_grid.ravel(), is_signal.ravel())
            )
        ]

    return {
        "event_number": int(event_number),
        "impact_parameter_fm": float(ev["fB"]) if np.isfinite(ev["fB"]) else None,
        "reaction_plane_angle_rad": float(ev["fPhi"]) if np.isfinite(ev["fPhi"]) else None,
        "num_event_steps": int(ev["fNes"]),
        "event_step_number": int(ev["fStepNr"]),
        "event_step_time": float(ev["fStepT"]) if np.isfinite(ev["fStepT"]) else None,
        "num_particles": n_particles,
        "generator_info": {
            # UEvent has no generator/target/projectile getters in this
            # UniGen version, so these were always None in the ROOT-based
            # script too.
            "generator": None,
            "target": None,
            "projectile": None,
        },
        "particles": particles,
        "pairs": pairs,
    }


def root_file_to_nested_dict(input_file_path, tree_name="events", max_events=None):
    """Read a ROOT file and return a fully Python-native nested dictionary.

    Loads all requested events into memory at once. Fine for a small
    max_events (interactive checks), but for a full multi-hundred-thousand
    event file use convert_root_to_json_gz() instead, which streams in
    chunks and never holds the whole dataset in memory.
    """
    with uproot.open(input_file_path) as f:
        tree = f[tree_name]
        n_entries = tree.num_entries
        n_to_read = n_entries if max_events is None else min(int(max_events), n_entries)

        arrays = tree.arrays(
            EVENT_BRANCHES + PARTICLE_BRANCHES,
            entry_stop=n_to_read,
            library="np",
        )

    data = {
        "meta": {
            "input_file": input_file_path,
            "tree_name": tree_name,
            "total_entries": n_entries,
            "events_loaded": n_to_read,
        },
        "events": [],
    }

    for i_ev in range(n_to_read):
        ev = {branch: arrays[branch][i_ev] for branch in EVENT_BRANCHES + PARTICLE_BRANCHES}
        data["events"].append(event_to_dict(ev, event_number=i_ev))

    return data


def convert_root_to_json_gz(
    input_file_path,
    output_path,
    tree_name="events",
    max_events=None,
    chunk_size=CHUNK_SIZE,
):
    """Stream a ROOT file straight to compressed JSON, chunk by chunk.

    Reads and converts `chunk_size` events at a time via uproot's
    tree.iterate(), writing each event's dict to the gzip stream as it goes.
    Peak memory stays bounded by chunk_size regardless of the file's total
    event count, unlike root_file_to_nested_dict() + save_event_dictionary_compact()
    which both materialize the whole dataset in memory first.

    Returns (total_entries, events_written, first_event_dict, output_path_str).
    """
    path = Path(output_path)
    if not str(path).endswith(".json.gz"):
        path = path.with_suffix(".json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)

    with uproot.open(input_file_path) as f:
        tree = f[tree_name]
        n_entries = tree.num_entries
    n_to_read = n_entries if max_events is None else min(int(max_events), n_entries)

    meta = {
        "input_file": input_file_path,
        "tree_name": tree_name,
        "total_entries": n_entries,
        "events_loaded": n_to_read,
    }

    first_event = None
    events_written = 0

    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as out:
        out.write('{"meta":')
        # json.dumps (not json.dump) so the fast C encoder is used instead
        # of json.dump's pure-Python fallback, which is ~6x slower.
        out.write(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
        out.write(',"events":[')

        with uproot.open(input_file_path) as f:
            tree = f[tree_name]
            for chunk in tree.iterate(
                EVENT_BRANCHES + PARTICLE_BRANCHES,
                step_size=chunk_size,
                entry_stop=n_to_read,
                library="np",
            ):
                chunk_len = len(chunk[EVENT_BRANCHES[0]])
                for i in range(chunk_len):
                    ev = {branch: chunk[branch][i] for branch in EVENT_BRANCHES + PARTICLE_BRANCHES}
                    ev_dict = event_to_dict(ev, event_number=events_written)

                    if events_written > 0:
                        out.write(",")
                    out.write(json.dumps(ev_dict, ensure_ascii=False, separators=(",", ":")))

                    if first_event is None:
                        first_event = ev_dict
                    events_written += 1

        out.write("]}")

    return n_entries, events_written, first_event, str(path.resolve())


# ================ IO Functions ========================================= #

def save_event_dictionary_compact(data, output_path):
    """Save as compressed gzip JSON."""
    path = Path(output_path)
    if not str(path).endswith(".json.gz"):
        path = path.with_suffix(".json.gz")

    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as f:
        f.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    return str(path.resolve())


def load_event_dictionary_compressed_json(input_path):
    """Load compressed gzip JSON."""
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ================ Main Execution Block ================================== #

if __name__ == "__main__":
    print(f"Processing ROOT file: {INPUT_FILE}")
    output_file = Path(OUTPUT_DIR) / (Path(INPUT_FILE).stem + ".json.gz")

    n_entries, n_written, first_event, json_gz_out = convert_root_to_json_gz(
        INPUT_FILE, output_file, max_events=None, chunk_size=CHUNK_SIZE
    )

    print("\n--- Summary ---")
    print("Total entries in file:", n_entries)
    print("Events written:", n_written)

    if first_event is not None:
        if first_event["particles"]:
            first_particle = first_event["particles"][0]
            print("\n--- Validation ---")
            print("First particle keys:", list(first_particle.keys()))
            print("Momentum keys:", list(first_particle["momentum"].keys()))
            print("Position keys:", list(first_particle["position"].keys()))

        n_pairs = len(first_event["pairs"])
        n_signal = sum(pair["is_signal"] for pair in first_event["pairs"])
        print(f"First event pairs: {n_pairs} (Signal: {n_signal})")

    print(f"\nSaved compressed JSON to: {json_gz_out}")
