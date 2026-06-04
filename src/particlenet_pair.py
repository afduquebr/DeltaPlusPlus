#!/usr/bin/env python3
"""
ParticleNet classifier for Delta++ signal vs combinatorial background.

Each proton-pion pair is treated as a 2-node point cloud:
  node 0 = proton,  node 1 = pion
EdgeConv computes pairwise features (Δη, Δφ, Δp, ...) — the angular and
momentum correlations that distinguish resonance decays from random pairs.
Pair-level features (inv_mass, pair_pt, ...) are concatenated at the head.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


# ─── Feature lists ────────────────────────────────────────────────────────────

# Observable kinematics per particle (no truth info)
PARTICLE_FEATS = [
    "pt_GeV", "eta", "phi_rad", "rapidity", "mass_GeV",
    "px_GeV", "py_GeV", "pz_GeV", "energy_GeV", "p_GeV",
]

# Observable pair-level features
PAIR_FEATS = [
    "inv_mass_GeV", "pair_pt_GeV", "pair_rapidity", "delta_rapidity",
]

N_PARTICLE_FEATS = len(PARTICLE_FEATS)   # 10
N_PAIR_FEATS     = len(PAIR_FEATS)       # 4


# ─── Data extraction ──────────────────────────────────────────────────────────

def _particle_row(p: dict) -> list:
    mom = p["momentum"]
    return [
        p.get("pt_GeV")    or 0.0,
        p.get("eta")       or 0.0,
        p.get("phi_rad")   or 0.0,
        p.get("rapidity")  or 0.0,
        p.get("mass_GeV")  or 0.0,
        mom["px_GeV"],
        mom["py_GeV"],
        mom["pz_GeV"],
        mom["energy_GeV"],
        p.get("p_GeV")     or 0.0,
    ]


PDG_PROTON = 2212
PDG_PION   = 211


def build_arrays(data: dict):
    """
    Iterate over all events and pairs; return numpy arrays ready for training.

    Returns
    -------
    node_feats : (N, 2, 10)  — row 0 = proton (PDG 2212), row 1 = pion (PDG 211)
    pair_feats : (N, 4)
    labels     : (N,)        — is_signal
    """
    nodes, pairs_out, labels = [], [], []
    skipped = 0

    for event in data["events"]:
        particles = event["particles"]
        for pair in event.get("pairs", []):
            p_idx  = pair["proton_idx"]
            pi_idx = pair["pion_idx"]
            if p_idx >= len(particles) or pi_idx >= len(particles):
                skipped += 1
                continue

            proton = particles[p_idx]
            pion   = particles[pi_idx]

            # Guard: confirm these are the expected particle types
            if proton.get("pdg") != PDG_PROTON or pion.get("pdg") != PDG_PION:
                skipped += 1
                continue

            nodes.append([
                _particle_row(proton),   # proton — node 0
                _particle_row(pion),     # pion   — node 1
            ])
            pairs_out.append([
                pair.get("inv_mass_GeV")    or 0.0,
                pair.get("pair_pt_GeV")     or 0.0,
                pair.get("pair_rapidity")   or 0.0,
                pair.get("delta_rapidity")  or 0.0,
            ])
            labels.append(pair["is_signal"])

    if skipped:
        print(f"  [build_arrays] skipped {skipped} pairs (index OOB or wrong PDG)")

    node_feats = np.array(nodes,     dtype=np.float32)  # (N, 2, 10)
    pair_feats = np.array(pairs_out, dtype=np.float32)  # (N, 4)
    labels     = np.array(labels,    dtype=np.float32)  # (N,)
    return node_feats, pair_feats, labels


# ─── Normalisation ────────────────────────────────────────────────────────────

class Normaliser:
    """Fit mean/std on train split, apply to all splits."""

    def fit(self, node_feats: np.ndarray, pair_feats: np.ndarray):
        # node_feats: (N, 2, F) — flatten particles to compute stats
        flat = node_feats.reshape(-1, N_PARTICLE_FEATS)
        self.node_mean = flat.mean(0)
        self.node_std  = flat.std(0) + 1e-8

        self.pair_mean = pair_feats.mean(0)
        self.pair_std  = pair_feats.std(0) + 1e-8
        return self

    def transform_nodes(self, x: np.ndarray) -> np.ndarray:
        return (x - self.node_mean) / self.node_std

    def transform_pairs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.pair_mean) / self.pair_std

    def save(self, path):
        np.savez(path, node_mean=self.node_mean, node_std=self.node_std,
                 pair_mean=self.pair_mean, pair_std=self.pair_std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls()
        obj.node_mean = d["node_mean"]
        obj.node_std  = d["node_std"]
        obj.pair_mean = d["pair_mean"]
        obj.pair_std  = d["pair_std"]
        return obj


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, node_feats, pair_feats, labels):
        self.node_feats = torch.from_numpy(node_feats)
        self.pair_feats = torch.from_numpy(pair_feats)
        self.labels     = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.node_feats[i], self.pair_feats[i], self.labels[i]


# ─── Loaders ──────────────────────────────────────────────────────────────────

def split_indices(labels: np.ndarray, random_state: int = 42):
    """Return (idx_tr, idx_va, idx_te) for the standard 80/10/10 split."""
    idx = np.arange(len(labels))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=0.20, random_state=random_state, stratify=labels
    )
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=0.50, random_state=random_state, stratify=labels[idx_tmp]
    )
    return idx_tr, idx_va, idx_te


def make_loaders(
    node_feats: np.ndarray,
    pair_feats: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    random_state: int = 42,
):
    """Split → normalise → wrap in DataLoaders.

    Returns (train_loader, val_loader, test_loader, normaliser).
    """
    idx_tr, idx_va, idx_te = split_indices(labels, random_state)

    norm = Normaliser().fit(node_feats[idx_tr], pair_feats[idx_tr])

    def _loader(idx, shuffle):
        ds = PairDataset(
            norm.transform_nodes(node_feats[idx]),
            norm.transform_pairs(pair_feats[idx]),
            labels[idx],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _loader(idx_tr, True), _loader(idx_va, False), _loader(idx_te, False), norm


# ─── Model ────────────────────────────────────────────────────────────────────

class EdgeConv2(nn.Module):
    """
    EdgeConv for a fixed 2-node point cloud.

    Each node i has exactly one neighbour j (the other particle).
    The edge feature is:

        e(i→j) = concat( h_i,  h_j − h_i )

    which encodes both the node's own kinematics AND the difference
    w.r.t. its partner (Δη, Δφ, Δpt, etc.).
    Both nodes share the same MLP weights.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, C)
        B = x.size(0)
        h0, h1 = x[:, 0], x[:, 1]                      # (B, C) each

        # Edge features for both nodes, stacked → single BN pass
        e0 = torch.cat([h0, h1 - h0], dim=-1)           # proton sees pion
        e1 = torch.cat([h1, h0 - h1], dim=-1)           # pion   sees proton
        edges = torch.cat([e0, e1], dim=0)               # (2B, 2C)

        out = self.mlp(edges)                            # (2B, out_ch)
        new_h0, new_h1 = out[:B], out[B:]

        return torch.stack([new_h0, new_h1], dim=1)     # (B, 2, out_ch)


class ParticleNetPair(nn.Module):
    """
    ParticleNet adapted for 2-particle pair classification.

    Graph:  2 nodes, 1 directed edge each way (fully connected for N=2).
    Skip-connections: intermediate pooled features are concatenated before
    the classifier, following the original ParticleNet design.

    Input
    -----
    node_feats : (B, 2, 10)   per-particle kinematics (normalised)
    pair_feats : (B, 4)       pair-level observables  (normalised)

    Output
    ------
    logits : (B,)   raw (pre-sigmoid) signal scores
    """

    def __init__(
        self,
        n_particle_feats: int = N_PARTICLE_FEATS,
        n_pair_feats: int = N_PAIR_FEATS,
        channels: tuple = (64, 128, 256),
    ):
        super().__init__()

        ch = channels
        self.conv1 = EdgeConv2(n_particle_feats, ch[0])
        self.conv2 = EdgeConv2(ch[0],            ch[1])
        self.conv3 = EdgeConv2(ch[1],            ch[2])

        # After global mean-pooling each layer: ch[0]+ch[1]+ch[2] = 448
        pooled = sum(ch)

        self.head = nn.Sequential(
            nn.Linear(pooled + n_pair_feats, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, node_feats: torch.Tensor, pair_feats: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(node_feats)                              # (B, 2, 64)
        h2 = self.conv2(h1)                                      # (B, 2, 128)
        h3 = self.conv3(h2)                                      # (B, 2, 256)

        # Mean-pool over the 2 nodes at each layer (skip connections)
        g = torch.cat([h1.mean(1), h2.mean(1), h3.mean(1)], -1) # (B, 448)

        x = torch.cat([g, pair_feats], dim=-1)                   # (B, 452)
        return self.head(x).squeeze(-1)                          # (B,)


# ─── Training utilities ───────────────────────────────────────────────────────


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    total_loss, n = 0.0, 0
    scores_all, labels_all = [], []

    with ctx:
        for nf, pf, y in loader:
            nf, pf, y = nf.to(device), pf.to(device), y.to(device)
            logits = model(nf, pf)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            n          += len(y)
            scores_all.append(torch.sigmoid(logits).detach().cpu())
            labels_all.append(y.cpu())

    scores = torch.cat(scores_all).numpy()
    labels = torch.cat(labels_all).numpy()
    auc    = roc_auc_score(labels, scores) if labels.std() > 0 else float("nan")
    return total_loss / n, auc


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?",
                        default=str(Path(__file__).parent / "AuAu_1230MeV_1000evts_1.json.gz"),
                        help="Path to input .json.gz file")
    parser.add_argument("--run", type=int, default=1,
                        help="Run index (1-5); sets random seed and output filename")
    args = parser.parse_args()

    data_path = Path(args.data)
    run_id    = args.run

    torch.manual_seed(run_id)
    np.random.seed(run_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {run_id}  |  Device : {device}")

    print(f"Loading {data_path.name} ...")
    with gzip.open(data_path, "rt") as f:
        data = json.load(f)

    print("Building pair arrays ...")
    node_feats, pair_feats, labels = build_arrays(data)
    n_sig = int(labels.sum())
    n_bg  = int((1 - labels).sum())
    print(f"  Pairs : {len(labels):,}  |  signal {n_sig:,}  |  background {n_bg:,}")
    print(f"  Imbalance ratio : {n_bg / n_sig:.1f}× more background than signal")

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, norm = make_loaders(
        node_feats, pair_feats, labels, random_state=run_id
    )
    _, _, idx_te = split_indices(labels, random_state=run_id)

    # Persist split and normalizer so evaluation never needs to re-split
    np.save(models_dir / f"test_idx_run{run_id}.npy", idx_te)
    norm.save(models_dir / f"normalizer_run{run_id}.npz")
    print(f"  Test set : {len(idx_te):,} pairs (indices saved)")

    model     = ParticleNetPair().to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters : {n_params:,}\n")

    pos_weight = torch.tensor([n_bg / n_sig], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    save_path    = models_dir / f"best_model_run{run_id}.pt"
    best_val_auc = 0.0

    print(f"{'Epoch':>5}  {'Train loss':>10}  {'Train AUC':>9}  {'Val loss':>8}  {'Val AUC':>7}")
    print("-" * 50)

    for epoch in range(1, 31):
        tr_loss, tr_auc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_auc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step(va_loss)

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            torch.save(model.state_dict(), save_path)

        if epoch % 1 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_auc:>9.4f}  {va_loss:>8.4f}  {va_auc:>7.4f}  lr={lr:.1e}")

    # ── Final test evaluation
    print(f"\nBest val AUC : {best_val_auc:.4f}")
    print(f"Saved → {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    te_loss, te_auc = run_epoch(model, test_loader, None, criterion, device, train=False)

    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for nf, pf, y in test_loader:
            logits = model(nf.to(device), pf.to(device))
            all_scores.append(torch.sigmoid(logits).cpu())
            all_labels.append(y)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    ap = average_precision_score(labels, scores)
    print(f"Test AUC-ROC  : {te_auc:.4f}")
    print(f"Test AP       : {ap:.4f}  (random baseline ≈ {labels.mean():.4f})")
