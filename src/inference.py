#!/usr/bin/env python3
"""
Multi-run evaluation of ParticleNetPair.

Loads best_model_run{1..5}.pt, evaluates each on its own held-out test split
(reproduced with the same seed used during training), and reports:
  - Per-run metrics table
  - Mean ± std across runs
  - Overlaid ROC curves
  - Score distributions for each run
  - Invariant mass spectrum (averaged model selection)
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from particlenet_pair import (
    build_arrays,
    Normaliser, PairDataset,
    ParticleNetPair,
    N_PARTICLE_FEATS, N_PAIR_FEATS,
)
from torch.utils.data import DataLoader

MODEL_NAME = "ParticleNetPair"
DATA_PATH  = Path(__file__).parent / "data/AuAu_1230MeV_1000evts_1.json.gz"
MODELS_DIR = Path(__file__).parent / "models"
N_RUNS     = 5
SCORE_CUT  = 0.5


# ─── Load data ────────────────────────────────────────────────────────────────

print(f"Loading {DATA_PATH.name} ...")
with gzip.open(DATA_PATH, "rt") as f:
    data = json.load(f)

print("Building pair arrays ...")
node_feats, pair_feats, labels = build_arrays(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}\n")


# ─── Evaluate each run ────────────────────────────────────────────────────────

results = []   # list of dicts, one per run

for run_id in range(1, N_RUNS + 1):
    model_path = MODELS_DIR / f"best_model_run{run_id}.pt"
    idx_path   = MODELS_DIR / f"test_idx_run{run_id}.npy"
    norm_path  = MODELS_DIR / f"normalizer_run{run_id}.npz"

    missing = [p for p in (model_path, idx_path, norm_path) if not p.exists()]
    if missing:
        print(f"  [run {run_id}] missing artifacts: {[p.name for p in missing]}, skipping.")
        continue

    # Load persisted split — no re-splitting, guaranteed no leakage
    idx_te      = np.load(idx_path)
    norm        = Normaliser.load(norm_path)
    inv_mass_te = pair_feats[idx_te, 0]   # raw inv_mass_GeV before normalisation

    test_ds     = PairDataset(
        norm.transform_nodes(node_feats[idx_te]),
        norm.transform_pairs(pair_feats[idx_te]),
        labels[idx_te],
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = ParticleNetPair(N_PARTICLE_FEATS, N_PAIR_FEATS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for nf, pf, y in test_loader:
            logits = model(nf.to(device), pf.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.numpy())

    y_prob = np.concatenate(all_probs)
    y_test = np.concatenate(all_labels).astype(int)
    y_pred = (y_prob >= SCORE_CUT).astype(int)

    results.append({
        "run":       run_id,
        "auc":       roc_auc_score(y_test, y_prob),
        "ap":        average_precision_score(y_test, y_prob),
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "fpr":       roc_curve(y_test, y_prob)[0],
        "tpr":       roc_curve(y_test, y_prob)[1],
        "y_prob":    y_prob,
        "y_test":    y_test,
        "y_pred":    y_pred,
        "inv_mass":  inv_mass_te,
    })
    print(f"  Run {run_id}  AUC {results[-1]['auc']:.4f}  "
          f"AP {results[-1]['ap']:.4f}  "
          f"F1 {results[-1]['f1']:.4f}")

if not results:
    print("No models found. Train first with sbatch exec/train.sh")
    sys.exit(1)


# ─── Summary statistics ───────────────────────────────────────────────────────

metrics = ["auc", "ap", "accuracy", "precision", "recall", "f1"]
print(f"\n{'Metric':<12}  " + "  ".join(f"Run{r['run']}" for r in results))
print("-" * (12 + 8 * len(results)))
for m in metrics:
    vals = [r[m] for r in results]
    row  = f"{m:<12}  " + "  ".join(f"{v:.4f}" for v in vals)
    print(row)

print(f"\n{'Metric':<12}  {'Mean':>8}  {'Std':>8}")
print("-" * 32)
for m in metrics:
    vals = [r[m] for r in results]
    print(f"{m:<12}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def sos_sim_label(ax):
    """Add SOS Simulation Internal text on a given axes."""
    ax.text(0.03, 0.92, r"$\bf{SOS}$ Simulation Internal", transform=ax.transAxes, fontsize=15)
    ax.text(0.03, 0.86, r"$E_\mathrm{kin}/A = 1.23$ GeV, ParticleNet",  transform=ax.transAxes, fontsize=13)


auc_vals = [r["auc"] for r in results]
run_ids  = [r["run"] for r in results]

# Interpolate all ROC curves onto a common FPR grid for averaging
fpr_grid   = np.linspace(0, 1, 500)
tpr_matrix = np.array([np.interp(fpr_grid, r["fpr"], r["tpr"]) for r in results])
tpr_mean   = tpr_matrix.mean(0)
tpr_std    = tpr_matrix.std(0)

# Average score distributions on a common bin grid
bins        = np.linspace(0, 1, 60)
centers     = 0.5 * (bins[:-1] + bins[1:])
sig_hists   = np.array([np.histogram(r["y_prob"][r["y_test"] == 1], bins=bins, density=True)[0]
                         for r in results])
bg_hists    = np.array([np.histogram(r["y_prob"][r["y_test"] == 0], bins=bins, density=True)[0]
                         for r in results])
sig_mean, sig_std = sig_hists.mean(0), sig_hists.std(0)
bg_mean,  bg_std  = bg_hists.mean(0),  bg_hists.std(0)

# Average normalised confusion matrix
cms = np.array([confusion_matrix(r["y_test"], r["y_pred"], normalize="true")
                for r in results])
cm_mean = cms.mean(0)
cm_std  = cms.std(0)

figs_dir = Path(__file__).parent / "figs"
figs_dir.mkdir(exist_ok=True)

mean_auc = np.mean(auc_vals)
std_auc  = np.std(auc_vals)

def save(fig, name):
    p = figs_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
    plt.close(fig)


# 1 — Average ROC curve with ±1σ band
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_grid, tpr_mean, color="steelblue", lw=2,
        label=f"Mean ROC  AUC = {mean_auc:.3f} ± {std_auc:.3f}")
ax.fill_between(fpr_grid, tpr_mean - tpr_std, tpr_mean + tpr_std,
                alpha=0.25, color="steelblue", label=r"±1$\sigma$")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve (mean ± 1σ)")
ax.legend(fontsize=9)
sos_sim_label(ax)
fig.tight_layout()
save(fig, "roc_curve.png")


# 2 — AUC per run with mean ± std band
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(run_ids, auc_vals, color="steelblue", alpha=0.75)
ax.axhline(mean_auc, color="k", lw=1.5, ls="--", label=f"Mean = {mean_auc:.4f}")
ax.axhspan(mean_auc - std_auc, mean_auc + std_auc,
           alpha=0.15, color="k", label=f"±1σ = {std_auc:.4f}")
ax.set_xlabel("Run")
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC per run")
ax.set_xticks(run_ids)
ax.legend(fontsize=9)
ax.set_ylim(max(0, min(auc_vals) - 0.05), min(1, max(auc_vals) + 0.05))
sos_sim_label(ax)
fig.tight_layout()
save(fig, "auc_per_run.png")


# 3 — Average score distribution with ±1σ band
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(centers, sig_mean, color="tomato",    lw=2, label="Signal (Δ⁺⁺)")
ax.fill_between(centers, sig_mean - sig_std, sig_mean + sig_std, alpha=0.25, color="tomato")
ax.plot(centers, bg_mean,  color="steelblue", lw=2, label="Background")
ax.fill_between(centers, bg_mean - bg_std,  bg_mean + bg_std,  alpha=0.25, color="steelblue")
ax.axvline(SCORE_CUT, color="k", ls=":", lw=1, label=f"Cut = {SCORE_CUT}")
ax.set_xlabel("Model score")
ax.set_ylabel("Density")
ax.set_title("Score distribution (mean ± 1σ)")
ax.legend(fontsize=9)
sos_sim_label(ax)
fig.tight_layout()
save(fig, "score_distribution.png")


# 4 — Average confusion matrix (mean values, std in parentheses)
labels_disp = ["background", "Δ⁺⁺"]
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_mean, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
tick_marks = np.arange(len(labels_disp))
ax.set_xticks(tick_marks); ax.set_xticklabels(labels_disp, fontsize=14)
ax.set_yticks(tick_marks); ax.set_yticklabels(labels_disp, rotation=90, fontsize=14, va="center")
for i in range(len(labels_disp)):
    for j in range(len(labels_disp)):
        color = "white" if cm_mean[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_mean[i,j]:.3f}\n(±{cm_std[i,j]:.3f})",
                ha="center", va="center", fontsize=14, color=color)
ax.set_xlabel("Predicted label", fontsize=14)
ax.set_ylabel("True label", fontsize=14)
# ax.set_title("Confusion matrix (mean ± std, normalised)")
ax.set_title(r"$\bf{SOS}$ Simluation Internal", fontsize=14, loc='left', pad=10)
fig.tight_layout()
save(fig, "confusion_matrix.png")
