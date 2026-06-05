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

PATH = "/pbs/home/a/aduque/private/Delta++"
MODEL_NAME = "ParticleNetPair"
DATA_PATH  = f"{PATH}/data/AuAu_1230MeV_1000evts_1.json.gz"
MODELS_DIR = f"{PATH}/models"
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
    model_path = f"{MODELS_DIR}/best_model_run{run_id}.pt"
    idx_path   = f"{MODELS_DIR}/test_idx_run{run_id}.npy"
    norm_path  = f"{MODELS_DIR}/normalizer_run{run_id}.npz"

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

figs_dir = f"{PATH}/figs"
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

# ─── Δ++ mass reconstruction ─────────────────────────────────────────────────
#
# For each run, the model score cut selects a subset of pairs. We then plot
# the invariant mass spectrum of the selected pairs and compare it to the
# unfiltered spectrum. The Δ++ peak at 1.232 GeV/c² should become more
# prominent after the cut as background is suppressed.
#
# Significance in a mass window: S / √B, where S and B are the number of
# true signal and background pairs with inv_mass ∈ [MASS_WIN_LO, MASS_WIN_HI].

DELTA_MASS  = 1.232   # PDG Δ++ mass, GeV/c²
MASS_WIN_LO = 1.15    # signal window
MASS_WIN_HI = 1.35
MASS_BINS   = np.linspace(1.0, 1.6, 100)
MASS_CTRS   = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])


# ── Per-run significance table ──────────────────────────────────────────────

print(f"\n{'─'*80}")
print(f"  Δ++ mass reconstruction  |  window [{MASS_WIN_LO}, {MASS_WIN_HI}] GeV/c²  |  cut = {SCORE_CUT}")
print(f"{'─'*80}")
print(f"  {'Run':<4}  {'S_bef':>6}  {'B_bef':>8}  {'S/√B bef':>9}  "
      f"{'S_aft':>6}  {'B_aft':>8}  {'S/√B aft':>9}  {'B-rej':>7}  {'Improvement':>12}")
print(f"  {'─'*95}")

snr_before_all, snr_after_all = [], []

for r in results:
    m        = r["inv_mass"]
    y_t      = r["y_test"]
    y_p      = r["y_pred"]
    win      = (m >= MASS_WIN_LO) & (m <= MASS_WIN_HI)
    cut      = y_p == 1

    nb_s = int(((y_t == 1) & win).sum())
    nb_b = int(((y_t == 0) & win).sum())
    na_s = int(((y_t == 1) & win & cut).sum())
    na_b = int(((y_t == 0) & win & cut).sum())

    snr_b = nb_s / np.sqrt(nb_b) if nb_b > 0 else 0.0
    snr_a = na_s / np.sqrt(na_b) if na_b > 0 else 0.0
    b_rej = (1 - na_b / nb_b) * 100 if nb_b > 0 else 0.0
    improv = snr_a / snr_b if snr_b > 0 else float("nan")

    snr_before_all.append(snr_b)
    snr_after_all.append(snr_a)

    print(f"  {r['run']:<4}  {nb_s:>6,}  {nb_b:>8,}  {snr_b:>9.3f}  "
          f"{na_s:>6,}  {na_b:>8,}  {snr_a:>9.3f}  {b_rej:>6.1f}%  {improv:>11.2f}×")

mean_snr_b = np.mean(snr_before_all)
mean_snr_a = np.mean(snr_after_all)
print(f"\n  Mean S/√B before cut : {mean_snr_b:.3f} ± {np.std(snr_before_all):.3f}")
print(f"  Mean S/√B after  cut : {mean_snr_a:.3f} ± {np.std(snr_after_all):.3f}")
print(f"  Mean improvement     : {mean_snr_a / mean_snr_b:.2f}×")


# ── Significance scan vs score threshold ───────────────────────────────────

cut_grid  = np.linspace(0.01, 0.99, 200)
snr_scans = []

for r in results:
    m     = r["inv_mass"]
    y_t   = r["y_test"]
    probs = r["y_prob"]
    win   = (m >= MASS_WIN_LO) & (m <= MASS_WIN_HI)
    scan  = []
    for c in cut_grid:
        sel  = probs >= c
        ns   = ((y_t == 1) & win & sel).sum()
        nb   = ((y_t == 0) & win & sel).sum()
        scan.append(ns / np.sqrt(nb) if nb > 0 else 0.0)
    snr_scans.append(scan)

snr_scan_mean = np.mean(snr_scans, axis=0)
snr_scan_std  = np.std(snr_scans,  axis=0)
optimal_cut   = cut_grid[np.argmax(snr_scan_mean)]
print(f"\n  Optimal score cut (max S/√B) : {optimal_cut:.3f}")
print(f"  S/√B at optimal cut          : {snr_scan_mean.max():.3f} ± {snr_scan_std[np.argmax(snr_scan_mean)]:.3f}")


# ── Mass spectrum histograms (averaged over runs) ───────────────────────────

# Accumulate per-run histograms then average to get mean ± std
def mass_hist(masses): return np.histogram(masses, bins=MASS_BINS)[0].astype(float)

hists_all, hists_cut, hists_sig = [], [], []
for r in results:
    m   = r["inv_mass"]
    y_t = r["y_test"]
    y_p = r["y_pred"]
    hists_all.append(mass_hist(m))
    hists_cut.append(mass_hist(m[y_p == 1]))
    hists_sig.append(mass_hist(m[y_t == 1]))

h_all_mean, h_all_std = np.mean(hists_all, 0), np.std(hists_all, 0)
h_cut_mean, h_cut_std = np.mean(hists_cut, 0), np.std(hists_cut, 0)
h_sig_mean, h_sig_std = np.mean(hists_sig, 0), np.std(hists_sig, 0)


# ── Plot 1: mass spectrum before vs after cut ───────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.step(MASS_CTRS, h_all_mean, where="mid", color="grey",      lw=1.5, label="All pairs (no cut)")
ax.fill_between(MASS_CTRS, h_all_mean - h_all_std, h_all_mean + h_all_std,
                step="mid", alpha=0.2, color="grey")
ax.step(MASS_CTRS, h_cut_mean, where="mid", color="steelblue", lw=1.5,
        label=f"Score ≥ {SCORE_CUT}")
ax.fill_between(MASS_CTRS, h_cut_mean - h_cut_std, h_cut_mean + h_cut_std,
                step="mid", alpha=0.25, color="steelblue")
ax.step(MASS_CTRS, h_sig_mean, where="mid", color="tomato",    lw=1.5, label="True signal (Δ⁺⁺)")
ax.fill_between(MASS_CTRS, h_sig_mean - h_sig_std, h_sig_mean + h_sig_std,
                step="mid", alpha=0.25, color="tomato")
ax.axvline(DELTA_MASS, color="k", ls="--", lw=1, label=f"M(Δ⁺⁺) = {DELTA_MASS} GeV/c²")
ax.axvspan(MASS_WIN_LO, MASS_WIN_HI, alpha=0.06, color="gold", label="Signal window")
ax.set_xlabel(r"$M(p\pi^+)$ [GeV/c²]", fontsize=13)
ax.set_ylabel("Pairs / bin", fontsize=13)
ax.set_ylim(-50, 2000)
ax.set_title(r"$\bf{SOS}$ Simulation Internal", fontsize=13, loc="left")
ax.legend(fontsize=9)
sos_sim_label(ax)

# ── Plot 2: significance scan ────────────────────────────────────────────────

ax = axes[1]
ax.plot(cut_grid, snr_scan_mean, color="steelblue", lw=2)
ax.fill_between(cut_grid, snr_scan_mean - snr_scan_std, snr_scan_mean + snr_scan_std,
                alpha=0.25, color="steelblue", label=r"Mean ± 1$\sigma$")
ax.axvline(SCORE_CUT,    color="tomato", ls="--", lw=1.5, label=f"Cut used = {SCORE_CUT}")
ax.axvline(optimal_cut,  color="k",      ls=":",  lw=1.5,
           label=f"Optimal = {optimal_cut:.2f}  (S/√B = {snr_scan_mean.max():.2f})")
ax.set_xlabel("Score cut", fontsize=13)
ax.set_ylabel(r"$S/\sqrt{B}$ in signal window", fontsize=13)
ax.set_title(r"$\bf{SOS}$ Simulation Internal", fontsize=13, loc="left")
ax.set_ylim(-0.1, 1.6)
ax.legend(fontsize=9)
sos_sim_label(ax)

fig.tight_layout()
save(fig, "mass_reconstruction.png")