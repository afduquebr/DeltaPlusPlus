#!/usr/bin/env python3
"""
Plot train vs validation loss for all runs of the latest SLURM array job.
Saves figs/loss_curves.png
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

LOGS_DIR = Path(__file__).parent.parent / "logs"
FIGS_DIR = Path(__file__).parent.parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# regex for epoch lines:  "    1      1.2639     0.6813    1.1920   0.7342  lr=1.0e-03"
EPOCH_RE = re.compile(
    r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+lr=([\de.+\-]+)"
)


def parse_log(path: Path) -> dict | None:
    epochs, tr_loss, va_loss, lrs = [], [], [], []
    for line in path.read_text().splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epochs.append(int(m.group(1)))
            tr_loss.append(float(m.group(2)))
            va_loss.append(float(m.group(4)))
            lrs.append(float(m.group(6)))
    if not epochs:
        return None
    return {
        "epochs":  np.array(epochs),
        "tr_loss": np.array(tr_loss),
        "va_loss": np.array(va_loss),
        "lrs":     np.array(lrs),
    }


def latest_job_logs(logs_dir: Path) -> list[tuple[int, Path]]:
    """Return (run_id, path) pairs from the most recent array job."""
    # group by job array id (filename: train_{array_id}_{run}.log)
    groups = defaultdict(list)
    for p in sorted(logs_dir.glob("train_*_*.log")):
        parts = p.stem.split("_")   # ["train", "47158575", "1"]
        if len(parts) == 3:
            groups[parts[1]].append((int(parts[2]), p))
    if not groups:
        return []
    latest = max(groups.keys())
    print(f"Using job array {latest}  ({len(groups[latest])} runs)")
    return sorted(groups[latest])


# ── Load ──────────────────────────────────────────────────────────────────────

runs = latest_job_logs(LOGS_DIR)
if not runs:
    print("No array-job log files found in", LOGS_DIR)
    sys.exit(1)

parsed = []
for run_id, path in runs:
    d = parse_log(path)
    if d:
        d["run"] = run_id
        parsed.append(d)
        print(f"  Run {run_id}: {len(d['epochs'])} epochs")

if not parsed:
    print("No epoch data found.")
    sys.exit(1)

# Align on common epoch grid (shortest run)
min_ep = min(len(d["epochs"]) for d in parsed)
epochs = parsed[0]["epochs"][:min_ep]

tr_mat = np.stack([d["tr_loss"][:min_ep] for d in parsed])  # (runs, epochs)
va_mat = np.stack([d["va_loss"][:min_ep] for d in parsed])

tr_mean, tr_std = tr_mat.mean(0), tr_mat.std(0)
va_mean, va_std = va_mat.mean(0), va_mat.std(0)

# LR change points (from first run)
lrs    = parsed[0]["lrs"][:min_ep]
lr_changes = [epochs[i] for i in range(1, len(lrs)) if lrs[i] != lrs[i-1]]


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

colors_tr = "#4878cf"
colors_va = "#d65f5f"

# Per-run thin lines
for d in parsed:
    ax.plot(d["epochs"][:min_ep], d["tr_loss"][:min_ep],
            color=colors_tr, lw=0.6, alpha=0.35)
    ax.plot(d["epochs"][:min_ep], d["va_loss"][:min_ep],
            color=colors_va, lw=0.6, alpha=0.35)

# Mean ± std bands
ax.plot(epochs, tr_mean, color=colors_tr, lw=2, label="Train (mean)")
ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                alpha=0.2, color=colors_tr)
ax.plot(epochs, va_mean, color=colors_va, lw=2, label="Val (mean)")
ax.fill_between(epochs, va_mean - va_std, va_mean + va_std,
                alpha=0.2, color=colors_va)

# LR drop markers
for ep in lr_changes:
    ax.axvline(ep, color="k", ls=":", lw=0.9, alpha=0.6)
if lr_changes:
    ax.axvline(lr_changes[0], color="k", ls=":", lw=0.9, alpha=0.6,
               label="LR decay")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Loss (BCE)", fontsize=13)
ax.set_title(r"$\bf{SOS}$ Simulation Internal", fontsize=13, loc="left")
ax.text(0.02, 0.08, r"$E_\mathrm{kin}/A = 1.23$ GeV, ParticleNet",
        transform=ax.transAxes, fontsize=11)
ax.legend(fontsize=10)
fig.tight_layout()

out = FIGS_DIR / "loss_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)
