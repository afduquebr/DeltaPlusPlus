#!/usr/bin/env python3
"""
Plot train vs validation AUC-ROC for all runs of the latest SLURM array job.
Saves figs/auc_curves.png
"""

import re
import sys
from pathlib import Path
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

PATH = "/pbs/home/a/aduque/private/Delta++"
LOGS_DIR = f"{PATH}/logs"
FIGS_DIR = f"{PATH}/figs"
os.makedirs(FIGS_DIR, exist_ok=True)

EPOCH_RE = re.compile(
    r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+lr=([\de.+\-]+)"
)


def parse_log(path: Path) -> dict | None:
    epochs, tr_auc, va_auc, lrs = [], [], [], []
    for line in path.read_text().splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epochs.append(int(m.group(1)))
            tr_auc.append(float(m.group(3)))
            va_auc.append(float(m.group(5)))
            lrs.append(float(m.group(6)))
    if not epochs:
        return None
    return {
        "epochs": np.array(epochs),
        "tr_auc": np.array(tr_auc),
        "va_auc": np.array(va_auc),
        "lrs":    np.array(lrs),
    }


def latest_job_logs(logs_dir: Path) -> list[tuple[int, Path]]:
    groups = defaultdict(list)
    for p in sorted(Path(logs_dir).glob("train_*_*.log")):
        parts = p.stem.split("_")
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
        best_ep = d["va_auc"].argmax()
        print(f"  Run {run_id}: {len(d['epochs'])} epochs  "
              f"best val AUC = {d['va_auc'].max():.4f} @ epoch {d['epochs'][best_ep]}")

if not parsed:
    print("No epoch data found.")
    sys.exit(1)

min_ep = min(len(d["epochs"]) for d in parsed)
epochs = parsed[0]["epochs"][:min_ep]

tr_mat = np.stack([d["tr_auc"][:min_ep] for d in parsed])
va_mat = np.stack([d["va_auc"][:min_ep] for d in parsed])

tr_mean, tr_std = tr_mat.mean(0), tr_mat.std(0)
va_mean, va_std = va_mat.mean(0), va_mat.std(0)

lrs        = parsed[0]["lrs"][:min_ep]
lr_changes = [epochs[i] for i in range(1, len(lrs)) if lrs[i] != lrs[i-1]]

best_mean_ep  = epochs[va_mean.argmax()]
best_mean_auc = va_mean.max()


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

colors_tr = "#4878cf"
colors_va = "#d65f5f"

for d in parsed:
    ax.plot(d["epochs"][:min_ep], d["tr_auc"][:min_ep],
            color=colors_tr, lw=0.6, alpha=0.35)
    ax.plot(d["epochs"][:min_ep], d["va_auc"][:min_ep],
            color=colors_va, lw=0.6, alpha=0.35)

ax.plot(epochs, tr_mean, color=colors_tr, lw=2, label="Train (mean)")
ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                alpha=0.2, color=colors_tr)
ax.plot(epochs, va_mean, color=colors_va, lw=2, label="Val (mean)")
ax.fill_between(epochs, va_mean - va_std, va_mean + va_std,
                alpha=0.2, color=colors_va)

ax.axvline(best_mean_ep, color="k", ls="--", lw=1,
           label=f"Best val AUC = {best_mean_auc:.4f} (ep {best_mean_ep})")

for ep in lr_changes:
    ax.axvline(ep, color="grey", ls=":", lw=0.9, alpha=0.7)
if lr_changes:
    ax.axvline(lr_changes[0], color="grey", ls=":", lw=0.9, alpha=0.7,
               label="LR decay")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("AUC-ROC", fontsize=13)
ax.text(0.02, 0.12, r"$\bf{SOS}$ Simulation Internal",
        transform=ax.transAxes, fontsize=13)
ax.text(0.02, 0.08, r"$E_\mathrm{kin}/A = 1.23$ GeV, ParticleNet",
        transform=ax.transAxes, fontsize=11)
ax.legend(fontsize=10)
fig.tight_layout()

out = f"{FIGS_DIR}/auc_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)
