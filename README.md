# Δ⁺⁺ Resonance Signal vs Combinatorial Background Classification

Machine learning pipeline for identifying **Δ⁺⁺ → p + π⁺** decays in heavy-ion collision simulations, using a ParticleNet-based classifier.

---

## Repository structure

```
Delta++/
├── data/                        # Input data (*.json.gz)
├── models/                      # Saved model weights + split indices + normalizers
├── figs/                        # Output plots
├── exec/
│   └── train.sh                 # Job submission script
├── src/ 
    ├── particlenet_pair.py          # Data pipeline + model definition + training loop
    └── inference.py                 # Evaluation, plots, and metrics
└── requirements.txt             # Packages and versions used

---

## Usage

**Train (5 independent runs on a V100 GPU):**
```bash
./exec/train.sh
```

**Evaluate:**
```bash
python inference.py
```

---

## References

H. Qu and L. Gouskos, *"ParticleNet: Jet Tagging via Particle Clouds"*,
Phys. Rev. D **101**, 056019 (2020). [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)

---

## Credits

The ParticleNet architecture was adapted from the original work by Qu & Gouskos (cited above) for the specific problem of proton–pion pair classification in heavy-ion collisions. **The architecture design, adaptation strategy, and all physics decisions are entirely the work of A. Duque.**

Development of the data pipeline, training infrastructure, and evaluation scripts was assisted by [Claude](https://claude.ai) (Anthropic).
