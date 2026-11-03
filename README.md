# Reproducibility Package

Reproducibility package for the manuscript  
**"Noise, not optimality: a capacity-to-budget diagnostic for stealthy poisoning in differentially private federated learning"**  
(submitted to *Neurocomputing*).

This package contains the full attack–defense pipeline code, the per-run
result tables (CSV) and reports (TXT), and an automated data-consistency
checker that ties every numeric claim in the manuscript back to raw outputs.

**DOI (Zenodo archive):** [10.5281/zenodo.23906936](https://doi.org/10.5281/zenodo.23906936)

---

## 1. What is included

| Path | Contents |
|------|----------|
| `code/` | All Python scripts that generate the manuscript's numerical results and figures. |
| `results/` | The per-run result tables (`.csv`) and reports (`.txt`) that back every numeric claim. Reviewers can verify **all** claims via `verify_paper_claims.py` without re-running the full (compute-intensive) pipeline. |
| `baselines/NDSS21-Model-Poisoning/` | The official NDSS 2021 implementation of Shejwalkar & Chakraborty, used only by `verify_shejwalkar_fidelity.py` to certify fidelity of the attack-direction construction. Third-party code — see its own README/LICENSE. |
| `paper/` | Manuscript LaTeX sources (`dp_noise_stealthy_capacity.tex` and the supplementary material), so that `verify_paper_claims.py` can cross-check every numeric claim against the manuscript text itself. |

Directories `paper/figures/`, `results/fig/`, and `data/` are placeholders;
generated figures and downloaded datasets are created at runtime (and are
git-ignored).

---

## 2. Environment

- Python 3.13 / 3.14
- PyTorch 2.14 (CPU only — no GPU required)
- torchvision (matching version), NumPy, matplotlib
- scikit-learn (optional, used only by the statistics scripts)

Install dependencies with, e.g.:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## 3. Datasets

All datasets are **public benchmarks** loaded through `torchvision`:

- FashionMNIST, MNIST, CIFAR-10.

They are downloaded automatically on first run into `<repo>/data`. For
CIFAR-10 you may instead point to a pre-downloaded directory by setting the
environment variable before running:

```bash
export CIFAR_ROOT=/path/to/your/cifar10   # optional
```

No personal, human-subject, or sensitive data are used; all perturbations
are synthetic.

---

## 4. Quick verification (no heavy recomputation)

The fastest way to confirm the manuscript's numbers is to run the
consistency checker against the **included** result tables:

```bash
python code/verify_paper_claims.py
```

It checks every numeric claim in the manuscript (165 checks) against the
shipped CSV/TXT files and prints `PASS/FAIL` per item. Expected result:
**165/165 PASS (FAIL 0, WARN 0)**.

---

## 5. Full reproduction

Run all commands from the repository root.

**Step 1 — benign warm-up** (pre-train the clean global model and collect
the benign update pool / per-client leverage):

```bash
python code/real_fl_exp.py --stage warmup --dataset fmnist
python code/real_fl_exp.py --stage warmup --dataset cifar10
python code/real_fl_exp.py --stage warmup --dataset mnist
```

**Step 2 — stealthy-capacity measurement** η_max(σ):

```bash
python code/real_fl_exp.py --stage r1 --dataset fmnist   # repeat for cifar10, mnist
```

**Step 3 — main six-strategy table** (accuracy drop, malicious-detection
rate, round-detection rate):

```bash
python code/real_fl_exp.py --stage r2 --dataset fmnist   # repeat for cifar10, mnist
```

**Step 4 — allocator runtime** (manuscript Table: computational and
detection overhead):

```bash
python code/timing_allocators.py
```

**Step 5 — theoretical / calibration checks:**

```bash
python code/measure_eta_max.py          # two-regime capacity law  η_max ∝ σ d^{1/4}
python code/measure_safety_threshold.py # safety-threshold boundary
python code/stackelberg.py              # non-monotone defender equilibrium, σ* ≈ 1.9
python code/t4_calibrate.py             # calibrate σ*
python code/optimal_ara.py              # water-filling /顺序统计量分配器 (OSA)
python code/proxy_nk_sweep.py           # proxy n–K sweep
```

**Step 6 — baseline fidelity certificate** (supports manuscript §7.3):

```bash
python code/verify_shejwalkar_fidelity.py
```

**Step 7 (optional) — regenerate figures:**

```bash
python code/fig4_collapse.py
python code/fig6_ablation.py
python code/fig7_r1_tworegime.py
python code/fig8_cifar_grid.py
python code/fig9_dirsens.py
python code/fig10_aggswap.py
python code/fig11_adaptive.py
python code/fig12_crossdataset.py
python code/fig13_pareto_bd.py
```

Figure scripts read from `results/` and write PNG/SVG to `paper/figures/`
or `results/fig/`.

---

## 6. Claim → artifact map

| Manuscript element | Script(s) | Result file(s) |
|---|---|---|
| C/Γ deployability verdict (FashionMNIST 8.8, CIFAR-10 0.48) | `real_fl_exp.py` (r2), `r2_median_validate.py` | `results/real/r2_report_*.txt`, `results/real/median_validate_*.csv` |
| Two-regime capacity law η_max ∝ σ d^{1/4} | `measure_eta_max.py` | `results/eta_max_*.csv`, `results/eta_max_report.txt` |
| Water-filling allocation & order-statistic allocator (OSA) | `optimal_ara.py` | `results/optimal_ara.csv` |
| Non-monotone defender equilibrium σ* ≈ 1.9 | `stackelberg.py`, `t4_calibrate.py` | `results/stackelberg.csv` |
| Allocator runtime (overhead Table) | `timing_allocators.py` | stdout (matches manuscript Table) |
| Pareto front / per-seed dispersion | `r2_median_validate.py`, `fig13_pareto_bd.py` | `results/real/*_seeds.csv` |

---

## 7. Third-party code

`baselines/NDSS21-Model-Poisoning/` is the official code release of
Shejwalkar & Chakraborty, *"Manipulating the Byzantine: Optimizing Model
Poisoning Attacks and Defenses for Federated Learning"* (NDSS 2021). It is
included verbatim only to support the fidelity check in §7.3 of the
manuscript. It retains its original license; the present authors are **not**
the authors of that code.

---

## 8. License

The code is released under the MIT License (see `LICENSE`). No personal
data, human-subject data, or sensitive information is included. Upon
acceptance, a DOI will be archived for this repository and linked from
the manuscript's Data availability statement.
