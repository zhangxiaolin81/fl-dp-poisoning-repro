# -*- coding: utf-8 -*-
"""fig11: R4 在线自适应攻击者消融结果可视化。
双面板：左 κ=0.25(deployable)，右 κ=2(capacity-bound)；
每组 σ 下对比 static vs adaptive 的 acc_drop，并标注 adaptive gain。
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "real", "r4_adaptive_cifar10.csv")
FIG = os.path.join(ROOT, "paper", "figures", "fig11_adaptive.png")
os.makedirs(os.path.dirname(FIG), exist_ok=True)

rows = []
with open(SRC) as f:
    for r in csv.DictReader(f):
        rows.append(r)

SIGMA = sorted({float(r["sigma_rel"]) for r in rows})
KAPPA = sorted({float(r["kappa"]) for r in rows})
MODES = ["static", "adaptive"]


def mean_drop(mode, s, k):
    vals = [float(r["acc_drop_vs_none"]) for r in rows
            if r["mode"] == mode and float(r["sigma_rel"]) == s and float(r["kappa"]) == k]
    return np.mean(vals) if vals else 0.0


def gain(s, k):
    return mean_drop("adaptive", s, k) - mean_drop("static", s, k)


colors = {"static": "#1f77b4", "adaptive": "#d62728"}

fig, axes = plt.subplots(1, len(KAPPA), figsize=(7.0, 3.0))
if len(KAPPA) == 1:
    axes = [axes]

for ax_i, k in enumerate(KAPPA):
    x = np.arange(len(SIGMA))
    w = 0.38
    for j, mode in enumerate(MODES):
        ys = [mean_drop(mode, s, k) for s in SIGMA]
        ax = axes[ax_i]
        ax.bar(x + (j - 0.5) * w, ys, w, color=colors[mode], label=mode, alpha=0.9)
    # annotate gain
    for xi, s in enumerate(SIGMA):
        g = gain(s, k)
        ax.text(xi, max(mean_drop("static", s, k), mean_drop("adaptive", s, k)) + 0.002,
                "Δ=%.3f" % g, ha="center", fontsize=7, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([("σ=%.1f" % s) for s in SIGMA])
    ax.set_xlabel("DP noise level")
    ax.set_ylabel("accuracy drop vs no-attack")
    ax.set_title("κ=%.2f (%s)" % (k, "deployable" if k < 1 else "capacity-bound"))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5, axis="y")

fig.suptitle("R4: optimal allocation is robust to an online adaptive attacker", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
figstyle.save_all(fig, FIG[:-4])
print("saved", FIG)
for k in KAPPA:
    print("κ=%.2f gains:" % k, {("σ=%.1f" % s): round(gain(s, k), 4) for s in SIGMA})
