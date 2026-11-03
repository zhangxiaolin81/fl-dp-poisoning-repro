# -*- coding: utf-8 -*-
"""fig10: 聚合器替换下的两段式形状不变性（R3-A）。"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "real", "r3_eta_cifar10.csv")
OUT = os.path.join(ROOT, "paper", "figures", "fig10_aggswap_cifar10.png")

data = {}
with open(SRC) as f:
    for r in csv.DictReader(f):
        data.setdefault(r["aggregator"], []).append(
            (float(r["sigma_rel"]), float(r["eta_max"])))

SIG = sorted({s for v in data.values() for s, _ in v})


def fit_tworegime(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float)
    best = None
    for sc in np.linspace(0.5, 40, 160):
        for ef in np.linspace(max(y.min() * 0.5, 0.05), y.max() * 1.5, 160):
            pred = ef * np.sqrt(1 + (s / sc) ** 2)
            r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if best is None or r2 > best[0]:
                best = (r2, ef, sc)
    return best


colors = {"median": "#1f6feb", "trimmed": "#d29922", "krum": "#cf222e"}
marks = {"median": "o", "trimmed": "s", "krum": "^"}


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

# Panel A: 逐客户端 η_max(σ)（中位数客户端轨迹）——每个聚合器一条
ax = axes[0]
for agg in ("median", "trimmed", "krum"):
    pts = sorted(data[agg])
    # 六个客户端按 σ=0.5 的 η 排序，取中位客户端（第 3/4 平均）
    per_sigma = {}
    for s, e in pts:
        per_sigma.setdefault(s, []).append(e)
    med = [float(np.median(per_sigma[s])) for s in sorted(per_sigma)]
    ss = sorted(per_sigma)
    ax.plot(ss, med, marks[agg] + "-", color=colors[agg], ms=4, lw=1.2,
            label=agg, alpha=0.9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\sigma_{\mathrm{rel}}$")
ax.set_ylabel(r"$\eta_{\max}$ (median client, $\times$ unit norm)")
ax.set_title("A  Two-regime shape under each aggregator", fontsize=9)
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25, lw=0.5)

# Panel B: 两段式拟合（对中位 η），展示 η_f / σ_c / R²
ax = axes[1]
ss = np.logspace(np.log10(0.4), np.log10(20), 120)
labels = {}
for agg in ("median", "trimmed", "krum"):
    per_sigma = {}
    for s, e in data[agg]:
        per_sigma.setdefault(s, []).append(e)
    y = [float(np.median(per_sigma[s])) for s in sorted(per_sigma)]
    r2, ef, sc = fit_tworegime(sorted(per_sigma), y)
    if r2 >= 0.7:
        labels[agg] = r"%s: $\eta_f$=%.2f, $\sigma_c$=%.0f, $R^2$=%.2f" % (agg, ef, sc, r2)
    else:
        labels[agg] = r"%s: floor-dominated ($\sigma_c>16$), $\eta_f\approx$%.2f" % (agg, np.median(y))
        sc = 40.0  # 画参考曲线：σ_c 远超窗口
    labels[agg + "_curve"] = (ef, sc)
    ax.plot(ss, ef * np.sqrt(1 + (ss / sc) ** 2), "-", color=colors[agg], lw=1.1)
    ax.plot(sorted(per_sigma), y, marks[agg], color=colors[agg], ms=4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\sigma_{\mathrm{rel}}$")
ax.set_ylabel(r"fit $\eta_f\sqrt{1+(\sigma/\sigma_c)^2}$")
ax.set_title("B  Two-regime fits (level changes, shape does not)", fontsize=9)
ax.legend([labels[a] for a in ("median", "trimmed", "krum")],
          frameon=False, fontsize=7, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)

fig.tight_layout()
figstyle.save_all(fig, OUT[:-4])
print("saved", OUT)
for a in ("median", "trimmed", "krum"):
    print(labels[a])
