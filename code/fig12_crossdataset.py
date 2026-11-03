# -*- coding: utf-8 -*-
"""fig12: 跨数据集验证——CIFAR-10 vs MNIST 真实管线。
Panel A: R1 两段式容量律 η_max(σ)（双数据集，含 η∝σ 朴素参考）。
Panel B: R2 攻击区间一致性——κ=2(capacity-bound) 下 OURS 与无约束分配器的
        损伤对比，验证『C<Γ 时 stealth-aware 最优被无约束超越』在两数据集都成立。
（MNIST 的 r2 文件若尚未生成，Panel B 自动跳过并打印提示。）
"""
import os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = os.path.join(ROOT, "results", "real")
FIG = os.path.join(ROOT, "paper", "figures", "fig12_crossdataset.png")
os.makedirs(os.path.dirname(FIG), exist_ok=True)

DS_META = {
    "cifar10": ("CIFAR-10", "#1f77b4", "r1_eta_max_cifar10.csv"),
    "mnist":   ("MNIST", "#d62728", "r1_eta_max_mnist.csv"),
}


def load_r1(fn):
    s, e, r = [], [], []
    with open(os.path.join(RR, fn)) as f:
        for row in csv.DictReader(f):
            s.append(float(row["sigma_rel"]))
            e.append(float(row["eta_max_unitnorms"]))
            r.append(float(row["eta_over_sigma"]))
    return np.array(s), np.array(e), np.array(r)


def load_r2(fn):
    asr = {}
    with open(os.path.join(RR, fn)) as f:
        for row in csv.DictReader(f):
            if float(row["kappa"]) != 2.0:
                continue
            asr.setdefault(float(row["sigma_rel"]), {}).setdefault(
                row["strategy"], []).append(float(row["acc_drop_vs_none"]))
    return {s: {k: np.mean(v) for k, v in d.items()} for s, d in asr.items()}



fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

# Panel A: R1
ax = axes[0]
has_r1 = True
for ds, (label, color, fn) in DS_META.items():
    p = os.path.join(RR, fn)
    if not os.path.exists(p):
        has_r1 = False
        continue
    s, e, r = load_r1(fn)
    ax.plot(s, e, "o-", color=color, lw=1.6, ms=5, label=label)
sx = np.linspace(0.4, 2.5, 50)
ax.plot(sx, sx, "k--", lw=1.3, label=r"$\eta_{\max}\propto\sigma$ (naive)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\sigma_{\mathrm{rel}}$")
ax.set_ylabel(r"$\eta_{\max}$ ($\times$ unit norm)")
ax.set_title("A  Two-regime law holds on both datasets", fontsize=9)
ax.legend(frameon=False, fontsize=7.5)
ax.grid(alpha=0.25, lw=0.5, which="both")
if not has_r1:
    ax.text(0.5, 0.5, "R1 data missing", transform=ax.transAxes, ha="center")

# Panel B: R2 regime consistency (κ=2)
ax = axes[1]
r2c = load_r2("r2_main_cifar10.csv") if os.path.exists(os.path.join(RR, "r2_main_cifar10.csv")) else {}
r2m = load_r2("r2_main_mnist.csv") if os.path.exists(os.path.join(RR, "r2_main_mnist.csv")) else {}
if r2c and r2m:
    sigs = sorted(set(r2c) & set(r2m))
    for ds, r2, color, label in (("cifar10", r2c, "#1f77b4", "CIFAR-10"),
                                 ("mnist", r2m, "#d62728", "MNIST")):
        ours = [r2[s].get("OURS", 0) for s in sigs]
        unc = [max(r2[s].get(k, 0) for k in ("Q25", "W25", "UNI")) for s in sigs]
        ax.plot(sigs, ours, "o-", color=color, lw=1.4, ms=4, label="%s OURS" % label)
        ax.plot(sigs, unc, "s--", color=color, lw=1.0, ms=3, alpha=0.7,
                label="%s unconstrained" % label)
    ax.set_xlabel(r"$\sigma_{\mathrm{rel}}$")
    ax.set_ylabel("accuracy drop (κ=2, capacity-bound)")
    ax.set_title("B  Accuracy drop, capacity-bound (κ=2)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(alpha=0.25, lw=0.5)
else:
    ax.text(0.5, 0.5, "r2_main_mnist.csv not ready yet", transform=ax.transAxes, ha="center")
    ax.set_title("B  (pending MNIST r2)", fontsize=9)

fig.tight_layout()
figstyle.save_all(fig, FIG[:-4])
print("saved", FIG, "| R1 ok:", has_r1, "| R2 ok:", bool(r2c and r2m))
