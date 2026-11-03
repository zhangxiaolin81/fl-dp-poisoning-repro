# -*- coding: utf-8 -*-
"""
选题三 fig8：CIFAR-10 R2 网格出图（跨数据集 + σ/κ 网格 + 10 种子）
================================================================
双面板（审稿压缩清单 4.5：原 A 面板 damage-envelope 热图已删）：
  a. stealthy-damage 权衡前沿散点（mal_flag% × acc_drop），72 点按策略着色 + Pareto 前沿
  b. 容量受限机制 bar：C=Ση_i²（隐蔽容量）vs Γ=κ·K·unit²，展示 FMNIST(可充分部署) ↔ CIFAR(容量受限) regime 翻转

产物：results/fig/fig8_cifar_grid.png  +  控制台打印关键统计量
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "real")
FIG = os.path.join(ROOT, "results", "fig")
os.makedirs(FIG, exist_ok=True)

STRATS = ["OURS", "S21", "W25", "H25", "Q25", "UNI"]
COLORS = {"OURS": "#1f77b4", "S21": "#ff7f0e", "W25": "#2ca02c",
          "H25": "#9467bd", "Q25": "#d62728", "UNI": "#8c564b"}
SIGMA = [0.5, 1.0, 2.0]
KAPPA = [0.25, 0.5, 1.0, 2.0]


def load(ds):
    rows = []
    fn = "r2_main.csv" if ds == "fmnist" else "r2_main_%s.csv" % ds
    with open(os.path.join(OUT, fn)) as f:
        for r in csv.DictReader(f):
            s = float(r["sigma_rel"]) if "sigma_rel" in r else 1.0
            rows.append((r["strategy"], s, float(r["kappa"]),
                         float(r["acc_drop_vs_none"]), float(r["mal_flag_pct"])))
    return rows


def pareto(drops, flags):
    """返回 Pareto 前沿点索引：drop 最大化且 flag 最小化（upper-left）。"""
    pts = sorted(zip(drops, flags), key=lambda t: (-t[0], t[1]))
    front = []
    best_flag = 1e9
    for d, fl in pts:
        if fl < best_flag - 1e-9:
            front.append((d, fl)); best_flag = fl
    return front


def main():
    cif = load("cifar10")
    fmn = load("fmnist")

    # ---------- 统计量 ----------
    # T3：每个 σ 下前沿最大损伤
    t3 = {}
    for s in SIGMA:
        best = max(d for (_, ss, _, d, _) in cif if abs(ss - s) < 1e-6)
        t3[s] = best
    print("[T3] CIFAR 前沿最大损伤随 σ： " + ", ".join("σ=%.1f→%.4f" % (s, t3[s]) for s in SIGMA))

    # 各策略在 CIFAR 上平均损伤 / 平均检出
    print("[策略均值] CIFAR:")
    for st in STRATS:
        sub = [d for (s, _, _, d, _) in cif if s == st]
        fl = [f for (s, _, _, _, f) in cif if s == st]
        print("   %4s  mean_drop=%.4f  mean_flag=%.1f" % (st, np.mean(sub), np.mean(fl)))

    # Pareto 前沿（全部 72 点）
    drops = [d for (_, _, _, d, _) in cif]
    flags = [f for (_, _, _, _, f) in cif]
    front = pareto(drops, flags)
    front.sort(key=lambda t: t[1])
    print("[Pareto] CIFAR 前沿 %d 点：" % len(front) +
          ", ".join("(%.4f,%.1f)" % p for p in front))

    # ---------- 图（双面板：审稿压缩清单 4.5 —— 原 A 面板热图删除，
    # 其信息由正文 upper-envelope 数字与 C/Γ 分析承载；保留 Pareto 前沿 + C/Γ flip） ----------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    plt.rcParams.update({"font.size": 11})

    # Panel a（原 B）：stealthy-damage 前沿散点
    for st in STRATS:
        xs = [f for (s, _, _, _, f) in cif if s == st]
        ys = [d for (s, _, _, d, _) in cif if s == st]
        ax[0].scatter(xs, ys, s=46, c=COLORS[st], label=st, alpha=0.85,
                      edgecolors="white", linewidths=0.5)
    fx = [p[1] for p in front]; fy = [p[0] for p in front]
    ax[0].plot(fx, fy, "k--", lw=1.4, alpha=0.7)
    ax[0].scatter(fx, fy, c="black", s=30, zorder=5)
    ax[0].set_xlabel("malicious flag rate (%)  ↓ stealthier")
    ax[0].set_ylabel("hidden accuracy drop ΔAcc  ↑ stronger")
    ax[0].set_title("a. Stealthy-damage trade-off frontier\n(CIFAR-10, 72 runs; lower-left = stealthy-efficient)")
    ax[0].legend(fontsize=9, ncol=3, loc="upper right")
    ax[0].grid(alpha=0.25)

    # Panel b（原 C）：容量受限机制
    # FMNIST σ=1: η_i=[1.24,1.67,1.42,1.55,1.35,1.75], unit=0.36
    # CIFAR σ=2:  η_i=[0.13,1.64,0.77,1.61,1.62,0.06], unit=1.214
    eta_fmn = np.array([1.24, 1.67, 1.42, 1.55, 1.35, 1.75])
    eta_cif = np.array([0.13, 1.64, 0.77, 1.61, 1.62, 0.06])
    K, unit_fmn, unit_cif = 6, 0.36, 1.214
    kappa = 2.0
    C_fmn = float((eta_fmn ** 2).sum())
    C_cif = float((eta_cif ** 2).sum())
    G_fmn = kappa * K * unit_fmn ** 2
    G_cif = kappa * K * unit_cif ** 2
    labels = ["FMNIST\n(σ=1, κ=2)", "CIFAR-10\n(σ=2, κ=2)"]
    C_vals = [C_fmn, C_cif]
    G_vals = [G_fmn, G_cif]
    x = np.arange(2); w = 0.35
    b1 = ax[1].bar(x - w / 2, C_vals, w, color="#2ca02c")
    b2 = ax[1].bar(x + w / 2, G_vals, w, color="#d62728")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("magnitude (log scale)")
    ax[1].set_title("b. Capacity-bound regime flip", pad=10)
    # 直接在柱上方标注符号，组上方标注 ratio + regime，避免图例遮挡
    for i, (c, g) in enumerate(zip(C_vals, G_vals)):
        ax[1].text(i - w / 2, c * 1.12, "C", ha="center", fontsize=10, color="#2ca02c", fontweight="bold")
        ax[1].text(i + w / 2, g * 1.12, "Γ", ha="center", fontsize=10, color="#d62728", fontweight="bold")
        ratio = c / g
        verdict = "deployable" if ratio >= 1 else "capacity-bound"
        ax[1].text(i, max(c, g) * 1.55, "C:Γ = %.2f\n(%s)" % (ratio, verdict),
                   ha="center", fontsize=9, fontweight="bold")
    ax[1].text(0.5, -0.30, "green = C = Σηᵢ² (aggregate stealth capacity)   ·   red = Γ = κ·K·unit² (nominal budget)",
               transform=ax[1].transAxes, ha="center", fontsize=8.5, color="dimgray")
    print("[容量] FMNIST C=%.2f Γ=%.2f ratio=%.2f (deployable)" % (C_fmn, G_fmn, C_fmn / G_fmn))
    print("[容量] CIFAR  C=%.2f Γ=%.2f ratio=%.2f (capacity-bound)" % (C_cif, G_cif, C_cif / G_cif))

    plt.tight_layout()
    figstyle.save_all(fig, os.path.join(ROOT, "paper", "figures", "fig8_cifar_grid"))


if __name__ == "__main__":
    main()
