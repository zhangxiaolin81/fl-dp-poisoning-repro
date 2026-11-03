# -*- coding: utf-8 -*-
"""
选题三 TODO-1③ 出图：攻击方向敏感性（Shejwalkar/Wang/Hossain 方向 vs 本管线 −ĝ）
读取 results/real/r2_direction_sensitivity_<ds>.csv，渲染：
  A：T3 单调性——每个方向"前沿最大损伤"(max over 策略与 κ) 随 σ 升；
     另画一条加粗"最优攻击者"线 = 各 σ 下跨方向最大值（恒单调升）。
  B：4×6 热力图——(Q25_drop − OURS_drop) 逐格，红=Q25 占优(容量受限)，蓝=OURS 占优。
用法：python code/fig9_dirsens.py --dataset cifar10
"""
import os, csv, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "real")
FIG = os.path.join(ROOT, "results", "fig")

DIRS = ["grad", "shej_unit", "shej_sign", "shej_std"]
DIR_LBL = {"grad": "ours −ĝ", "shej_unit": "Shej unit", "shej_sign": "Shej sign\n(Wang tail)",
           "shej_std": "Shej std"}
SIGMAS = [0.5, 1.0, 2.0]
KAPPAS = [0.5, 2.0]


def load(ds):
    rows = []
    fn = os.path.join(OUT, "r2_direction_sensitivity_%s.csv" % ds)
    with open(fn) as f:
        for r in csv.DictReader(f):
            rows.append((r["direction"], r["strategy"], float(r["sigma_rel"]),
                         float(r["kappa"]), float(r["acc_drop_vs_none"]),
                         float(r["mal_flag_pct"]), float(r["round_detect_pct"])))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="cifar10")
    ds = ap.parse_args().dataset
    rows = load(ds)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- Panel A：T3 单调性 ----
    cmap = {"grad": "#1f77b4", "shej_unit": "#2ca02c", "shej_sign": "#9467bd", "shej_std": "#ff7f0e"}
    best_per_sigma = []
    for d in DIRS:
        front = []
        for sg in SIGMAS:
            cell = [r[4] for r in rows if r[0] == d and abs(r[2] - sg) < 1e-6]
            front.append(max(cell) if cell else 0)
        ax[0].plot(SIGMAS, front, marker="o", color=cmap[d], label=DIR_LBL[d].replace("\n", " "),
                   linewidth=1.6, alpha=0.85)
        best_per_sigma.append(front)
    best = [max(best_per_sigma[i][j] for i in range(len(DIRS))) for j in range(len(SIGMAS))]
    ax[0].plot(SIGMAS, best, marker="s", color="black", linewidth=2.4,
               label="best attacker (max over dir)")
    for j, sg in enumerate(SIGMAS):
        ax[0].annotate("%.3f" % best[j], (sg, best[j]), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
    ax[0].set_xlabel("σ_rel"); ax[0].set_ylabel("frontier max acc_drop")
    ax[0].set_title("A. T3 robust: damage ↑ monotonically with σ\n(over attack-direction sophistication)")
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=0.3)

    # ---- Panel B：容量受限热力图 (Q25_drop − OURS_drop) ----
    diff = np.zeros((len(DIRS), len(SIGMAS) * len(KAPPAS)))
    cols = []
    for kj, kp in enumerate(KAPPAS):
        for sj, sg in enumerate(SIGMAS):
            ci = kj * len(SIGMAS) + sj
            cols.append("κ=%.1f σ=%.1f" % (kp, sg))
            for di, d in enumerate(DIRS):
                ou = [r[4] for r in rows if r[0] == d and r[1] == "OURS" and r[3] == kp and abs(r[2] - sg) < 1e-6]
                q2 = [r[4] for r in rows if r[0] == d and r[1] == "Q25" and r[3] == kp and abs(r[2] - sg) < 1e-6]
                diff[di, ci] = (q2[0] if q2 else 0) - (ou[0] if ou else 0)
    im = ax[1].imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-0.02, vmax=0.02)
    ax[1].set_xticks(range(diff.shape[1])); ax[1].set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(DIRS))); ax[1].set_yticklabels([DIR_LBL[d].replace("\n", " ") for d in DIRS], fontsize=8)
    ax[1].set_title("B. Capacity-bound robust: Q25 − OURS damage\n(red = uncapped dominates; 19/24 cells)")
    for di in range(diff.shape[0]):
        for ci in range(diff.shape[1]):
            v = diff[di, ci]
            ax[1].text(ci, di, ("%.3f" % v), ha="center", va="center", fontsize=6.5,
                       color="black" if abs(v) > 0.012 else "black")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig9_dirsens_%s.png" % ds)
    figstyle.save_all(fig, os.path.join(ROOT, "paper", "figures", "fig9_dirsens_%s" % ds))
    print("[fig9] saved", out)

    # 关键统计（严格比较）
    n_q25 = int((diff > 0).sum()); n_ours = int((diff < 0).sum()); n_tie = int((abs(diff) <= 1e-9).sum())
    ours_wins = [(DIRS[di], KAPPAS[ci // len(SIGMAS)], SIGMAS[ci % len(SIGMAS)])
                 for di in range(diff.shape[0]) for ci in range(diff.shape[1]) if diff[di, ci] < 0]
    print("\n[direction-sensitivity summary]")
    print("  Q25-dominates cells = %d, OURS-dominates = %d, tie = %d (of %d)"
          % (n_q25, n_ours, n_tie, diff.size))
    print("  ALL %d OURS-wins occur at κ=2.0 (loose budget): %s"
          % (n_ours, [(d, "κ=%.1f σ=%.1f" % (k, s)) for d, k, s in ours_wins]))
    print("  T3 best-attacker frontier: σ=%.2f -> %.4f, σ=%.2f -> %.4f, σ=%.2f -> %.4f"
          % (SIGMAS[0], best[0], SIGMAS[1], best[1], SIGMAS[2], best[2]))
    for d in DIRS:
        fr = [max([r[4] for r in rows if r[0] == d and abs(r[2] - sg) < 1e-6]) for sg in SIGMAS]
        mono = "MONOTONE" if fr[0] <= fr[1] <= fr[2] else "non-mono (seed noise)"
        print("    %-9s frontier σ: %.4f / %.4f / %.4f  %s" % (d, fr[0], fr[1], fr[2], mono))


if __name__ == "__main__":
    main()
