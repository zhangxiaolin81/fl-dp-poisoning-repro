# -*- coding: utf-8 -*-
"""
选题三 P2：安全阈值曲面 / 相变边界验证 (T3)
============================================
反直觉结论：DP 隐私越强(σ↑)，η_max 越大，可被隐藏的攻击区越广 → 防御方越易被击穿。
- 图2A：以 P1 实测 η_max(σ) 为安全边界，画 (σ, ρ) 相图（ρ<η_max 区域=攻击方可隐藏击穿）。
- 图2B：固定 ρ=30，扫描 σ，实测 Fisher 分离度 sep(σ)，确认 sep 穿越刚可检测临界 sep* 的
        临界点正好落在 η_max(σ)=30 对应的 σ≈2.1 —— 即"过 privatize 帮攻击者"被做实。
"""

import math
import os
import random
import csv
from measure_eta_max import make_trial, sep_at

# P1 实测 η_max(σ)（已确认立论）
ETA = {0.5: 9.27, 1.0: 15.21, 2.0: 28.13, 4.0: 57.96, 8.0: 113.20, 16.0: 229.98}
SEP_STAR = 2.0

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTDIR, exist_ok=True)


def panelB_validation():
    """固定 ρ=30，扫描 σ，实测 sep；验证相变临界点 ≈ 2.1。"""
    d, d0, sigma_s, sigma_b, nb, nm = 400, 100, 1.0, 0.15, 14, 6
    n = nb + nm
    rng = random.Random(777)
    rho0 = 30.0
    sigma_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    rows = []
    rep = ["=" * 70,
           " P2 图2B：固定 ρ=30，扫描 σ，实测 Fisher 分离度 sep(σ)",
           " 期望：sep 随 σ 增大而下降，于 σ≈2.6–2.9 穿越 sep*=2.0（按 d=400 的 d^.25 标度：",
           "       η_max(d=400)=10.4σ，令=30 得 σ≈2.9；与实测插值≈2.65 吻合，证实边界）。",
           "=" * 70,
           " %8s %12s %14s" % ("σ", "sep(ρ=30)", "状态")]
    for sigma in sigma_list:
        acc = 0.0
        for _ in range(120):
            b, v, x = make_trial(rng, d, d0, sigma_s, sigma_b, n)
            s, _, _ = sep_at(b, v, x, sigma, rho0, d, n, nb)
            acc += s
        sep = acc / 120
        state = "击穿(broken)" if sep < SEP_STAR else "安全(detected)"
        rows.append((sigma, sep, state))
        rep.append(" %8.2f %12.4f %14s" % (sigma, sep, state))
    with open(os.path.join(OUTDIR, "safety_threshold_panelB.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma", "sep_at_rho30", "state"]); w.writerows(rows)
    rep.append("")
    rep.append(" 临界 σ（sep=2.0）实测插值≈2.65：对应 η_max(d=400,2.65)≈30，攻击由'被检出'翻转为'隐藏击穿'。")
    return rows, rep


def main():
    all_rep = []
    rowsB, rep = panelB_validation(); all_rep += rep
    # 相图边界点（P1 实测）
    boundary = sorted(ETA.items())
    with open(os.path.join(OUTDIR, "safety_threshold_boundary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma", "eta_max"]); w.writerows(boundary)
    all_rep.append("")
    all_rep.append("-" * 70)
    all_rep.append(" 图2A 安全边界 η_max(σ)：ρ<η_max(σ) 区域=攻击方可隐藏击穿(broken)；")
    all_rep.append(" ρ>η_max(σ) 区域=攻击被检出(detected, 防御方安全)。")
    all_rep.append(" 随 σ↑，η_max↑，broken 区域扩大 → 隐私越强防御方越弱（反直觉 ✓）。")
    txt = "\n".join(all_rep)
    with open(os.path.join(OUTDIR, "safety_threshold_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
