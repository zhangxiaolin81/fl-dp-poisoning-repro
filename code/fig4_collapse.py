# -*- coding: utf-8 -*-
"""
选题三 图4 数据：标度律坍缩验证 sep ∝ ρ²/(σ²·√d)
================================================
理论（T1 推导）：Fisher 分离度 sep(ρ,σ,d) = c · ρ² / (σ²·√d)。
若把所有 (ρ,σ,d) 组合的 sep 对 x = ρ²/(σ²·√d) 作图，应坍缩到
过原点直线 y = c·x —— 这即"标度律镜像"（攻击侧读法：
η_max = sqrt(sep*·σ²·√d / c) ∝ σ·d^¼）。
输出：results/fig4_collapse.csv, results/fig4_report.txt
"""
import math
import random
import csv
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTDIR, exist_ok=True)


def median(vals):
    s = sorted(vals); m = len(s)
    return s[m // 2] if m % 2 == 1 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def sep_once(rng, d, d0, sigma_s, sigma_b, n, nb, sigma, rho):
    bases = [[(sigma_s if j < d0 else sigma_b) * rng.gauss(0.0, 1.0) for j in range(d)] for _ in range(n)]
    v = [rng.gauss(0.0, 1.0) for _ in range(d)]
    nv = math.sqrt(sum(x * x for x in v)); v = [x / nv for x in v]
    xi = [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(n)]
    obs = [[bases[i][j] + (rho * v[j] if i >= nb else 0.0) + sigma * xi[i][j] for j in range(d)] for i in range(n)]
    med = [median([obs[i][j] for i in range(n)]) for j in range(d)]
    scores = []
    for i in range(n):
        s = 0.0
        oi = obs[i]
        for j in range(d):
            diff = oi[j] - med[j]; s += diff * diff
        scores.append(math.sqrt(s))
    ben, mal = scores[:nb], scores[nb:]
    mb = sum(ben) / len(ben); mm = sum(mal) / len(mal)
    var = (sum((x - mb) ** 2 for x in ben) + sum((x - mm) ** 2 for x in mal)) / (nb + len(mal))
    return (mm - mb) / math.sqrt(var + 1e-12)


def main():
    rng = random.Random(31415926)
    d0, sigma_s, sigma_b, nb, nm = 100, 1.0, 0.15, 14, 6
    n = nb + nm
    combos = [(s, d, 24 if d <= 1500 else 14) for s in (2.0, 4.0, 8.0) for d in (600, 1500, 3000)]
    x_targets = [0.03, 0.10, 0.30, 1.0, 3.0, 10.0]
    rows, rep = [], ["=" * 70, " 图4 标度律坍缩：sep vs x = ρ²/(σ²·√d)，理论 sep = c·x", "=" * 70]
    for sigma, d, trials in combos:
        for xt in x_targets:
            rho = math.sqrt(xt * sigma * sigma * math.sqrt(d))
            acc = sum(sep_once(rng, d, d0, sigma_s, sigma_b, n, nb, sigma, rho) for _ in range(trials))
            sep = acc / trials
            rows.append((sigma, d, rho, xt, sep))
            rep.append(" σ=%4.1f d=%5d ρ=%8.2f  x=%6.3f  sep=%7.4f" % (sigma, d, rho, xt, sep))
    # 坍缩直线拟合 sep = c·x（过原点最小二乘）
    num = sum(r[4] * r[3] for r in rows); den = sum(r[3] ** 2 for r in rows)
    c = num / den
    ss_res = sum((r[4] - c * r[3]) ** 2 for r in rows)
    ss_tot = sum((r[4] - sum(x[4] for x in rows) / len(rows)) ** 2 for r in rows)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rep += ["", " 过原点拟合：sep = %.4f · x,  R² = %.4f" % (c, r2),
            " （R²≈1 ⇒ 坍缩成立 ⇒ η_max ∝ σ·d^¼ 的理论推导被数据支持）"]
    with open(os.path.join(OUTDIR, "fig4_collapse.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma", "d", "rho", "x", "sep"]); w.writerows(rows)
    txt = "\n".join(rep)
    with open(os.path.join(OUTDIR, "fig4_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
