# -*- coding: utf-8 -*-
"""
选题三 P1 地基实验 v2：测量 η_max(σ) —— 验证 T1 立论（稳健版）
================================================================
立论（DAB-FL 的镜像）：DP 高斯噪声淹没投毒检测；噪声越强(σ↑)，
攻击者可藏身的扰动幅度越大。定义 η_max(σ)=攻击方在不被鲁棒聚合
(坐标中位数)检出的前提下能注入的最大扰动幅度。

v2 改进：用 Fisher 分离度 sep(ρ,σ) = (μ_mal−μ_benign)/σ_pooled 作为
"可检测性"度量（与 DAB-FL 同范式），η_max(σ) 取 sep 达到刚可检测
临界 sep*（默认 2.0）时的 ρ。避免小样本阈值噪声，曲线平滑可插值。

输出：
  results/eta_max_main.csv   η_max(σ) 主曲线
  results/eta_max_d.csv     η_max 随 d 标度
  results/eta_max_d0.csv    η_max 随 d0 标度
  results/eta_max_report.txt 报告
"""

import math
import os
import random
import csv

CFG = dict(
    d=1500, d0=100,
    sigma_s=1.0,      # 良性流形坐标方差(前 d0 维)
    sigma_b=0.15,     # 良性各向同性地板(其余维)
    nb=14, nm=6,      # 良性/恶意客户端数
    sep_star=2.0,     # 刚可检测临界分离度
    trials=60,
    seed=20261027,
)

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTDIR, exist_ok=True)


def median(vals):
    s = sorted(vals); m = len(s)
    return s[m // 2] if m % 2 == 1 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def make_trial(rng, d, d0, sigma_s, sigma_b, n):
    bases, v, xi = [], None, []
    for i in range(n):
        b = [ (sigma_s if j < d0 else sigma_b) * rng.gauss(0.0, 1.0) for j in range(d) ]
        bases.append(b)
    v = [rng.gauss(0.0, 1.0) for _ in range(d)]
    nv = math.sqrt(sum(x * x for x in v)); v = [x / nv for x in v]
    xi = [[rng.gauss(0.0, 1.0) for _ in range(d)] for _ in range(n)]
    return bases, v, xi


def sep_at(bases, v, xi, sigma, rho, d, n, nb):
    """返回 (sep, mean_benign, mean_mal)：sep 越大越可检测。"""
    obs = []
    for i in range(n):
        vb = rho if i >= nb else 0.0
        bi, xii = bases[i], xi[i]
        obs.append([bi[j] + vb * v[j] + sigma * xii[j] for j in range(d)])
    med = [median([obs[i][j] for i in range(n)]) for j in range(d)]
    scores = []
    for i in range(n):
        oi = obs[i]; s = 0.0
        for j in range(d):
            diff = oi[j] - med[j]; s += diff * diff
        scores.append(math.sqrt(s))
    ben = scores[:nb]; mal = scores[nb:]
    mb = sum(ben) / len(ben); mm = sum(mal) / len(mal)
    var = (sum((x - mb) ** 2 for x in ben) + sum((x - mm) ** 2 for x in mal)) / (len(ben) + len(mal))
    std = math.sqrt(var + 1e-12)
    sep = (mm - mb) / std
    return sep, mb, mm


def eta_max_for(rng, d, d0, sigma, sigma_s, sigma_b, n, nb, sep_star, trials, rho_grid):
    seps = []
    for rho in rho_grid:
        acc = 0.0
        for _ in range(trials):
            b, v, x = make_trial(rng, d, d0, sigma_s, sigma_b, n)
            s, _, _ = sep_at(b, v, x, sigma, rho, d, n, nb)
            acc += s
        seps.append(acc / trials)
    # sep 随 ρ 单调上升；找 sep=sep_star 的 ρ（插值）
    eta = None
    for k in range(1, len(rho_grid)):
        a, bv = seps[k - 1], seps[k]
        if (a <= sep_star <= bv) or (bv <= sep_star <= a):
            r0, r1 = rho_grid[k - 1], rho_grid[k]
            if abs(bv - a) < 1e-9:
                eta = r0
            else:
                eta = r0 + (sep_star - a) * (r1 - r0) / (bv - a)
            break
    if eta is None:
        eta = rho_grid[-1] if seps[-1] < sep_star else rho_grid[0]
    return eta, seps


def rho_grid_for(sigma, d, npts=9):
    lo = 0.05 * sigma
    hi = 9.0 * sigma * (d ** 0.25)   # 覆盖可达检测区间
    return [lo * math.exp((k / (npts - 1)) * math.log(hi / lo)) for k in range(npts)]


def run_main():
    c = CFG; rng = random.Random(c["seed"])
    sigma_list = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    rows, rep = [], []
    rep += ["=" * 70,
            " P1 主曲线：η_max(σ)  (d=%d, d0=%d, nb=%d, nm=%d, trials=%d, sep*=%.1f)"
            % (c["d"], c["d0"], c["nb"], c["nm"], c["trials"], c["sep_star"]),
            "=" * 70,
            " %8s %12s %12s %10s" % ("σ", "η_max", "η_max/σ", "η_max/d^.25")]
    for sigma in sigma_list:
        rg = rho_grid_for(sigma, c["d"])
        eta, _ = eta_max_for(rng, c["d"], c["d0"], sigma, c["sigma_s"], c["sigma_b"],
                             c["nb"] + c["nm"], c["nb"], c["sep_star"], c["trials"], rg)
        rows.append((sigma, eta, eta / sigma, eta / (c["d"] ** 0.25)))
        rep.append(" %8.2f %12.4f %12.4f %10.4f" % (sigma, eta, eta / sigma, eta / (c["d"] ** 0.25)))
    with open(os.path.join(OUTDIR, "eta_max_main.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma", "eta_max", "eta_max_over_sigma", "eta_max_over_d025"]); w.writerows(rows)
    rep += ["", " 期望：η_max 随 σ 单调上升；η_max/σ 近似常数(∝σ)；η_max/d^.25 近似常数(∝d^.25)。"]
    return rows, rep


def run_d_sweep():
    c = CFG; rng = random.Random(c["seed"] + 1)
    sigma = 4.0; d_list = [500, 1000, 2000, 3000]
    rows, rep = [], []
    rep += ["", "=" * 70, " P1 标度：η_max 随 d  (固定 d0=%d, σ=%.1f)" % (c["d0"], sigma), "=" * 70,
            " %8s %12s %14s" % ("d", "η_max", "η_max/d^.25")]
    for d in d_list:
        rg = rho_grid_for(sigma, d)
        eta, _ = eta_max_for(rng, d, c["d0"], sigma, c["sigma_s"], c["sigma_b"],
                             c["nb"] + c["nm"], c["nb"], c["sep_star"], 40, rg)
        rows.append((d, eta, eta / (d ** 0.25)))
        rep.append(" %8d %12.4f %14.4f" % (d, eta, eta / (d ** 0.25)))
    with open(os.path.join(OUTDIR, "eta_max_d.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d", "eta_max", "eta_max_over_d025"]); w.writerows(rows)
    return rows, rep


def run_d0_sweep():
    c = CFG; rng = random.Random(c["seed"] + 2)
    sigma = 4.0; d0_list = [25, 100, 400, 800]
    rows, rep = [], []
    rep += ["", "=" * 70, " P1 标度：η_max 随 d0  (固定 d=%d, σ=%.1f)" % (c["d"], sigma), "=" * 70,
            " %8s %12s %14s %14s" % ("d0", "η_max", "η_max·√d0", "η_max·d0^.25")]
    for d0 in d0_list:
        rg = rho_grid_for(sigma, c["d"])
        eta, _ = eta_max_for(rng, c["d"], d0, sigma, c["sigma_s"], c["sigma_b"],
                             c["nb"] + c["nm"], c["nb"], c["sep_star"], 40, rg)
        rows.append((d0, eta, eta * math.sqrt(d0), eta * (d0 ** 0.25)))
        rep.append(" %8d %12.4f %14.4f %14.4f" % (d0, eta, eta * math.sqrt(d0), eta * (d0 ** 0.25)))
    with open(os.path.join(OUTDIR, "eta_max_d0.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d0", "eta_max", "eta_max_times_sqrt_d0", "eta_max_times_d0_25"]); w.writerows(rows)
    return rows, rep


if __name__ == "__main__":
    all_rep = []
    r_main, rep = run_main(); all_rep += rep
    r_d, rep = run_d_sweep(); all_rep += rep
    r_d0, rep = run_d0_sweep(); all_rep += rep
    all_rep += ["", "-" * 70, " 结论速读："]
    if len(r_main) >= 2:
        inc = all(x[1] <= y[1] for x, y in zip(r_main, r_main[1:]))
        all_rep.append("   η_max 随 σ 单调上升？ %s" % ("是 ✓" if inc else "否 ✗"))
        ratios = [r[2] for r in r_main if r[0] > 0]
        all_rep.append("   η_max/σ 区间: [%.3f, %.3f]  (常数⇔∝σ)" % (min(ratios), max(ratios)))
        d25 = [r[3] for r in r_main]
        all_rep.append("   η_max/d^.25 区间: [%.3f, %.3f]  (常数⇔∝d^.25)" % (min(d25), max(d25)))
    txt = "\n".join(all_rep)
    with open(os.path.join(OUTDIR, "eta_max_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
