# -*- coding: utf-8 -*-
"""
代理框架 N/K 规模扩展（审稿 Major 5：更大 N,K）
================================================
在 p4_baselines.py 的同一测量框架内，把 (N,K) 从参考设定 (30,8) 扩展到
(20,6)/(30,8)/(50,12)/(100,20)，每档 10 个场景种子，Γ=500·K（保持每恶意
客户端预算不变），扫 σ∈{0.5..16}，检验结论不变性：
  1) OURS 全 (σ,Γ,seed) 不塌（影响 ≥ 最强基线 - 容差）；
  2) 各基线 distinct 失效模式跨 (N,K) 保持。
输出：results/proxy_nk_sweep.csv + 报告打印
"""
import math, random, csv, os, statistics as st
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from p4_baselines import strategy_p, SELECT, NAMES, eta_of, realized

OUTDIR = os.path.join(os.path.dirname(sys_path), "results")
os.makedirs(OUTDIR, exist_ok=True)

GRID = [(20, 6), (30, 8), (50, 12), (100, 20)]
N_SEEDS = 10
SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
TOL = 1e-6  # 平局容差


def run_one(N, K, seed):
    rng = random.Random(1000 * N + seed)  # 每 (N,K) 独立场景流
    alpha = [max(0.2, rng.lognormvariate(0.0, 0.8)) for _ in range(N)]
    u = [rng.uniform(0.5, 1.5) for _ in range(N)]
    Gamma = 500.0 * K
    cells = {}
    for s in SIGMAS:
        for nm in NAMES:
            sel = SELECT[nm](alpha, u, K)
            a_sel = [alpha[i] for i in sel]
            u_sel = [u[i] for i in sel]
            p = strategy_p(nm, a_sel, u_sel, Gamma, s, K)
            eta = eta_of(u_sel, s)
            cells[(nm, s)] = realized(a_sel, p, eta)
    return cells


rows_csv = []
rep = ["=" * 96,
       " 代理框架 (N,K) 扩展：Γ=500·K, σ∈{0.5..16}, %d 场景种子/格, 影响=Σα_i·p_i·1[p_i≤η_i]" % N_SEEDS,
       "  判据：OURS 不塌 = 在每个 σ 上 OURS ≥ max(基线) - 容差（按种子均值计）",
       "=" * 96]
for N, K in GRID:
    agg = {nm: {s: [] for s in SIGMAS} for nm in NAMES}
    collapse_cnt = {nm: 0 for nm in NAMES if nm != "OURS"}
    n_cells = 0
    for seed in range(N_SEEDS):
        cells = run_one(N, K, seed)
        for s in SIGMAS:
            for nm in NAMES:
                agg[nm][s].append(cells[(nm, s)])
        n_cells += 1
    rep.append("-" * 96)
    rep.append(" N=%3d K=%2d  (各 σ 上 6 策略平均影响，10 场景种子)" % (N, K))
    header = "     σ=%6s" % "" + "".join("%9s" % n for n in NAMES)
    rep.append(header)
    for s in SIGMAS:
        m = {nm: st.mean(agg[nm][s]) for nm in NAMES}
        best_base = max(m[nm] for nm in NAMES if nm != "OURS")
        for nm in NAMES:
            if nm != "OURS" and m[nm] < m["OURS"] - TOL:
                collapse_cnt[nm] += 1
        rows_csv.append([N, K, s] + ["%.2f" % m[nm] for nm in NAMES] +
                        ["%.3f" % (m["OURS"] / best_base) if best_base > 0 else "inf"])
        rep.append("     σ=%6.1f" % s + "".join("%9.2f" % m[nm] for nm in NAMES))
    ratio = []
    for s in SIGMAS:
        m = {nm: st.mean(agg[nm][s]) for nm in NAMES}
        bb = max(m[nm] for nm in NAMES if nm != "OURS")
        ratio.append(m["OURS"] / bb if bb > 0 else float("inf"))
    rep.append("  OURS/最强基线 比（按 σ）: %s" % ", ".join("%.2f" % r for r in ratio))
    rep.append("  塌方格数（基线 < OURS 视为基线塌；%d 个 σ 格）: %s" % (
        len(SIGMAS), {k: v for k, v in collapse_cnt.items() if v > 0} or "无"))

txt = "\n".join(rep)
print(txt)
with open(os.path.join(OUTDIR, "proxy_nk_sweep.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "K", "sigma"] + NAMES + ["OURS_over_best_base"])
    w.writerows(rows_csv)
with open(os.path.join(OUTDIR, "proxy_nk_sweep_report.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
