# -*- coding: utf-8 -*-
"""
选题三 图6 数据：消融实验（T2 各组件边际贡献）
================================================
全量 Optimal-ARA = 客户端选择(top-K by α·η) + water-filling 封顶 + 预算优化。
消融变体：
  full        全量
  -select     随机选 K 个（去掉 α·η 选择准则）
  -cap        去掉隐形上界封顶（= Qiu 式，保留选择）
  -budget     去掉预算优化（等分 p=sqrt(Γ/K)，保留选择与封顶）
  sel-a       只按 α 选择（不知 η）
  sel-e       只按 η 选择（不知 α）
输出：results/fig6_ablation.csv, results/fig6_report.txt
"""
import math
import random
import csv
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTDIR, exist_ok=True)

C_ETA = 14.5


def water_filling(alpha, eta, Gamma):
    lo, hi = 1e-9, max(alpha) / 1e-9 + 1.0
    for _ in range(80):
        mu = 0.5 * (lo + hi)
        p = [min(e, a / (2 * mu)) if a > 0 else 0.0 for a, e in zip(alpha, eta)]
        if sum(x * x for x in p) > Gamma:
            lo = mu
        else:
            hi = mu
    mu = 0.5 * (lo + hi)
    return [min(e, a / (2 * mu)) if a > 0 else 0.0 for a, e in zip(alpha, eta)]


def realized(alpha, p, eta):
    return sum(a * x for a, x, e in zip(alpha, p, eta) if x <= e + 1e-9)


def variant(name, alpha, u, K, Gamma, rng_abl):
    n = len(alpha)
    if name == "full" or name == "-cap" or name == "-budget":
        sel = sorted(range(n), key=lambda i: alpha[i] * C_ETA * u[i], reverse=True)[:K]
    elif name == "-select":
        sel = sorted(rng_abl.sample(range(n), K))
    elif name == "sel-a":
        sel = sorted(range(n), key=lambda i: alpha[i], reverse=True)[:K]
    elif name == "sel-e":
        sel = sorted(range(n), key=lambda i: C_ETA * u[i], reverse=True)[:K]
    a_s = [alpha[i] for i in sel]; e_s = [C_ETA * u[i] for i in sel]
    if name == "-cap":
        s2 = sum(x * x for x in a_s)
        lam = math.sqrt(Gamma / s2) if s2 > 0 else 0.0
        p = [lam * a for a in a_s]
    elif name == "-budget":
        p_each = math.sqrt(Gamma / K)
        p = [min(e, p_each) for e in e_s]
    else:
        p = water_filling(a_s, e_s, Gamma)
    return realized(a_s, p, e_s)


def main():
    rng = random.Random(2026)
    rng_abl = random.Random(555)
    n, K = 30, 8
    alpha = [max(0.2, rng.lognormvariate(0.0, 0.8)) for _ in range(n)]
    u = [rng.uniform(0.5, 1.5) for _ in range(n)]
    names = ["full", "-select", "-cap", "-budget", "sel-a", "sel-e"]
    gammas = [500, 1000, 2000, 4000, 8000, 16000]
    rows, rep = [], ["=" * 84,
                     " 图6 消融：各组件边际贡献（n=%d, K=%d, η_i=%.1f·u_i）" % (n, K, C_ETA),
                     " 影响 = Σ α_i·p_i·1[p_i≤η_i]",
                     "=" * 84,
                     " %8s" % "Gamma" + "".join("%10s" % nm for nm in names)]
    for G in gammas:
        vals = [variant(nm, alpha, u, K, G, rng_abl) for nm in names]
        rows.append((G, *vals))
        rep.append(" %8d" % G + "".join("%10.2f" % v for v in vals))
    # 汇总各消融相对 full 的平均损失
    # 注意：rows 元组为 (G, full, -select, -cap, -budget, sel-a, sel-e)，
    # r[0]=Gamma、r[1]=full，故分母取 r[1]、变体列取 r[k+1]。
    rep += ["", " 相对 full 的平均损失%："]
    for k, nm in enumerate(names[1:], start=1):
        losses = [100 * (1 - r[k + 1] / r[1]) for r in rows if r[1] > 0]
        rep.append("   %-8s %6.1f%%" % (nm, sum(losses) / len(losses)))
    rep += ["", " 读法：-cap 在封顶普及时损失最大 ⇒ water-filling 封顶是核心机制；",
            "       -select 平均损失最大 ⇒ 客户端选择准则(α·η)贡献独立且稳定；",
            "       -budget 在中段 Γ(1000–2000) 有损失、大 Γ 时与 full 重合（预算≫容量时",
            "       water-filling 退化为全封顶，等分封顶≈最优，数学上必然）；",
            "       注意：小 Γ(500) 时 sel-a 略优于 full——无封顶约束时最优选择准则是纯 α，",
            "       α·η 准则的价值在封顶起作用的 regime 才显现（诚实呈现，不做普遍性宣称）。"]
    with open(os.path.join(OUTDIR, "fig6_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Gamma"] + names); w.writerows(rows)
    txt = "\n".join(rep)
    with open(os.path.join(OUTDIR, "fig6_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
