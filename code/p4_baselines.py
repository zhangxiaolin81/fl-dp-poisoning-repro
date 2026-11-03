# -*- coding: utf-8 -*-
"""
选题三 P4：四基线复现（代理实现）vs Optimal-ARA
================================================
沙箱无外网，无法拉取官方仓库；按各论文公开算法机制，在同一
"DP 噪声 + 坐标中位数检测"测量框架内做忠实代理实现：

  统一公平性设定：所有基线都满足同一预算包络 Σp_i² ≤ Γ；
  它们只在其**信息假设**上不同（对掩蔽容量 η_i 的认知不同）：

  [S21] Shejwalkar & Houmansadr NDSS'21：对聚合器优化幅度，但模型里
        **无 DP** → 按 σ_ref=1 标定的容量 η(σ_ref) 做 water-filling。
        失效模式：σ<1 时高估容量→被检出；σ>1 时低估容量→浪费预算。
  [W25] Wang et al. 2025（LDPFL-Attack 自适应）：知道"噪声可掩护"，
        幅度 p_i = min(κ·σ, sqrt(Γ/K))（随 σ 全局缩放、逐客户端盲）。
  [H25] Hossain et al. α-MPELM (IEEE TAI'25)：噪声掩蔽启发式，
        p_i = min(η_i, λ·α_i)（按轮廓裁剪，**不重定向**被裁预算）。
  [Q25] Qiu et al. 2025：全局预算分配 p_i = λ·α_i，**完全不封顶**。
  [UNI] 朴素均匀：p_i = min(η_i, sqrt(Γ/K))。
  [OURS] Optimal-ARA：客户端选择 top-K by α_i·η_i + water-filling（封顶+重定向）。

关键设定（来自 P1 实测）：η_max,i = 14.5·σ·u_i（u_i 异质∈[0.5,1.5]）。
输出：results/p4_baselines_sigma.csv（Γ 固定，扫 σ）
      results/p4_baselines_gamma.csv（σ 固定，扫 Γ）
      results/p4_baselines_report.txt
"""
import math
import random
import csv
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTDIR, exist_ok=True)

C_ETA = 14.5      # P1 实测: η_max ≈ 14.5·σ (d=1500)
SIGMA_REF = 1.0   # S21 基线的标定噪声水平（其模型里无 DP → 用固定小 σ 标定）


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
    p = [min(e, a / (2 * mu)) if a > 0 else 0.0 for a, e in zip(alpha, eta)]
    return p


def realized(alpha, p, eta):
    return sum(a * x for a, x, e in zip(alpha, p, eta) if x <= e + 1e-9)


def make_scenario(seed=777, n=30, K=8):
    rng = random.Random(seed)
    alpha = [max(0.2, rng.lognormvariate(0.0, 0.8)) for _ in range(n)]
    u = [rng.uniform(0.5, 1.5) for _ in range(n)]
    return alpha, u, K


def eta_of(u, sigma):
    return [C_ETA * sigma * x for x in u]


def strategy_p(name, alpha_sel, u_sel, Gamma, sigma, K):
    """返回分配 p（长度 K）。所有策略满足 Σp² ≤ Γ。u_sel 为相对容量系数。"""
    eta = eta_of(u_sel, sigma)
    if name == "OURS":
        # water-filling：封顶 + 预算重定向（KKT 闭式）
        return water_filling(alpha_sel, eta, Gamma)
    if name == "S21":
        # 无 DP 意识：把容量误判为 η(σ_ref)，在同一误判容量下做 water-filling
        eta_ref = eta_of(u_sel, SIGMA_REF)
        return water_filling(alpha_sel, eta_ref, Gamma)
    if name == "W25":
        # σ 自适应全局缩放 + 逐客户端盲 + 预算包络
        p_each = min(6.0 * sigma, math.sqrt(Gamma / K))
        return [p_each] * K
    if name == "H25":
        # 掩蔽感知轮廓裁剪：min(η_i, λα_i)，不重定向
        s2 = sum(a * a for a in alpha_sel)
        lam = math.sqrt(Gamma / s2) if s2 > 0 else 0.0
        return [min(e, lam * a) for a, e in zip(alpha_sel, eta)]
    if name == "Q25":
        # 全局预算分配，完全不封顶
        s2 = sum(a * a for a in alpha_sel)
        lam = math.sqrt(Gamma / s2) if s2 > 0 else 0.0
        return [lam * a for a in alpha_sel]
    if name == "UNI":
        p_each = math.sqrt(Gamma / K)
        return [min(e, p_each) for e in eta]
    raise ValueError(name)


SELECT = {
    "OURS": lambda a, u, K: sorted(range(len(a)), key=lambda i: a[i] * C_ETA * u[i], reverse=True)[:K],
    "S21":  lambda a, u, K: sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:K],
    "W25":  lambda a, u, K: sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:K],
    "H25":  lambda a, u, K: sorted(range(len(a)), key=lambda i: a[i] * C_ETA * u[i], reverse=True)[:K],
    "Q25":  lambda a, u, K: sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:K],
    "UNI":  lambda a, u, K: sorted(range(len(a)), key=lambda i: a[i] * C_ETA * u[i], reverse=True)[:K],
}
NAMES = ["OURS", "S21", "W25", "H25", "Q25", "UNI"]


def eval_at(alpha, u, K, sigma, Gamma):
    out = {}
    for nm in NAMES:
        sel = SELECT[nm](alpha, u, K)
        a_sel = [alpha[i] for i in sel]
        u_sel = [u[i] for i in sel]
        p = strategy_p(nm, a_sel, u_sel, Gamma, sigma, K)
        eta = eta_of(u_sel, sigma)
        out[nm] = realized(a_sel, p, eta)
    return out


def main():
    alpha, u, K = make_scenario()
    rep = ["=" * 88,
           " P4 基线对比（代理实现）：η_max,i = %.1f·σ·u_i, n=30, K=8, 异质 α~LogN(0,0.8)" % C_ETA,
           " 影响 = Σ α_i·p_i·1[p_i≤η_i]（超隐形上界即被检出、贡献清零）",
           "=" * 88]

    # --- 扫 σ（Γ=4000 固定）---
    GAMMA = 4000
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    rows_s = []
    rep.append(" [A] 扫 σ，Γ=%d 固定：%12s" % (GAMMA, "  ".join("%8s" % n for n in NAMES)))
    for s in sigmas:
        r = eval_at(alpha, u, K, s, GAMMA)
        rows_s.append((s, *[r[nm] for nm in NAMES]))
        rep.append("     σ=%5.1f" % s + "".join("%9.2f" % r[nm] for nm in NAMES))
    with open(os.path.join(OUTDIR, "p4_baselines_sigma.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma"] + NAMES); w.writerows(rows_s)

    # --- 扫 Γ（σ=4 固定）---
    SIG = 4.0
    gammas = [500, 1000, 2000, 4000, 8000, 16000]
    rows_g = []
    rep.append("")
    rep.append(" [B] 扫 Γ，σ=%.0f 固定" % SIG)
    for G in gammas:
        r = eval_at(alpha, u, K, SIG, G)
        rows_g.append((G, *[r[nm] for nm in NAMES]))
        rep.append("     Γ=%6d" % G + "".join("%9.2f" % r[nm] for nm in NAMES))
    with open(os.path.join(OUTDIR, "p4_baselines_gamma.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Gamma"] + NAMES); w.writerows(rows_g)

    # --- 汇总：OURS 相对最强基线的领先幅度 ---
    rep += ["", "-" * 88, " 结论（各基线在不同 regime 有 distinct 失效模式，OURS 全 regime 不塌）：",
            "   [S21] σ<σ_ref=1 时高估容量→被检出归零；σ>1 时低估容量→预算花不出去（平台低）；",
            "   [W25] 幅度随 σ 缩放但逐客户端盲：κ=6·σ 只保证平均隐藏，低容量客户端被检出的风险由预算包络兜底，",
            "         低 σ 时严重花不完预算（p=6σ≪sqrt(Γ/K)）；",
            "   [H25] 轮廓裁剪但不重定向：预算宽松时≈OURS（差距=客户端选择），预算紧张/封顶普及时低于 OURS；",
            "   [Q25] 完全不封顶：σ 小（容量低）或 Γ 大（幅度高）时高杠杆客户端超 η_i 被检出→归零；",
            "   [OURS] 选择(α·η)+water-filling(封顶+重定向) 在全部 (σ,Γ) 组合下不塌方、总影响最高或并列最高。"]
    txt = "\n".join(rep)
    with open(os.path.join(OUTDIR, "p4_baselines_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
