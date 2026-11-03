# -*- coding: utf-8 -*-
"""
P2 实证验证骨架：中位数聚合器下的顺序统计量分配（thm:t2median）
=========================================================
验证论文 Theorem thm:t2median 的经验断言：在相同的 Γ 预算与逐客户端 stealth
容量 η_max,i 下，顺序统计量分配器（OSA）比均值型水位线分配器（OURS /
water_filling）在"中位数聚合器的真实损害"上更优——因为中位数聚合器的真实
移位是一个顺序统计量，而非均值；α_i 只是均值聚合器的一阶影响权重。

复用 real_fl_exp 的全部管线与配置（数据、模型、warmup checkpoint、eta_max
探测、后门方向、coord-median 聚合、sep* 检出）。

两种度量：
  (1) 单发标量中位损害 D = median_i(<U_i, v_bd>) - m0   【主度量，直接检验定理】
      从 ck["U"] 直接计算，无需跑 FL 训练，快。
  (2) 端到端：跑完整 FL 攻击循环（coord-median + sep* 检出 + flag-and-remove），
      报告最终后门 ASR / 主任务精度 / 末轮标量中位损害。   【次度量，--end2end】

运行（CPU；建议在 r2bd 后门实验结束后跑，避免抢 CPU）：
  python r2_median_validate.py --dataset cifar10 --seeds 5 --kappas 0.5 2.0
  python r2_median_validate.py --dataset cifar10 --end2end --seeds 3

产物：results/real/median_validate_<ds>.csv（+ _seeds.csv / _report.txt）
"""
import os, csv, math, argparse, time
import torch
import torch.nn.functional as F

import real_fl_exp as R

torch.set_num_threads(8)


# ----------------------------------------------------------------------
# 顺序统计量分配器（thm:t2median 的算法实现）
# ----------------------------------------------------------------------
def osa_alloc(u_all, u_mal, eta_proj, Gamma, N, K, unit):
    """
    在投影空间里求解 thm:t2median：
      p_k^* = min{η_max,k, (t^* - u_k)_+},   t^* = max{t : B(t) ≤ Γ}
      B(t)  = Σ_{S*}(t - u_k)^2,  S* = 低于 t 且间隙最小的 r(t) 个恶意客户端
      r(t)  = max{0, m_rank - |{i: u_i ≥ t}|},  m_rank = N//2 + 1
    返回 p_mal: (K,) FL 量级（投毒时 U[i] += p_k * unit * v_bd）。
    """
    m_rank = N // 2 + 1
    # 阈值扫描：从全客户端最小投影到 最大投影+最大容量
    lo = u_all.min().item()
    hi = u_all.max().item() + eta_proj.max().item()
    t_grid = torch.linspace(lo, hi, 50000)  # 细网格：u_i 分布宽度 ~ 网格步长，粗网格会量化 t*
    best_t = u_all.median().item()          # 退化时 D=0
    for t in t_grid.tolist():
        cnt = int((u_all >= t).sum().item())
        r = max(0, m_rank - cnt)
        if r == 0:
            continue                        # 良性已满足，无需推动（D=0）
        # 可推的恶意客户端：u_k < t 且在 stealth 容量内
        cand = [(u_mal[k].item(), k) for k in range(K)
                if u_mal[k].item() < t and (t - u_mal[k].item()) <= eta_proj[k].item()]
        if len(cand) < r:
            continue
        cand.sort(reverse=True)             # 取最靠近阈值的（间隙最小）
        sel = cand[:r]
        cost = sum((t - uk) ** 2 for uk, _ in sel)
        if cost <= Gamma:
            best_t = t
    # 用 best_t 构造 p
    p = torch.zeros(K)
    cnt = int((u_all >= best_t).sum().item())
    r = max(0, m_rank - cnt)
    if r > 0:
        cand = [(u_mal[k].item(), k) for k in range(K)
                if u_mal[k].item() < best_t and (best_t - u_mal[k].item()) <= eta_proj[k].item()]
        cand.sort(reverse=True)
        sel = cand[:r]
        cost = sum((best_t - uk) ** 2 for uk, _ in sel)
        if cost <= Gamma:
            for uk, k in sel:
                p[k] = (best_t - uk) / unit     # 投影移位 / unit = FL 量级
    return p


# ----------------------------------------------------------------------
# 端到端攻击循环（与 run_attack_bd 同源，但接受外部 p，并记录末轮中位损害）
# 注意：本函数内含 local_train 反向传播，不可整体包 no_grad
# ----------------------------------------------------------------------
def run_end2end(p, sigma_rel, seed, ck, v_bd, bx, by, eta, name, shards, test):
    g = torch.Generator().manual_seed(R.SEED + 1000 + seed)
    benign_std = ck["U"].std(dim=0).mean().item()
    sigma = sigma_rel * benign_std
    gm = R.get_model(); R.load_flat(gm, ck["gmodel"])
    mal_flags, round_det = 0, 0
    u_final = None
    for r in range(R.ATTACK_ROUNDS):
        ups = []
        for i in range(R.N_CLIENTS):
            m = R.local_train(gm, shards[i], R.ATTACK_LOCAL_EPOCHS)
            ups.append(R.flat(m) - R.flat(gm))
        U = torch.stack(ups)
        if p is not None:
            for k, i in enumerate(R.mal_idx):
                U[i] = U[i] + p[k].item() * ck["unit_norm"] * v_bd
        noise = torch.randn(U.shape, generator=g) * sigma
        Un = U + noise
        med0 = R.coord_median(Un)
        s = R.distance_scores(Un, med0)
        ben_idx = [i for i in range(R.N_CLIENTS) if i not in R.mal_idx]
        thr = s[ben_idx].mean() + 1.645 * s[ben_idx].std()   # ≈5% FPR
        flagged = s > thr
        if p is not None:
            mal_flags += int(flagged[R.mal_idx].sum())
            if bool(flagged[R.mal_idx].any()):
                round_det += 1
        keep = ~flagged if p is not None else torch.ones(R.N_CLIENTS, dtype=torch.bool)
        if keep.sum() < 3:
            keep = s <= s.median()
        agg = R.coord_median(Un[keep])
        R.load_flat(gm, R.flat(gm) + agg)
        if r == R.ATTACK_ROUNDS - 1:
            # 末轮标量中位损害（投影到 v_bd）
            u_final = (U @ v_bd)
    fa = R.evaluate(gm, test)
    asr_ = R.eval_bd(gm, bx, by)
    mfr = mal_flags / (R.K_MAL * R.ATTACK_ROUNDS) * 100
    rdr = round_det / R.ATTACK_ROUNDS * 100
    m0 = (ck["U"] @ v_bd).median().item()
    D_final = u_final.median().item() - m0 if u_final is not None else float("nan")
    return fa, asr_, mfr, rdr, D_final


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["fmnist", "cifar10", "mnist"])
    ap.add_argument("--seeds", type=int, default=None, help="override R2_SEEDS")
    ap.add_argument("--sigmas", type=float, nargs="+", default=None)
    ap.add_argument("--kappas", type=float, nargs="+", default=None)
    ap.add_argument("--end2end", action="store_true", help="额外跑端到端 FL 循环（慢）")
    ap.add_argument("--e2e-seeds", type=int, default=1,
                    help="端到端每个格子的种子数（默认 1；与 r2bd 对齐用 5）")
    a = ap.parse_args()

    R.DATASET = a.dataset
    if a.seeds is not None:
        R.R2_SEEDS = a.seeds
    if a.sigmas is not None:
        R.SIGMA_GRID[:] = list(a.sigmas)
    if a.kappas is not None:
        R.KAPPA_GRID[:] = list(a.kappas)

    ck = torch.load(R.ckpt_path(), weights_only=False)
    U0 = ck["U"]; unit = ck["unit_norm"]; N = R.N_CLIENTS; K = R.K_MAL
    R.mal_idx = list(range(N - K, N))
    name, shards, val, test = R.load_data()
    gmodel0 = R.get_model(); R.load_flat(gmodel0, ck["gmodel"])
    bx, by = R.make_backdoor_set(test)
    v_bd = R.backdoor_direction(gmodel0, bx, by)     # 攻击方向（与 r2bd 一致）

    u_all = U0 @ v_bd                                # (N,) 全客户端良性投影
    m0 = u_all.median().item()
    u_mal = u_all[R.mal_idx]                         # (K,) 恶意客户端投影
    alpha_mal = ck["alpha"][R.mal_idx].clone()       # 均值型杠杆（OURS 用）

    print("[validate] ds=%s N=%d K=%d unit=%.3f m0=%.4f |v_bd|=%.3f"
          % (name, N, K, unit, m0, v_bd.norm().item()))

    # 逐 σ 预计算 η_proj（投影空间容量 = η_max * unit）
    eta_by_sigma = {s: torch.tensor(R.eta_max_per_client(ck, s)) for s in R.SIGMA_GRID}
    for s in R.SIGMA_GRID:
        print("   σ=%.2f η_i=%s" % (s, ["%.2f" % e for e in eta_by_sigma[s].tolist()]))

    rows, rep = [], []
    e2e_rows = []   # (strategy, sigma_rel, kappa, seed, final_acc, asr, mal_flag_pct, round_detect_pct, D_final)
    rep.append("=" * 96)
    rep.append(" P2 中位数聚合器顺序统计量分配验证（thm:t2median） ds=%s" % name)
    rep.append(" 单发标量中位损害 D=median_i(<U_i,v>)-m0；OURS=均值水位线，OSA=顺序统计量")
    rep.append(" 主张（与定理自洽）：相同损害下 OSA 只需 OURS 的一小部分预算；")
    rep.append("   或在等额预算 Γ 下 OSA 的中位损害不低于 OURS（OURS 过度投放会略微越阈值）。")
    rep.append("=" * 96)

    t0 = time.time()
    for s in R.SIGMA_GRID:
        eta = eta_by_sigma[s]
        eta_proj = eta * unit
        rep.append("-" * 96)
        rep.append(" σ_rel=%.2f  η_max=%s" % (s, ["%.2f" % e for e in eta.tolist()]))
        for kappa in R.KAPPA_GRID:
            Gamma = kappa * K * unit ** 2
            # ---- OURS（均值型水位线，thm:t2）----
            p_o = R.water_filling(alpha_mal, eta.clone(), Gamma)
            Uo = U0.clone()
            for k, i in enumerate(R.mal_idx):
                Uo[i] = Uo[i] + p_o[k].item() * unit * v_bd
            D_o = (Uo @ v_bd).median().item() - m0
            B_o = (p_o ** 2).sum().item()                 # OURS 实际消耗预算
            # ---- OSA（顺序统计量，thm:t2median）----
            # 预算约定与 OURS 对齐：water_filling 用 Σp²≤κK·unit²（p 为单位范数单位），
            # 投影位移=p·unit ⇒ 投影位移平方预算 = (κK·unit²)·unit² = κK·unit⁴。
            p_a = osa_alloc(u_all, u_mal, eta_proj, Gamma * unit ** 2, N, K, unit)
            Ua = U0.clone()
            for k, i in enumerate(R.mal_idx):
                Ua[i] = Ua[i] + p_a[k].item() * unit * v_bd
            D_a = (Ua @ v_bd).median().item() - m0
            B_a = (p_a ** 2).sum().item()                 # OSA 实际消耗预算
            # ---- 等额预算对照：给 OSA 与 OURS 相同的预算 B_o（同为 p² 单位）----
            p_a2 = osa_alloc(u_all, u_mal, eta_proj, B_o * unit ** 2, N, K, unit)
            Ua2 = U0.clone()
            for k, i in enumerate(R.mal_idx):
                Ua2[i] = Ua2[i] + p_a2[k].item() * unit * v_bd
            D_a2 = (Ua2 @ v_bd).median().item() - m0
            eff = (B_o / B_a) if B_a > 1e-9 else float("inf")   # 同损害下的预算效率倍数
            rows.append(("OURS", s, kappa, D_o, B_o, float("nan")))
            rows.append(("OSA", s, kappa, D_a, B_a, eff))
            rep.append("  κ=%.2f Γ=%.3f" % (kappa, Gamma))
            rep.append("    OURS: D=%.4f 消耗预算 B=%.3f" % (D_o, B_o))
            rep.append("    OSA : D=%.4f 消耗预算 B=%.3f  → 同损害预算效率 ×%.2f" % (D_a, B_a, eff))
            rep.append("    OSA@同B: D=%.4f  (等额预算下 OSA≥OURS: %s)"
                       % (D_a2, "YES" if D_a2 >= D_o - 1e-9 else "no(略低:过量投放越阈值)"))
            # ---- 可选端到端 ----
            if a.end2end:
                for strat, p in (("OURS", p_o), ("OSA", p_a)):
                    res = [run_end2end(p, s, sd, ck, v_bd, bx, by, eta, name, shards, test)
                           for sd in range(a.e2e_seeds)]
                    for sd, (fa, asr_, mfr, rdr, Df) in enumerate(res):
                        e2e_rows.append((strat, s, kappa, sd, fa, asr_, mfr, rdr, Df))
                    n = len(res)
                    fa_m = sum(x[0] for x in res) / n
                    asr_m = sum(x[1] for x in res) / n
                    asr_sd = (sum((x[1] - asr_m) ** 2 for x in res) / max(n - 1, 1)) ** 0.5
                    mfr_m = sum(x[2] for x in res) / n
                    rdr_m = sum(x[3] for x in res) / n
                    Df_m = sum(x[4] for x in res) / n
                    rep.append("     [%s e2e] seeds=%d acc=%.4f ASR=%.4f±%.4f malflg%%=%.1f rdet%%=%.1f D_final=%.4f"
                               % (strat, n, fa_m, asr_m, asr_sd, mfr_m, rdr_m, Df_m))
    rep.append("")
    rep.append(" 用时 %.1fs；预算效率 × 越大，thm:t2median 的『顺序统计量分配是最优的』经验支撑越强。"
               % (time.time() - t0))
    rep.append(" 注：eta_max 沿 -gvec 探测（与 r2bd 同约定），沿 v_bd 投毒；容量量级近似方向无关。")

    with open(os.path.join(R.OUT, "median_validate_%s.csv" % a.dataset), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "sigma_rel", "kappa", "D_scalar", "budget_used", "eff_gain_vs_ours"])
        w.writerows(rows)
    if e2e_rows:
        with open(os.path.join(R.OUT, "median_validate_e2e_%s_seeds.csv" % a.dataset), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strategy", "sigma_rel", "kappa", "seed", "final_acc",
                        "asr", "mal_flag_pct", "round_detect_pct", "D_final"])
            w.writerows(e2e_rows)
        print("[validate] e2e seeds -> results/real/median_validate_e2e_%s_seeds.csv" % a.dataset)
    with open(os.path.join(R.OUT, "median_validate_report_%s.txt" % a.dataset), "w",
              encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("\n".join(rep))
    print("[validate] done -> results/real/median_validate_%s.csv" % a.dataset)


if __name__ == "__main__":
    main()
