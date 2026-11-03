# -*- coding: utf-8 -*-
"""
R3: 聚合器替换验证（Limitations 中 "Krum/Trimmed-Mean variants change c
but not the two-regime shape" 的实验落地）
=================================================================
A 部分（廉价，无本地训练）：每个聚合器在 σ_rel∈{0.5..16} 下探测逐客户端
    η_max(σ)，按两段式模型 η_f·sqrt(1+(σ/σ_c)^2) 拟合，检验：
    水平（c/η_f）随聚合器变、形状（两段式+交叉点结构）不变。
B 部分（训练，后台）：OURS/Q25/W25 × 3 聚合器 × σ{0.5,1,2} × κ{0.25,2}
    × 5 种子，检验前沿归属与封顶生效模式（κ 大→OURS 优）跨聚合器不变。

聚合器与配套检测统计：
  median  : 坐标中位数 + 到中位数距离（与 R2 一致）
  trimmed : 逐坐标 β=0.2 截尾均值 + 到截尾均值距离
  krum    : Krum 选择（f=6）+ Krum 分数（到 n-f-2 最近邻距离和）

运行：python r3_aggregator_swap.py --part A|B|AB --dataset cifar10
产物：results/real/r3_eta_{ds}.csv, r3_main_{ds}.csv, r3_report_{ds}.txt
"""
import os, sys, math, csv, time, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_fl_exp as rfe

OUT = rfe.OUT
DATASET = "cifar10"
CKPT = os.path.join(OUT, "warmup_cifar10.pt")

SIGMA_PROBE = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]     # A 部分
SIGMA_GRID = [0.5, 1.0, 2.0]                      # B 部分
KAPPA_GRID = [0.25, 2.0]
STRATS = ["OURS", "Q25", "W25"]
AGGS = ["median", "trimmed", "krum"]
SEEDS = 5
TRIM_BETA = 0.2
K_MAL = rfe.K_MAL
N_CLIENTS = rfe.N_CLIENTS


# ---------------- 聚合器 ----------------
def trimmed_mean(U, beta=TRIM_BETA):
    n = U.shape[0]
    k = int(math.floor(beta * n))
    if n - 2 * k < 1:
        k = (n - 1) // 2
    s, _ = torch.sort(U, dim=0)
    return s[k:n - k].mean(dim=0)


def krum_scores(U, f):
    n = U.shape[0]
    d2 = torch.cdist(U, U) ** 2
    m = max(1, n - f - 2)
    out = torch.empty(n)
    for i in range(n):
        row = d2[i].clone()
        row[i] = float("inf")
        out[i] = torch.topk(row, m, largest=False).values.sum()
    return out


def flame_aggregate(U, f):
    """FLAME (Allouah et al., ICML 2023)：迭代余弦过滤 + 接受集均值 + 自适应噪声。
    每轮至多移除 f 个负相似客户端；噪声 std = ||median||/(f·√d)（论文默认设定，
    f 已知）。防御侧噪声用全局 torch 流（执行顺序固定，可复现）。"""
    n, d = U.shape
    m = rfe.coord_median(U)
    S = list(range(n))
    while True:
        m = rfe.coord_median(U[S])
        norms = U[S].norm(dim=1)
        sims = (U[S] @ m) / (norms * m.norm() + 1e-12)
        bad = [S[j] for j in range(len(S)) if sims[j] < 0]
        if not bad or len(S) <= max(2, n - f):
            break
        out = max(bad, key=lambda i: U[i].norm().item())
        S.remove(out)
    agg = U[S].mean(dim=0)
    delta_m = m.norm().item() / (f * math.sqrt(d))
    return agg + torch.randn(d) * delta_m


def aggregate(U, agg):
    if agg == "median":
        return rfe.coord_median(U)
    if agg == "trimmed":
        return trimmed_mean(U)
    if agg == "krum":
        f = max(0, min(K_MAL, U.shape[0] - 3))
        return U[int(krum_scores(U, f).argmin().item())]
    if agg == "flame":
        f = max(0, min(K_MAL, U.shape[0] - 3))
        return flame_aggregate(U, f)
    raise ValueError(agg)


def detect_scores(U, agg):
    """与聚合器配套的检测统计量。"""
    if agg == "median":
        return rfe.distance_scores(U, rfe.coord_median(U))
    if agg == "trimmed":
        return rfe.distance_scores(U, trimmed_mean(U))
    if agg == "krum":
        f = max(0, min(K_MAL, U.shape[0] - 3))
        return krum_scores(U, f)
    if agg == "flame":
        m = rfe.coord_median(U)
        return 1.0 - (U @ m) / (U.norm(dim=1) * m.norm() + 1e-12)
    raise ValueError(agg)


# ---------------- A 部分：η_max 探测 ----------------
def eta_probe(ck, sigma_rel, agg, trials=6):
    """单客户端 flag-prob 穿越 0.5 的 ρ（二分），方向 -ĝ，逐恶意客户端。"""
    U0 = ck["U"]; gv = ck["gvec"]; un = ck["unit_norm"]
    benign_std = U0.std(dim=0).mean().item()
    sigma = sigma_rel * benign_std
    g = torch.Generator().manual_seed(rfe.SEED + 99)
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    v = -gv

    def flag_prob(i, rho):
        cnt = 0
        for _ in range(trials):
            noise = torch.randn(U0.shape, generator=g) * sigma
            U = U0 + noise
            U[i] = U[i] + rho * un * v
            s = detect_scores(U, agg)
            ben = torch.cat([s[:i], s[i + 1:]])
            thr = ben.mean() + 1.645 * ben.std()
            cnt += 1 if s[i] > thr else 0
        return cnt / trials

    etas = []
    for i in mal:
        lo, hi = 0.05, 12.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if flag_prob(i, mid) < 0.5:
                lo = mid
            else:
                hi = mid
        etas.append(0.5 * (lo + hi))
    return etas


def fit_tworegime(sigmas, etas):
    """η = η_f·sqrt(1+(σ/σ_c)²) 的网格+精修拟合；返回 (eta_f, sigma_c, R2)。"""
    import numpy as np
    s = np.array(sigmas); y = np.array(etas)
    best = None
    for sc in np.linspace(0.5, 40, 80):
        for ef in np.linspace(max(y.min() * 0.5, 0.1), y.max() * 1.5, 80):
            pred = ef * np.sqrt(1 + (s / sc) ** 2)
            r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if best is None or r2 > best[0]:
                best = (r2, ef, sc)
    return best[1], best[2], best[0]


def out_suffix():
    """限定 --aggs 时输出到独立文件，避免覆盖 verify 依赖的全量 CSV。"""
    return "" if AGGS == ["median", "trimmed", "krum"] else "_" + "_".join(AGGS)


def part_A(ck):
    rows, rep = [], ["=" * 90, " R3-A：聚合器替换下的 η_max(σ) 两段式形状检验（无训练，探测）", "=" * 90]
    fits = {}
    for agg in AGGS:
        rep.append("-" * 90)
        rep.append(" aggregator = %s" % agg)
        etas_by_sigma = {}
        for s in SIGMA_PROBE:
            etas = eta_probe(ck, s, agg)
            etas_by_sigma[s] = etas
            C = sum(e * e for e in etas)
            rep.append("   σ_rel=%5.1f  η_i=%s  C=Ση²=%.2f" %
                       (s, ["%.2f" % e for e in etas], C))
            for e in etas:
                rows.append((agg, s, e))
        ef, sc, r2 = fit_tworegime(SIGMA_PROBE,
                                   [float(torch.tensor(etas_by_sigma[s]).median()) for s in SIGMA_PROBE])
        fits[agg] = (ef, sc, r2)
        rep.append("   两段式拟合（η 取中位数）：η_f=%.2f  σ_c=%.2f  R²=%.3f" % (ef, sc, r2))
    rep.append("")
    rep.append(" 判读：形状不变 = 各聚合器均为『地板 + 亚线性恢复』两段式（R² 高、η_f 同量级、")
    rep.append("       σ_c 落在探测区间内）；c 变 = η_f/η_i 水平随聚合器不同。")
    fn = os.path.join(OUT, "r3_eta_%s%s.csv" % (DATASET, out_suffix()))
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aggregator", "sigma_rel", "eta_max"])
        w.writerows(rows)
    with open(os.path.join(OUT, "r3_report_%s.txt" % DATASET), "a", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return fits


# ---------------- B 部分：攻击网格 ----------------
def part_B(ck, fits_report):
    U0 = ck["U"]; alpha = ck["alpha"]; unit = ck["unit_norm"]; d = ck["d"]
    benign_std = U0.std(dim=0).mean().item()
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    name, shards, val, test = rfe.load_data()
    gmodel0 = rfe.get_model(); rfe.load_flat(gmodel0, ck["gmodel"])
    acc_start = rfe.evaluate(gmodel0, test)
    print("[r3-B] dataset=%s 起始 acc=%.4f" % (name, acc_start))

    eta_by_agg_sigma = {(a, s): torch.tensor(eta_probe(ck, s, a))
                        for a in AGGS for s in SIGMA_GRID}

    def run_attack(agg, strat, kappa, sigma_rel, eta, seed):
        g = torch.Generator().manual_seed(rfe.SEED + 1000 + seed)
        sigma = sigma_rel * benign_std
        Gamma = kappa * K_MAL * unit ** 2
        p = rfe.alloc(strat, alpha[mal].clone(), eta.clone(), Gamma, K_MAL, sigma_rel) \
            if strat != "NONE" else None
        gm = rfe.get_model(); rfe.load_flat(gm, ck["gmodel"])
        mal_flags, round_det = 0, 0
        for r in range(rfe.ATTACK_ROUNDS):
            ups = []
            for i in range(N_CLIENTS):
                m = rfe.local_train(gm, shards[i], rfe.ATTACK_LOCAL_EPOCHS)
                ups.append(rfe.flat(m) - rfe.flat(gm))
            U = torch.stack(ups)
            if p is not None:
                v = -ck["gvec"]
                for k, i in enumerate(mal):
                    U[i] = U[i] + p[k].item() * unit * v
            noise = torch.randn(U.shape, generator=g) * sigma
            Un = U + noise
            s = detect_scores(Un, agg)
            ben_idx = [i for i in range(N_CLIENTS) if i not in mal]
            thr = s[ben_idx].mean() + 1.645 * s[ben_idx].std()
            flagged = s > thr
            if p is not None:
                mal_flags += int(flagged[mal].sum())
                if bool(flagged[mal].any()):
                    round_det += 1
            keep = ~flagged if p is not None else torch.ones(N_CLIENTS, dtype=torch.bool)
            if keep.sum() < 4:
                keep = s <= s.median()
            aggv = aggregate(Un[keep], agg)
            rfe.load_flat(gm, rfe.flat(gm) + aggv)
        return rfe.evaluate(gm, test), mal_flags / (K_MAL * rfe.ATTACK_ROUNDS), \
            round_det / rfe.ATTACK_ROUNDS

    rows, seed_rows, rep = [], [], ["=" * 96,
                     " R3-B：聚合器替换攻击网格 dataset=%s K=%d %d轮×%d种子" %
                     (name, K_MAL, rfe.ATTACK_ROUNDS, SEEDS),
                     " 起始 acc=%.4f。σ_rel∈%s κ∈%s 策略=%s 聚合器=%s" %
                     (acc_start, SIGMA_GRID, KAPPA_GRID, STRATS, AGGS),
                     "=" * 96]
    t0 = time.time()
    for agg in AGGS:
        acc_none_list = [run_attack(agg, "NONE", 0.0, 1.0, eta_by_agg_sigma[(agg, 1.0)], sd)[0]
                         for sd in range(SEEDS)]
        acc_none = sum(acc_none_list) / len(acc_none_list)
        rep.append("-" * 96)
        rep.append(" aggregator=%s  NONE 对照 final_acc=%.4f" % (agg, acc_none))
        for s in SIGMA_GRID:
            eta = eta_by_agg_sigma[(agg, s)]
            C = float((eta ** 2).sum())
            for kappa in KAPPA_GRID:
                Gam = kappa * K_MAL * unit ** 2
                rep.append(" σ_rel=%.1f κ=%.2f  C/Γ=%.2f  η_i=%s" %
                           (s, kappa, C / Gam, ["%.2f" % e for e in eta.tolist()]))
                for strat in STRATS:
                    res = [run_attack(agg, strat, kappa, s, eta, sd) for sd in range(SEEDS)]
                    fa = sum(x[0] for x in res) / len(res)
                    mfr = sum(x[1] for x in res) / len(res) * 100
                    rdr = sum(x[2] for x in res) / len(res) * 100
                    drop = acc_none - fa
                    rows.append((agg, strat, s, kappa, fa, drop, mfr, rdr))
                    for sd in range(SEEDS):
                        seed_rows.append((agg, strat, s, kappa, sd, res[sd][0],
                                          acc_none_list[sd] - res[sd][0],
                                          res[sd][1] * 100, res[sd][2] * 100))
                        rep.append("   %5s drop=%+.4f malflg=%5.1f%% rdet=%5.1f%%" %
                                   (strat, drop, mfr, rdr))
            print("[r3-B] %s σ=%.1f done (%.1f min elapsed)" %
                  (agg, s, (time.time() - t0) / 60))
        rep.append("")
    # 汇总：每个 (agg,σ,κ) 下 OURS vs Q25 排序与封顶生效判据
    rep.append(" 汇总：OURS−Q25 损伤差（正=OURS 优）")
    for agg in AGGS:
        diffs = []
        for s in SIGMA_GRID:
            for kappa in KAPPA_GRID:
                d_ours = next(r[5] for r in rows if r[0] == agg and r[1] == "OURS"
                              and r[2] == s and r[3] == kappa)
                d_q25 = next(r[5] for r in rows if r[0] == agg and r[1] == "Q25"
                             and r[2] == s and r[3] == kappa)
                diffs.append((s, kappa, d_ours - d_q25))
        rep.append("   %s: %s" % (agg, ["σ%.1f/κ%.2f:%+.4f" % t for t in diffs]))
    with open(os.path.join(OUT, "r3_main_%s%s.csv" % (DATASET, out_suffix())), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aggregator", "strategy", "sigma_rel", "kappa",
                    "final_acc", "acc_drop_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(rows)
    with open(os.path.join(OUT, "r3_main_%s%s_seeds.csv" % (DATASET, out_suffix())), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aggregator", "strategy", "sigma_rel", "kappa", "seed",
                    "final_acc", "acc_drop_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(seed_rows)
    with open(os.path.join(OUT, "r3_report_%s.txt" % DATASET), "a", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("[r3-B] done -> r3_main_%s.csv (%.1f min)" % (DATASET, (time.time() - t0) / 60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--aggs", default=None,
                    help="逗号分隔，限定聚合器（如 flame）；限定时输出到带后缀的独立文件")
    a = ap.parse_args()
    DATASET = a.dataset
    if a.aggs:
        AGGS = [x.strip() for x in a.aggs.split(",")]
    rfe.DATASET = a.dataset          # 关键：load_data/get_model 派发用 rfe 的全局量
    CKPT = os.path.join(OUT, "warmup_%s.pt" % DATASET)
    ck = torch.load(CKPT, weights_only=False)
    torch.manual_seed(rfe.SEED)
    if a.part in ("A", "AB"):
        part_A(ck)
    if a.part in ("B", "AB"):
        part_B(ck, None)
