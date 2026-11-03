# -*- coding: utf-8 -*-
"""
R4: 在线自适应攻击者消融（针对 Limitations "attacker-knowledge assumption"
中提出的最大风险——"最优性只在 proxy/静态攻击者下验证"）
=========================================================================
在 CIFAR-10 真实管线上，对比两类攻击者（方向=-ĝ，与部署攻击一致）：

  static    : 离线一次性估计逐客户端隐形上界 η_i，water-filling 算定 p 后
              5 轮固定不变（= 论文 R2 的 OURS，即当前"最优"主张的对象）。
  adaptive  : 在线自适应——每轮根据上一轮"哪些恶意客户端被坐标中位数检出"，
              动态下调暴露者的隐形上界、向存活者重新分配整个 Γ（water-filling
              在活跃集上重解）。这是真实自适应攻击者会做的预算再分配。

两个机制结论由此被压力测试：
  (1) 容量上界 C=Ση_i² 是根本约束，自适应无法突破，只能追回静态攻击者因
      检出淘汰而浪费的预算 → 给出"适应度收益"的可量化上界。
  (2) 在 deployable 区（κ 小、检出少）static≈adaptive；在 capacity-bound 区
      （κ 大、检出多）adaptive 追回部分浪费预算，但增量有界。

运行：python r4_adaptive_ablation.py
产物：results/real/r4_adaptive_cifar10.csv, r4_adaptive_cifar10.txt
"""
import os, sys, math, csv, time, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_fl_exp as rfe

rfe.DATASET = "cifar10"            # 关键：load_data/get_model 派发用 rfe 全局量
rfe.CKPT = os.path.join(rfe.OUT, "warmup_cifar10.pt")
OUT = rfe.OUT
DATASET = "cifar10"
CKPT = rfe.CKPT

SIGMA_GRID = [0.5, 1.0, 2.0]
KAPPA_GRID = [0.25, 2.0]
MODES = ["static", "adaptive"]
SEEDS = 5
N = rfe.N_CLIENTS
K = rfe.K_MAL
MAL = list(range(N - K, N))
SEED = rfe.SEED


def build():
    ck = torch.load(CKPT, weights_only=False)
    name, shards, val, test = rfe.load_data()
    U0 = ck["U"]; alpha = ck["alpha"]; unit = ck["unit_norm"]; d = ck["d"]
    benign_std = U0.std(dim=0).mean().item()
    gvec = ck["gvec"]
    return dict(ck=ck, name=name, shards=shards, test=test, U0=U0,
                alpha=alpha, unit=unit, d=d, benign_std=benign_std, gvec=gvec)


def run_attack(mode, kappa, sigma_rel, eta_offline, seed, ctx):
    ck = ctx["ck"]; shards = ctx["shards"]; test = ctx["test"]
    unit = ctx["unit"]; benign_std = ctx["benign_std"]; gvec = ctx["gvec"]
    g = torch.Generator().manual_seed(SEED + 2000 + seed)
    sigma = sigma_rel * benign_std
    Gamma = kappa * K * unit ** 2
    v = -gvec
    gm = rfe.get_model(); rfe.load_flat(gm, ck["gmodel"])
    caps = eta_offline.clone()            # 离线估计的逐客户端隐形上界
    active = list(range(K))               # 当前未暴露的恶意客户端（MAL 下标）
    p = torch.zeros(K)
    mal_flags, round_det = 0, 0
    prev_flagged = torch.zeros(N, dtype=torch.bool)
    for r in range(rfe.ATTACK_ROUNDS):
        ups = []
        for i in range(N):
            m = rfe.local_train(gm, shards[i], rfe.ATTACK_LOCAL_EPOCHS)
            ups.append(rfe.flat(m) - rfe.flat(gm))
        U = torch.stack(ups)
        if mode == "static":
            p = rfe.water_filling(alphaM(ctx), caps, Gamma)
        else:  # adaptive
            if r == 0:
                p = rfe.water_filling(alphaM(ctx), caps, Gamma)
            else:
                caps_now = caps.clone()
                for idx, i in enumerate(MAL):
                    if prev_flagged[i]:
                        caps_now[idx] = caps_now[idx] * 0.5      # 暴露 -> 退避
                    else:
                        caps_now[idx] = min(eta_offline[idx].item(),
                                           caps_now[idx] * 1.15)  # 推高剩余余量
                if len(active) > 0:
                    a_act = ctx["alpha"][[MAL[j] for j in active]]
                    c_act = caps_now[[j for j in active]]
                    p_act = rfe.water_filling(a_act, c_act, Gamma)
                    p = torch.zeros(K)
                    for pos, j in enumerate(active):
                        p[j] = p_act[pos]
                else:
                    p = torch.zeros(K)
        for idx, i in enumerate(MAL):
            if p[idx] > 0:
                U[i] = U[i] + p[idx].item() * unit * v
        noise = torch.randn(U.shape, generator=g) * sigma
        Un = U + noise
        med0 = rfe.coord_median(Un)
        s = rfe.distance_scores(Un, med0)
        ben_idx = [i for i in range(N) if i not in MAL]
        thr = s[ben_idx].mean() + 1.645 * s[ben_idx].std()
        flagged = s > thr
        prev_flagged = flagged.clone()
        mal_flags += int(flagged[MAL].sum())
        if bool(flagged[MAL].any()):
            round_det += 1
        new_active = [j for j in active if not flagged[MAL[j]]]
        active = new_active
        keep = ~flagged
        if keep.sum() < 3:
            keep = s <= s.median()
        agg = rfe.coord_median(Un[keep])
        rfe.load_flat(gm, rfe.flat(gm) + agg)
    return (rfe.evaluate(gm, test),
            mal_flags / (K * rfe.ATTACK_ROUNDS),
            round_det / rfe.ATTACK_ROUNDS)


def alphaM(ctx):
    return ctx["alpha"][MAL].clone()


def run_none(sigma_rel, seed, ctx):
    ck = ctx["ck"]; shards = ctx["shards"]; test = ctx["test"]
    unit = ctx["unit"]; benign_std = ctx["benign_std"]
    g = torch.Generator().manual_seed(SEED + 3000 + seed)
    sigma = sigma_rel * benign_std
    gm = rfe.get_model(); rfe.load_flat(gm, ck["gmodel"])
    for r in range(rfe.ATTACK_ROUNDS):
        ups = []
        for i in range(N):
            m = rfe.local_train(gm, shards[i], rfe.ATTACK_LOCAL_EPOCHS)
            ups.append(rfe.flat(m) - rfe.flat(gm))
        U = torch.stack(ups)
        noise = torch.randn(U.shape, generator=g) * sigma
        Un = U + noise
        agg = rfe.coord_median(Un)
        rfe.load_flat(gm, rfe.flat(gm) + agg)
    return rfe.evaluate(gm, test)


def main():
    t0 = time.time()
    ctx = build()
    print("[r4] data:", ctx["name"], "unit_norm=%.3f d=%d" % (ctx["unit"], ctx["d"]))
    # 无攻击对照（每 σ 取均值）
    acc_none = {}
    for s in SIGMA_GRID:
        vals = [run_none(s, sd, ctx) for sd in range(SEEDS)]
        acc_none[s] = sum(vals) / len(vals)
        print("   NONE σ=%.2f acc=%.4f" % (s, acc_none[s]))

    rows, rep = [], ["=" * 96,
                     " R4 在线自适应攻击者消融（CIFAR-10 真实管线, K=%d, %d 轮×%d 种子）"
                     % (K, rfe.ATTACK_ROUNDS, SEEDS),
                     " 对比 static(离线固定 p) vs adaptive(在线按检出再分配 Γ)。Γ=κ·K·unit²",
                     "=" * 96]
    for s in SIGMA_GRID:
        eta_off = torch.tensor(rfe.eta_max_per_client(ctx["ck"], s))
        rep.append("-" * 96)
        rep.append(" σ_rel=%.2f  η_i=%s" % (s, ["%.2f" % e for e in eta_off.tolist()]))
        for kappa in KAPPA_GRID:
            line = "  κ=%.2f" % kappa
            agg = {}
            for mode in MODES:
                res = [run_attack(mode, kappa, s, eta_off, sd, ctx) for sd in range(SEEDS)]
                fa = sum(x[0] for x in res) / len(res)
                mfr = sum(x[1] for x in res) / len(res) * 100
                rdr = sum(x[2] for x in res) / len(res) * 100
                drop = acc_none[s] - fa
                agg[mode] = (fa, drop, mfr, rdr)
                for sd, x in enumerate(res):
                    rows.append((mode, s, kappa, sd, x[0],
                                 acc_none[s] - x[0], x[1] * 100, x[2] * 100))
                line += " | %s drop=%.4f mflg=%.1f%% rdet=%.1f%%" % (mode, drop, mfr, rdr)
            # 适应度收益 = adaptive.drop - static.drop（带符号）
            gain = agg["adaptive"][1] - agg["static"][1]
            line += "  => adaptive gain=%.4f" % gain
            rep.append(line)
        rep.append("")
    with open(os.path.join(OUT, "r4_adaptive_%s.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "sigma_rel", "kappa", "seed", "final_acc",
                    "acc_drop_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(rows)
    rep += ["", " 读法：adaptive gain=0 表示最优性对自适应鲁棒；>0 表示自适应追回了",
            "       静态攻击者因检出淘汰浪费的预算，且该增量受容量上界 C 约束（有界）。"]
    with open(os.path.join(OUT, "r4_adaptive_%s.txt" % DATASET), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("[r4] done -> r4_adaptive_%s.csv (%.1f min)" % (DATASET, (time.time() - t0) / 60))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    main()
