# -*- coding: utf-8 -*-
"""
T4 Stackelberg 系数真实标定（CIFAR-10 管线）
=============================================
原 T4 的 U(σ)=a·lnσ−b·σ²−d·σ 用合成系数 (a=2.5,b=0.03,d=1.16)。本脚本：
A 部分（新实验）：无攻击、σ_rel∈{0,0.5,1,2,4,8}×5 种子，测干净聚合噪声的
    效用损失 L(σ)=acc(0)−acc(σ)，拟合 L≈b·σ²（判别二次 vs 线性）。
B 部分（组装）：D(σ)=best-attacker 损伤包络（从 r2_main_cifar10.csv 提取，
    地板段线性化 D≈d0+d1·σ）；U(σ)=a·lnσ−b·σ²−D(σ)；a 为隐私交换率
    （不可测的制度偏好），给 σ*(a) 曲线与过隐私化临界 a*。
    重绘 fig5：实测分量 + σ*(a)。

运行：python t4_calibrate.py --part A|B|AB
产物：results/real/t4_utility_cifar10.csv, t4_report_cifar10.txt,
      paper/figures/fig5_stackelberg.png（覆盖重绘）
"""
import os, sys, csv, math, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_fl_exp as rfe

OUT = rfe.OUT
DATASET = "cifar10"
CKPT = os.path.join(OUT, "warmup_cifar10.pt")
SIGMA_UTILITY = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = 5


# ---------------- A 部分：无攻击效用损失 ----------------
def part_A():
    ck = torch.load(CKPT, weights_only=False)
    U0 = ck["U"]; unit = ck["unit_norm"]
    benign_std = U0.std(dim=0).mean().item()
    name, shards, val, test = rfe.load_data()
    rows = []
    for s in SIGMA_UTILITY:
        sigma = s * benign_std
        accs = []
        for sd in range(SEEDS):
            g = torch.Generator().manual_seed(rfe.SEED + 1000 + sd)
            gm = rfe.get_model(); rfe.load_flat(gm, ck["gmodel"])
            for r in range(rfe.ATTACK_ROUNDS):
                ups = []
                for i in range(rfe.N_CLIENTS):
                    m = rfe.local_train(gm, shards[i], rfe.ATTACK_LOCAL_EPOCHS)
                    ups.append(rfe.flat(m) - rfe.flat(gm))
                U = torch.stack(ups)
                noise = torch.randn(U.shape, generator=g) * sigma
                agg = rfe.coord_median(U + noise)
                rfe.load_flat(gm, rfe.flat(gm) + agg)
            accs.append(rfe.evaluate(gm, test))
        mean_acc = sum(accs) / len(accs)
        rows.append((s, mean_acc, accs))
        print("[t4-A] σ_rel=%.1f  acc=%.4f  (seeds: %s)" %
              (s, mean_acc, ["%.4f" % a for a in accs]))
    acc0 = next(r[1] for r in rows if r[0] == 0.0)
    with open(os.path.join(OUT, "t4_utility_%s.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_rel", "mean_acc", "utility_loss_vs_sigma0"])
        for s, acc, _ in rows:
            w.writerow([s, acc, acc0 - acc])
    rep = ["T4-A 无攻击效用损失（CIFAR-10, %d 轮窗口, %d 种子）" % (rfe.ATTACK_ROUNDS, SEEDS),
           "acc(σ=0)=%.4f" % acc0] + \
          ["  σ_rel=%.1f  L=%.4f" % (s, acc0 - acc) for s, acc, _ in rows]
    with open(os.path.join(OUT, "t4_report_%s.txt" % DATASET), "a", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return rows


# ---------------- B 部分：组装 U(σ)、解 σ*(a)、重绘 fig5 ----------------
def part_B():
    import numpy as np
    # 1) 效用损失拟合：混合 L = b1·σ + b2·σ²（主口径），纯二次作稳健性
    urows = list(csv.DictReader(open(os.path.join(OUT, "t4_utility_%s.csv" % DATASET))))
    sig = np.array([float(r["sigma_rel"]) for r in urows if float(r["sigma_rel"]) > 0])
    loss = np.array([float(r["utility_loss_vs_sigma0"]) for r in urows if float(r["sigma_rel"]) > 0])
    A = np.vstack([sig, sig ** 2]).T
    (b1, b2), *_ = np.linalg.lstsq(A, loss, rcond=None)
    pred = b1 * sig + b2 * sig ** 2
    r2m = 1 - ((loss - pred) ** 2).sum() / ((loss - loss.mean()) ** 2).sum()
    bq = float((sig ** 2 * loss).sum() / (sig ** 4).sum())
    r2q = 1 - ((loss - bq * sig ** 2) ** 2).sum() / ((loss - loss.mean()) ** 2).sum()
    print("[t4-B] 混合 b1=%.5f b2=%.5f (R²=%.4f)  纯二次 b=%.5f (R²=%.3f)" %
          (b1, b2, r2m, bq, r2q))
    # 2) 损伤包络 D(σ)（r2_main_cifar10 best-attacker，地板段线性化）
    r2 = list(csv.DictReader(open(os.path.join(OUT, "r2_main_cifar10.csv"))))
    env = {}
    for r in r2:
        s = float(r["sigma_rel"]); d = float(r["acc_drop_vs_none"])
        env[s] = max(env.get(s, -9), d)
    ss = np.array(sorted(env)); dd = np.array([env[s] for s in ss])
    A2 = np.vstack([np.ones_like(ss), ss]).T
    (d0, d1), *_ = np.linalg.lstsq(A2, dd, rcond=None)
    print("[t4-B] D(σ)=%.4f+%.4f·σ (points: %s)" %
          (d0, d1, ["%.1f:%.4f" % (s, v) for s, v in zip(ss, dd)]))
    # 3) σ*(a)：a/σ = (b1 + 2b2σ) + d1 → 2b2σ² + (b1+d1)σ − a = 0
    def sigma_star(a):
        disc = (b1 + d1) ** 2 + 8 * b2 * a
        return (-(b1 + d1) + math.sqrt(disc)) / (4 * b2)
    a_grid = np.linspace(0.005, 0.12, 60)
    stars = [sigma_star(a) for a in a_grid]
    a_19 = (b1 + d1) * 1.9 + 2 * b2 * 1.9 ** 2      # σ*=1.9 对应的 a
    s_star = sigma_star(a_19)
    # 边际分解 @σ*=1.9：攻击面扩大 vs 直接效用损失
    marg_noise = b1 + 2 * b2 * 1.9
    share_dmg = d1 / (marg_noise + d1)
    rep = ["T4-B Stackelberg 真实标定（CIFAR-10）",
           "  效用损失（混合）：L=%.5f·σ+%.5f·σ² (R²=%.4f)；纯二次 b=%.5f (R²=%.3f)" % (b1, b2, r2m, bq, r2q),
           "  损伤包络：D(σ)=%.4f+%.4f·σ（r2 best-attacker, 地板段）" % (d0, d1),
           "  σ*(a)：2b2σ²+(b1+d1)σ−a=0；σ*=1.9 ⟺ a=%.4f（回代 σ*=%.2f）" % (a_19, s_star),
           "  边际分解 @σ*=1.9：攻击面扩大 d1=%.4f 占 %.0f%%，直接效用损失占 %.0f%%" %
           (d1, 100 * share_dmg, 100 * (1 - share_dmg)),
           "  σ*(a) 曲线：" + "  ".join("a=%.3f:σ*=%.2f" % (a, s) for a, s in zip(a_grid[::8], stars[::8]))]
    print("\n".join(rep))
    with open(os.path.join(OUT, "t4_report_%s.txt" % DATASET), "a", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    np.savez(os.path.join(OUT, "t4_params_%s.npz" % DATASET),
             b1=b1, b2=b2, bq=bq, r2m=r2m, r2q=r2q, d0=d0, d1=d1,
             a_19=a_19, share_dmg=share_dmg,
             a_grid=a_grid, stars=stars, sig=sig, loss=loss, env_s=ss, env_d=dd)
    # 4) 重绘 fig5
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axes[0]
    xs = np.linspace(0.05, 8, 200)
    U = a_19 * np.log(xs) - (b1 * xs + b2 * xs ** 2) - (d0 + d1 * xs)
    ax.plot(xs, U, "-", color="#1f6feb", lw=1.5)
    ax.axvline(1.9, color="#cf222e", ls="--", lw=1.0)
    ax.annotate(r"$\sigma^\star=1.9$", (1.9, U.max()),
                textcoords="offset points", xytext=(6, -2), fontsize=8, color="#cf222e")
    # 分量分解（堆叠展示代价结构）
    ax.plot(xs, a_19 * np.log(xs), ":", color="#8250df", lw=1.0, label=r"privacy benefit $a\ln\sigma$")
    ax.plot(xs, -(b1 * xs + b2 * xs ** 2), "--", color="#d29922", lw=1.0, label=r"utility loss (measured)")
    ax.plot(xs, -(d0 + d1 * xs), "-.", color="#cf222e", lw=1.0, label=r"breach envelope $D(\sigma)$ (measured)")
    ax.set_xlabel(r"$\sigma_{\mathrm{rel}}$"); ax.set_ylabel(r"$U(\sigma)$ (acc-equiv.)")
    ax.set_title(r"A  Calibrated $U(\sigma)$, $a=%.3f$" % a_19, fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    ax.plot(a_grid, stars, "-", color="#1f6feb", lw=1.5)
    ax.axhline(14.2, color="gray", ls=":", lw=0.8)
    ax.annotate(r"$\sigma_c=14.2$ (floor $>$ noise)", (0.012, 14.2),
                textcoords="offset points", xytext=(0, 4), fontsize=7, color="gray")
    ax.set_xlabel(r"privacy exchange rate $a$ (acc-equiv./ln-unit)")
    ax.set_ylabel(r"$\sigma^\star(a)$")
    ax.set_title("B  Optimum vs. privacy valuation", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "paper", "figures", "fig5_stackelberg.png")
    fig.savefig(fn, dpi=200)
    print("saved", fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="AB", choices=["A", "B", "AB"])
    a = ap.parse_args()
    rfe.DATASET = "cifar10"
    if a.part in ("A", "AB"):
        part_A()
    if a.part in ("B", "AB"):
        part_B()
