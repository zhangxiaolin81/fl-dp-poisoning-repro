# -*- coding: utf-8 -*-
"""
fig13_pareto_bd.py — R2b-ASR 统一端到端比较：表 (tab:r2bde2e) + Pareto 图 (fig:pareto)。

回应模拟审稿 Major 2/3：把 OSA 与 OURS/W25/Q25 放进同一张端到端表
（ΔASR、检测率、部署预算、主任务精度），并画固定预算下"检测率 vs ΔASR"
Pareto 平面。全部数字由本脚本从原始 CSV 计算，可追溯。

数据源：
  results/real/r2_main_bd_cifar10.csv            — 6 allocator × 3σ × 2κ × 5 seeds 均值
  results/real/median_validate_e2e_cifar10_seeds.csv — OURS/OSA × 3σ × 2κ × 5 seeds 逐种子
  results/real/r2_report_bd_cifar10.txt          — 每 σ 的 η_i（预算计算用）
  results/real/warmup_cifar10.pt                 — alpha（OURS 预算计算用）

产出：
  results/real/r2bde2e_unified.csv               — 统一表数据
  stdout: LaTeX 表体 + 口径检查（OURS 两源交叉验证）
  paper/figures/fig13_pareto_bd.pdf/.png
"""
import os, csv, math, re
import torch

import figstyle
figstyle.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "real")
FIGDIR = os.path.join(os.path.dirname(HERE), "paper", "figures")

K = 6
ASR_NONE = 0.0866          # r2bd 报告 NONE 基线（CSV 中 asr_gain_vs_none 与之一致）
ACC_NONE = 0.6149          # r2bd 报告 NONE final_acc
SIGMAS = [0.5, 1.0, 2.0]
KAPPAS = [0.5, 2.0]

# ----------------------------------------------------------------------
# 1. 读 r2bd 主 CSV（6 allocator）
# ----------------------------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(RES, "r2_main_bd_cifar10.csv"))))
bd = {}   # (strat, sigma, kappa) -> dict
for r in rows:
    key = (r["strategy"], float(r["sigma_rel"]), float(r["kappa"]))
    bd[key] = dict(asr=float(r["asr"]), gain=float(r["asr_gain_vs_none"]),
                   acc=float(r["final_acc"]), rdr=float(r["round_detect_pct"]),
                   mfr=float(r["mal_flag_pct"]))

# ----------------------------------------------------------------------
# 2. 读 OSA/OURS 5-seed e2e CSV
# ----------------------------------------------------------------------
e2e_p = os.path.join(RES, "median_validate_e2e_cifar10_seeds.csv")
osa = {}   # (strat, sigma, kappa) -> list of seed dicts
if os.path.exists(e2e_p):
    for r in csv.DictReader(open(e2e_p)):
        key = (r["strategy"], float(r["sigma_rel"]), float(r["kappa"]))
        osa.setdefault(key, []).append(dict(
            asr=float(r["asr"]), acc=float(r["final_acc"]),
            rdr=float(r["round_detect_pct"]), mfr=float(r["mal_flag_pct"]),
            D=float(r["D_final"])))
else:
    raise SystemExit("缺少 %s —— 先跑 r2_median_validate.py --end2end --e2e-seeds 5" % e2e_p)

def mean(xs):
    return sum(xs) / len(xs)

def sd(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5

# 口径检查：OURS 在两源里应一致（同协议、5 seeds）
print("== 口径检查：OURS 两源对比（ΔASR / rdet%） ==")
for s in SIGMAS:
    for k in KAPPAS:
        a = bd[("OURS", s, k)]
        seeds = osa[("OURS", s, k)]
        g2 = mean([x["asr"] for x in seeds]) - ASR_NONE
        r2 = mean([x["rdr"] for x in seeds])
        print("  σ=%.1f κ=%.1f  r2bd: %+.4f/%4.0f   validate: %+.4f/%4.0f"
              % (s, k, a["gain"], a["rdr"], g2, r2))

# ----------------------------------------------------------------------
# 3. 部署预算 ‖p‖²/Γ（确定性：alpha/eta 来自 warm-up ck 与探测）
# ----------------------------------------------------------------------
# η_i per σ：从 r2bd 报告解析
eta_by_sigma = {}
pat = re.compile(r"σ_rel=\s*([\d.]+)\s+η_i=\[([^\]]+)\]")
txt = open(os.path.join(RES, "r2_report_bd_cifar10.txt"), encoding="utf-8").read()
for m in pat.finditer(txt):
    s = float(m.group(1))
    eta_by_sigma[s] = torch.tensor([float(x.strip().strip("'\"")) for x in m.group(2).split(",")])
assert set(eta_by_sigma) == set(SIGMAS), eta_by_sigma.keys()

ck = torch.load(os.path.join(RES, "warmup_cifar10.pt"), weights_only=False)
alpha_mal = ck["alpha"][list(range(ck["alpha"].shape[0] - K, ck["alpha"].shape[0]))].clone()
unit = float(ck["unit_norm"])

def water_filling_budget(alpha, eta, Gamma):
    lo, hi = 1e-9, max(alpha).item() / 1e-9 + 1.0
    p = None
    for _ in range(80):
        mu = 0.5 * (lo + hi)
        p = torch.clamp(alpha / (2 * mu), max=eta)
        if (p ** 2).sum().item() > Gamma:
            lo = mu
        else:
            hi = mu
    mu = 0.5 * (lo + hi)
    p = torch.clamp(alpha / (2 * mu), max=eta)
    return (p ** 2).sum().item()

# OSA 预算：median_validate 主 CSV（确定性）
osa_budget = {}
for r in csv.DictReader(open(os.path.join(RES, "median_validate_cifar10.csv"))):
    if r["strategy"] == "OSA":
        osa_budget[(float(r["sigma_rel"]), float(r["kappa"]))] = float(r["budget_used"])

def budget_frac(strat, s, k):
    Gamma = k * K * unit ** 2
    if strat == "Q25":
        return 1.0                       # 无封顶，lam 缩放到恰好用满 Γ
    if strat == "W25":
        eta_med = eta_by_sigma[s].median().item()
        p_each = min(max(s, 0.5) * eta_med, math.sqrt(Gamma / K))
        return K * p_each ** 2 / Gamma
    if strat == "OURS":
        return water_filling_budget(alpha_mal, eta_by_sigma[s].clone(), Gamma) / Gamma
    if strat == "OSA":
        return osa_budget[(s, k)] / Gamma
    raise ValueError(strat)

# ----------------------------------------------------------------------
# 4. 统一表数据 + LaTeX
# ----------------------------------------------------------------------
ORDER = ["W25", "Q25", "OURS", "OSA"]
out_rows = {}
print("\n== 统一端到端表（ΔASR / rdet% / ‖p‖²/Γ，均值 over σ∈{0.5,1,2} ± 范围） ==")
for strat in ORDER:
    for k in KAPPAS:
        cells = []
        for s in SIGMAS:
            if strat == "OSA":
                seeds = osa[("OSA", s, k)]
                g = mean([x["asr"] for x in seeds]) - ASR_NONE
                gsd = sd([x["asr"] for x in seeds])
                rdr = mean([x["rdr"] for x in seeds])
                acc = mean([x["acc"] for x in seeds])
                n = len(seeds)
            else:
                a = bd[(strat, s, k)]
                g, gsd, rdr, acc, n = a["gain"], float("nan"), a["rdr"], a["acc"], 5
            bf = budget_frac(strat, s, k)
            cells.append((g, gsd, rdr, acc, bf))
        d = dict(
            gain_mean=mean([c[0] for c in cells]),
            gain_sd=mean([c[1] for c in cells if not math.isnan(c[1])]) if strat == "OSA" else None,
            rdr_mean=mean([c[2] for c in cells]),
            acc_min=min(c[3] for c in cells),
            bf_min=min(c[4] for c in cells), bf_max=max(c[4] for c in cells),
            cells=cells, n=n)
        out_rows[(strat, k)] = d
        print("  %-4s κ=%.1f  ΔASR=%+.3f  rdet=%4.0f%%  B/Γ=%s  n=%d"
              % (strat, k, d["gain_mean"], d["rdr_mean"],
                 "%.0f–%.0f%%" % (100 * d["bf_min"], 100 * d["bf_max"]), n))

# 写 CSV（逐格）
with open(os.path.join(RES, "r2bde2e_unified.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["strategy", "sigma_rel", "kappa", "delta_asr", "asr_sd",
                "round_detect_pct", "final_acc", "budget_frac", "n_seeds"])
    for strat in ORDER:
        for k in KAPPAS:
            for s in SIGMAS:
                if strat == "OSA":
                    seeds = osa[("OSA", s, k)]
                    g = mean([x["asr"] for x in seeds]) - ASR_NONE
                    gsd = sd([x["asr"] for x in seeds])
                    rdr = mean([x["rdr"] for x in seeds])
                    acc = mean([x["acc"] for x in seeds])
                else:
                    a = bd[(strat, s, k)]
                    g, gsd, rdr, acc = a["gain"], "", a["rdr"], a["acc"]
                w.writerow([strat, s, k, "%.4f" % g, ("%.4f" % gsd) if gsd != "" else "",
                            "%.1f" % rdr, "%.4f" % acc, "%.3f" % budget_frac(strat, s, k), 5])
print("written: %s" % os.path.join(RES, "r2bde2e_unified.csv"))

# LaTeX 表体
print("\n== LaTeX 表体 ==")
for strat in ORDER:
    cols = []
    for k in KAPPAS:
        d = out_rows[(strat, k)]
        if strat == "OSA":
            gtxt = "$%.3f{\\pm}.%d$" % (d["gain_mean"], int(round(1000 * d["gain_sd"])))
        else:
            gtxt = "$%.3f$" % d["gain_mean"]
        cols += [gtxt, "$%d\\%%$" % round(d["rdr_mean"]),
                 "$%.0f{-}\\%.0f\\%%$" % (100 * d["bf_min"], 100 * d["bf_max"])]
    print("%s & %s \\\\" % (strat, " & ".join(cols)))

# ----------------------------------------------------------------------
# 5. Pareto 图：rdet% vs ΔASR（均值 over σ），双面板 κ=0.5 / κ=2
# ----------------------------------------------------------------------
COLOR = {"OURS": figstyle.OKABE["blue"], "OSA": figstyle.OURS,
         "W25": figstyle.OKABE["green"], "Q25": figstyle.OKABE["orange"]}
MARK = {"OURS": "s", "OSA": "o", "W25": "^", "Q25": "D"}

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
for ax, k in zip(axes, KAPPAS):
    # 每个 allocator：σ 逐格淡点 + 均值大点
    for strat in ORDER:
        xs, ys = [], []
        for s in SIGMAS:
            if strat == "OSA":
                seeds = osa[("OSA", s, k)]
                xs.append(mean([x["rdr"] for x in seeds]))
                ys.append(mean([x["asr"] for x in seeds]) - ASR_NONE)
            else:
                a = bd[(strat, s, k)]
                xs.append(a["rdr"]); ys.append(a["gain"])
        ax.scatter(xs, ys, s=14, color=COLOR[strat], alpha=0.30, linewidths=0)
        mx, my = mean(xs), mean(ys)
        ax.scatter([mx], [my], s=52, marker=MARK[strat], color=COLOR[strat],
                   edgecolors="white", linewidths=0.6, zorder=5,
                   label=strat if k == 0.5 else None)
        dy = 0.012 if strat != "Q25" else -0.020
        ax.annotate(strat, (mx, my), xytext=(0, 6 if strat != "Q25" else -12),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=COLOR[strat])
    ax.set_xlabel("round detection rate (%)")
    ax.set_title(r"$\kappa=%s$" % ({0.5: "0.5", 2.0: "2"}[k]))
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlim(-4, 104)
axes[0].set_ylabel(r"$\Delta$ASR (gain over no-attack)")
axes[0].legend(loc="upper left", handletextpad=0.2)
fig.tight_layout()
figstyle.save_all(fig, os.path.join(FIGDIR, "fig13_pareto_bd"))
print("done.")
