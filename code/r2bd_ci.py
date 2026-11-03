# -*- coding: utf-8 -*-
"""
R2b-ASR 表的 95% 置信区间（审稿 Major 5：至少报告 CI 或统计功效）
================================================================
从 r2_main_bd_cifar10_seeds.csv（逐种子）计算每个 (strategy, kappa) 的
per-cell 95% CI 半宽（5 seeds，df=4，t=2.776），再对 σ∈{0.5,1,2} 取最大
半宽作为表中 ± 值（保守、诚实）。
输出：results/real/r2bd_ci_cifar10.csv + LaTeX 行打印
"""
import csv, math, os
import statistics as st

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "real")
T95_DF4 = 2.776  # two-sided 95%, df=4

rows = list(csv.DictReader(open(os.path.join(RES, "r2_main_bd_cifar10_seeds.csv"))))
cells = {}
for r in rows:
    key = (r["strategy"], float(r["kappa"]), float(r["sigma_rel"]))
    cells.setdefault(key, []).append(float(r["asr_gain_vs_none"]))

ORDER = ["OURS", "S21", "W25", "H25", "Q25", "UNI"]
out = []
for strat in ORDER:
    for kap in (0.5, 2.0):
        keys = [k for k in cells if k[0] == strat and k[1] == kap]
        assert len(keys) == 3, (strat, kap, keys)
        means, hws, rdet = [], [], []
        for key in sorted(keys):
            v = cells[key]
            m = st.mean(v)
            hw = T95_DF4 * st.stdev(v) / math.sqrt(len(v))
            means.append(m); hws.append(hw)
        pooled = [x for k in sorted(keys) for x in cells[k]]
        out.append({
            "strategy": strat, "kappa": kap,
            "mean_dasr": st.mean(pooled),
            "ci_max": max(hws), "ci_mean": st.mean(hws),
            "pooled_n": len(pooled),
        })

with open(os.path.join(RES, "r2bd_ci_cifar10.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["strategy", "kappa", "mean_dasr", "ci_max", "ci_mean", "pooled_n"])
    for o in out:
        w.writerow([o["strategy"], o["kappa"], "%.4f" % o["mean_dasr"],
                    "%.4f" % o["ci_max"], "%.4f" % o["ci_mean"], o["pooled_n"]])

print("== tab:r2bd LaTeX 行（ΔASR ± max per-cell 95% CI half-width）==")
for strat in ORDER:
    d = {o["kappa"]: o for o in out if o["strategy"] == strat}
    print("%s & $%.3f{\\pm}%.3f$ & $%.3f{\\pm}%.3f$ \\\\" % (
        strat, d[0.5]["mean_dasr"], d[0.5]["ci_max"],
        d[2.0]["mean_dasr"], d[2.0]["ci_max"]))
print("\n== 全局功效语句 ==")
hw_all = max(o["ci_max"] for o in out)
print("max per-cell CI half-width = %.4f -> 设计可分辨 ΔASR 差异 >= %.3f" % (hw_all, 2 * hw_all))

# --- 配对差（同 seed 同 cell）：关键排序主张的 95% CI ---
print("\n== 配对差 per-cell（5 seeds，df=4）==")
byc = {}
for r in rows:
    key = (r["strategy"], float(r["kappa"]), float(r["sigma_rel"]), int(r["seed"]))
    byc[key] = float(r["asr_gain_vs_none"])
for a, b in (("W25", "OURS"), ("W25", "Q25"), ("Q25", "OURS")):
    for kap in (0.5, 2.0):
        diffs = []
        for s in (0.5, 1.0, 2.0):
            dv = [byc[(a, kap, s, sd)] - byc[(b, kap, s, sd)] for sd in range(5)]
            diffs.append((st.mean(dv), T95_DF4 * st.stdev(dv) / math.sqrt(5)))
        print(" %s-%s kappa=%.1f: per-cell mean+-CI = %s; all_pos=%s" % (
            a, b, kap,
            ", ".join("%+.3f+-%.3f" % d for d in diffs),
            all(m > 0 for m, _ in diffs)))
