# -*- coding: utf-8 -*-
"""fig_paper_missing.py — 生成论文 Fig.2/4/5（数据全部来自 results/ 下 CSV/NPZ）。
  fig2: 安全阈值边界（单面板: η_max(σ) 检出/逃逸边界; 原 d^{1/4} 面板已并入 fig4B）
  fig4: T1 标度坍缩 sep vs x = ρ²/(σ²√d)，过原点拟合 c≈0.30, R²=0.984
        + B面板: η_max vs d 的 d^{1/4} 标度（自 fig2 迁入）
  fig5: Stackelberg 实测标定 U(σ)（t4_params_cifar10.npz），内点最优 σ*=1.9
        单面板; σ*(a) 曲线压缩为正文一句话（审稿压缩清单 4.4/4.5）
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
figstyle.apply()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

# ---------- fig2（单面板: 安全阈值边界; d^{1/4} 面板迁入 fig4B） ----------
b = rd(os.path.join(R, "safety_threshold_boundary.csv"))
fig, ax = plt.subplots(figsize=(4.6, 3.2))
sx = [float(r["sigma"]) for r in b]; sy = [float(r["eta_max"]) for r in b]
ax.fill_between(sx, sy, max(sy) * 1.15, color="#f6c", alpha=0.15)
ax.fill_between(sx, 0, sy, color="#69c", alpha=0.15)
ax.plot(sx, sy, "o-", color="#246", lw=1.5, ms=3.5, label=r"$\eta_{\max}(\sigma)$")
ax.text(0.35, max(sy) * 0.92, "escape\n(stealthy)", color="#a36", fontsize=8)
ax.text(3.2, max(sy) * 0.30, "detected", color="#246", fontsize=8)
ax.set_xlabel(r"DP noise scale $\sigma$"); ax.set_ylabel(r"stealthy capacity $\eta_{\max}$")
ax.set_title("safety-threshold boundary", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
figstyle.save_all(fig, os.path.join(OUT, "fig2_safety_boundary")); plt.close(fig)

# ---------- fig4（双面板: A 数据坍缩; B η_max vs d 的 d^{1/4} 标度） ----------
rows = rd(os.path.join(R, "fig4_collapse.csv"))
xs = [float(r["x"]) for r in rows]; ys = [float(r["sep"]) for r in rows]
c = sum(y * x for y, x in zip(ys, xs)) / sum(x * x for x in xs)
ss_res = sum((y - c * x) ** 2 for y, x in zip(ys, xs))
ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
r2 = 1 - ss_res / ss_tot
d = rd(os.path.join(R, "eta_max_d.csv"))
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
xx = [10 ** (math.log10(min(xs)) - 0.15), 10 ** (math.log10(max(xs)) + 0.15)]
ax[0].loglog(xs, ys, "o", color="#246", ms=4, label="measurement (54 cfgs)")
ax[0].loglog(xx, [c * t for t in xx], "-", color="#a33", lw=1.5,
             label=r"fit $\mathrm{sep}=0.30\,x$, $R^2=0.984$")
ax[0].set_xlabel(r"$x=\rho^2/(\sigma^2\sqrt{d})$"); ax[0].set_ylabel("Fisher separation sep")
ax[0].set_title("A  T1 data collapse onto $\\mathrm{sep}=c\\,x$", fontsize=9)
ax[0].legend(fontsize=8)
dx = [float(r["d"]) for r in d]; dy = [float(r["eta_max"]) for r in d]
q = [float(r["eta_max_over_d025"]) for r in d]
ax[1].loglog(dx, dy, "o-", color="#246", lw=1.5, ms=3.5, label=r"$\eta_{\max}(d)$")
ax[1].loglog(dx, q, "--", color="#a66", lw=1.2, label=r"$\propto d^{1/4}$ ref.")
ax[1].set_xlabel(r"dimension $d$"); ax[1].set_ylabel(r"$\eta_{\max}$")
ax[1].set_title("B  $d^{1/4}$ scaling (refutes $\\sqrt{d/d_0}$)", fontsize=9)
ax[1].legend(fontsize=7)
figstyle.save_all(fig, os.path.join(OUT, "fig4_collapse")); plt.close(fig)

# ---------- fig5（单面板: 实测标定 U(σ) 与分量分解; σ*(a) 曲线入正文一句话） ----------
import numpy as np
p = np.load(os.path.join(R, "real", "t4_params_cifar10.npz"))
b1, b2, d0, d1 = float(p["b1"]), float(p["b2"]), float(p["d0"]), float(p["d1"])
a_19 = float(p["a_19"])
fig, ax = plt.subplots(figsize=(4.8, 3.1))
xs2 = np.linspace(0.05, 8, 200)
U = a_19 * np.log(xs2) - (b1 * xs2 + b2 * xs2 ** 2) - (d0 + d1 * xs2)
i = int(np.argmax(U))
ax.plot(xs2, U, "-", color="#246", lw=1.6)
ax.plot(xs2[i], U[i], "o", color="#a33", ms=6)
ax.annotate(r"$\sigma^\star=1.9$", (xs2[i], U[i]), textcoords="offset points",
            xytext=(8, 8), fontsize=9, color="#a33")
ax.axvline(xs2[i], color="#a33", ls=":", lw=0.9, alpha=0.6)
ax.plot(xs2, a_19 * np.log(xs2), ":", color="#686", lw=1.0, label=r"privacy benefit $a\ln\sigma$")
ax.plot(xs2, -(b1 * xs2 + b2 * xs2 ** 2), "--", color="#a66", lw=1.0, label=r"utility loss (measured)")
ax.plot(xs2, -(d0 + d1 * xs2), "-.", color="#a33", lw=1.0, label=r"breach envelope $D(\sigma)$ (measured)")
ax.set_xlabel(r"$\sigma_{\mathrm{rel}}$"); ax.set_ylabel(r"defender utility $U(\sigma)$")
ax.set_title("calibrated non-monotone defender utility (T4)", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
figstyle.save_all(fig, os.path.join(OUT, "fig5_stackelberg")); plt.close(fig)

# ---------- fig6 ----------
rows = rd(os.path.join(R, "fig6_ablation.csv"))
cols = ["full", "-select", "-cap", "-budget"]
styles = {"full": ("#246", "-", "o"), "-select": ("#a66", "--", "s"),
          "-cap": ("#a33", "-.", "^"), "-budget": ("#686", ":", "D")}
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for cname in cols:
    st, ls, mk = styles[cname]
    gx = [float(r["Gamma"]) for r in rows]
    gy = [float(r[cname]) for r in rows]
    ax.plot(gx, gy, ls, color=st, marker=mk, ms=3.5, lw=1.4, label=cname)
ax.set_xscale("log")
ax.set_xlabel(r"budget $\Gamma$"); ax.set_ylabel("hidden impact $\\sum\\alpha_i p_i$")
ax.set_title("ablation: marginal value of each component", fontsize=9)
ax.legend(fontsize=8)
figstyle.save_all(fig, os.path.join(OUT, "fig6_ablation")); plt.close(fig)

print("saved: fig2_safety_boundary / fig4_collapse / fig5_stackelberg ->", OUT)
