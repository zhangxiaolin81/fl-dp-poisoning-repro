# -*- coding: utf-8 -*-
"""fig5_update.py — Fig.5 改版：σ*(a) 曲线升为主结果（审稿意见：a 不可识别，
σ*=1.9 只作校准示例）。
  A 面板：U(σ) 实测分量（原单面板内容不变）。
  B 面板：最优响应 σ*(a) 曲线（a∈[0.005,0.12]），标注 σ_c=13.4 floor-regime
          有效性上界与中心校准点 (a=0.0286, σ*=1.9)。
数据：results/real/t4_params_cifar10.npz（b1,b2,d0,d1,a_19）。
运行：python fig5_update.py  （只重绘 fig5，不触碰 fig2/fig4）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.apply()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "paper", "figures")

p = np.load(os.path.join(BASE, "results", "real", "t4_params_cifar10.npz"))
b1, b2, d0, d1 = float(p["b1"]), float(p["b2"]), float(p["d0"]), float(p["d1"])
a_19 = float(p["a_19"])
SIGMA_C = 13.4   # 两区制 crossover（Proposition tworegime，正文口径）

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))

# ---- A：U(σ) 分量分解（中心校准 a=0.0286） ----
xs = np.linspace(0.05, 8, 200)
U = a_19 * np.log(xs) - (b1 * xs + b2 * xs ** 2) - (d0 + d1 * xs)
i = int(np.argmax(U))
ax[0].plot(xs, U, "-", color="#0072B2", lw=1.6)
ax[0].plot(xs[i], U[i], "o", color="#D55E00", ms=6)
ax[0].annotate(r"$\sigma^\star=1.9$", (xs[i], U[i]), textcoords="offset points",
               xytext=(8, 8), fontsize=9, color="#D55E00")
ax[0].axvline(xs[i], color="#D55E00", ls=":", lw=0.9, alpha=0.6)
ax[0].plot(xs, a_19 * np.log(xs), ":", color="#009E73", lw=1.0,
           label=r"privacy benefit $a\ln\sigma$")
ax[0].plot(xs, -(b1 * xs + b2 * xs ** 2), "--", color="#E69F00", lw=1.0,
           label=r"utility loss (measured)")
ax[0].plot(xs, -(d0 + d1 * xs), "-.", color="#D55E00", lw=1.0,
           label=r"breach envelope $D(\sigma)$ (measured)")
ax[0].set_xlabel(r"$\sigma_{\mathrm{rel}}$")
ax[0].set_ylabel(r"defender utility $U(\sigma)$")
ax[0].set_title(r"A  $U(\sigma)$ at the central calibration $a=%.4f$" % a_19,
                fontsize=9)
ax[0].legend(fontsize=7, loc="lower left")

# ---- B：σ*(a) 最优响应曲线（主结果） ----
aa = np.linspace(0.005, 0.12, 200)
star = (-(b1 + d1) + np.sqrt((b1 + d1) ** 2 + 8 * b2 * aa)) / (4 * b2)
ax[1].plot(aa, star, "-", color="#0072B2", lw=1.6,
           label=r"$\sigma^\star(a)$: interior optimum")
ax[1].axhline(SIGMA_C, color="#8C8C8C", ls=":", lw=1.0)
ax[1].annotate(r"$\sigma_c=13.4$ (floor-regime validity bound)",
               (0.008, SIGMA_C), textcoords="offset points", xytext=(0, 4),
               fontsize=7, color="#8C8C8C")
ax[1].plot(a_19, 1.9, "o", color="#D55E00", ms=6)
ax[1].annotate(r"$(a{=}0.0286,\ \sigma^\star{=}1.9)$",
               (a_19, 1.9), textcoords="offset points", xytext=(8, -12),
               fontsize=8, color="#D55E00")
ax[1].set_xlabel(r"privacy exchange rate $a$ (acc-equiv./ln-unit)")
ax[1].set_ylabel(r"optimal noise $\sigma^\star(a)$")
ax[1].set_title("B  optimal response: the primary calibrated object",
                fontsize=9)
ax[1].legend(fontsize=7, loc="upper left")

fig.tight_layout(pad=0.4)
figstyle.save_all(fig, os.path.join(OUT, "fig5_stackelberg"))
plt.close(fig)
