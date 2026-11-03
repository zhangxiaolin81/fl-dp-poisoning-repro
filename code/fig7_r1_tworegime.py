"""Fig.7: 真实管线 R1 —— T1 两段式容量律的跨数据集验证。
左面板: eta_max 对 sigma_rel(双数据集)，并叠加 eta=sigma(正比参考虚线)。
右面板: eta_max/sigma 对 sigma_rel，展示噪声主导区 eta/sigma 下降 = 两段式证据。
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import figstyle
figstyle.apply()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "real")
FIG = os.path.join(ROOT, "paper", "figures")
C_FMN = figstyle.OURS        # FashionMNIST（主数据集）
C_CIF = figstyle.CONTRAST    # CIFAR-10

def load(ds):
    s, e, r = [], [], []
    with open(os.path.join(OUT, "r1_eta_max%s.csv" % ("_%s" % ds if ds == "cifar10" else ""))) as f:
        for row in csv.DictReader(f):
            s.append(float(row["sigma_rel"])); e.append(float(row["eta_max_unitnorms"]))
            r.append(float(row["eta_over_sigma"]))
    return np.array(s), np.array(e), np.array(r)

sf, ef, rf = load("fmnist")
sc, ec, rc = load("cifar10")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))

# Panel A: eta_max vs sigma, with eta = sigma reference (naive proportionality)
ax[0].plot(sf, ef, "o-", color=C_FMN, lw=2, ms=6, label="FashionMNIST (d=20490)")
ax[0].plot(sc, ec, "s-", color=C_CIF, lw=2, ms=6, label="CIFAR-10 (d=60362)")
sx = np.linspace(0.4, 16, 50)
ax[0].plot(sx, sx, "k--", lw=1.4, label=r"$\eta_{\max}\propto\sigma$ (naive)")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel(r"$\sigma_{\mathrm{rel}}$ (DP noise, $\times$benign std)")
ax[0].set_ylabel(r"$\eta_{\max}$ (stealthy capacity per client)")
ax[0].set_title("A. $\eta_{\max}$ grows far slower than $\sigma$\n(two-regime refinement: heterogeneity floor dominates)")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].grid(True, which="both", ls=":", alpha=0.4)
for x, y in zip(sf, ef):
    ax[0].annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8.5, color=C_FMN)
for x, y in zip(sc, ec):
    ax[0].annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8.5, color=C_CIF)

# Panel B: eta_max/sigma vs sigma
ax[1].plot(sf, rf, "o-", color=C_FMN, lw=2, ms=6, label="FashionMNIST")
ax[1].plot(sc, rc, "s-", color=C_CIF, lw=2, ms=6, label="CIFAR-10")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].yaxis.set_major_locator(MaxNLocator(5))
ax[1].set_xlabel(r"$\sigma_{\mathrm{rel}}$")
ax[1].set_ylabel(r"$\eta_{\max}/\sigma$ (per-unit-noise capacity)")
ax[1].set_title("B. $\eta_{\max}/\sigma$ falls with $\sigma$\n(escape zone shrinks per noise unit in noise-dominated regime)")
ax[1].legend(fontsize=8, loc="upper right")
ax[1].grid(True, which="both", ls=":", alpha=0.4)
for x, y in zip(sf, rf):
    ax[1].annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8.5, color=C_FMN)
for x, y in zip(sc, rc):
    ax[1].annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8.5, color=C_CIF)

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "fig7_r1_tworegime.png")
figstyle.save_all(fig, os.path.join(FIG, "fig7_r1_tworegime"))
print("saved", out)
print("FMNIST eta_max slope (log-log):", np.polyfit(np.log(sf), np.log(ef), 1)[0])
print("CIFAR  eta_max slope (log-log):", np.polyfit(np.log(sc), np.log(ec), 1)[0])
