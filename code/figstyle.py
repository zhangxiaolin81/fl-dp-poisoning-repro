# -*- coding: utf-8 -*-
"""figstyle.py — 论文全部图件的统一出版样式（academic-plotting skill 规范）。

用法：
    import figstyle
    figstyle.apply()          # 在创建 figure 之前调用
    ...
    figstyle.save_all(fig, "…/paper/figures/fig7_r1_tworegime")  # 同时导出 PDF(矢量) + PNG(300dpi)

规范来源：Elsevier 排版（vector PDF 优先、Type 42 字体嵌入）+
Okabe-Ito 色盲安全色板 + serif 字体匹配 elsarticle(times) 正文。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito 色盲安全色板（数据图强制）
OKABE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "pink": "#CC79A7", "black": "#000000",
}
OURS = OKABE["vermillion"]      # 本文贡献/主数据集
CONTRAST = OKABE["blue"]        # 对照数据集/基线
BASELINE = "#8C8C8C"            # 中性灰（弱化基线）

def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "-",
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # Type 42 可嵌入字体
        "savefig.bbox": "tight",
    })

def save_all(fig, path_no_ext, dpi=300):
    """同基名导出矢量 PDF + 300 DPI PNG。"""
    d = os.path.dirname(path_no_ext)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path_no_ext + ".pdf", dpi=dpi)
    fig.savefig(path_no_ext + ".png", dpi=dpi)
    print("saved: %s.pdf + %s.png" % (path_no_ext, path_no_ext))
