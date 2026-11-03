# -*- coding: utf-8 -*-
"""
B 实验探测：在 FashionMNIST 上用更少 warmup 轮数（25 vs 60），
制造一个「难且 capacity 充裕」的场景 —— 未充分训练的模型 benign 更新更分散
（eta_max 增大 → C/Gamma 翻转至 >1, deployable），同时 margin 更紧使 damage 可观测。
通过 monkey-patch DATASET 带 tag，输出到独立文件，不污染现有 60 轮结果。

目标：验证 OURS 在 deployable regime 实打实展示 frontier 优势
（OURS drop >= 无约束基线 W25/Q25，且 C/Gamma 估计 >1）。
"""
import real_fl_exp as R

# 用 tag 隔离：DATASET=fmnist_r25 -> 模型走 TinyCNN(非cifar)、数据走 load_fmnist、
# CKPT=warmup_fmnist_r25_r25.pt、r2 输出 r2_main_fmnist_r25.csv（不覆盖正式结果）
R.DATASET = "fmnist_r25"
R.WARMUP_ROUNDS = 25
R.SIGMA_GRID = [0.5, 1.0, 2.0]
R.KAPPA_GRID = [0.25, 0.5, 1.0, 2.0]
R.R2_SEEDS = 3          # 探测用少种子，快速看趋势
R.ATTACK_ROUNDS = 5

print("[probe] reduced-warmup FMNIST: warmup(25) + r2 small grid")
R.stage_warmup()
R.stage_r2()
print("[probe] DONE -> results/real/r2_main_fmnist_r25.csv")
