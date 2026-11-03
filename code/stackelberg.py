# -*- coding: utf-8 -*-
"""
选题三 P5：隐私-鲁棒性 Stackelberg 均衡 (T4) 合成验证
=====================================================
防御方选 σ 最大化自身效用，攻击方最优响应（P3 的 Optimal-ARA）。
防御方效用 = 隐私合规收益 − 模型精度损失 − 被击穿代价。
  - 隐私合规收益 ∝ ln σ（越大越合规，但边际递减）
  - 精度损失 ∝ σ²（噪声损害模型）
  - 被击穿代价 ∝ K·η_max(σ) ∝ σ（攻击方最优响应隐藏影响随隐私增强而增大，P1/P2 结论）
⇒ 效用呈倒 U 形，存在内部最优 σ*（非单调），纠正"隐私越强越安全"直觉。
"""

import math
import csv
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUTDIR, exist_ok=True)

C = 14.5          # η_max = C·σ (P1, d=1500)
K = 8             # 攻击者腐蚀预算
P_PRIV = 2.5      # 隐私合规收益系数
P_ACC = 0.03      # 精度损失系数 (σ²)
P_BREACH = 0.01   # 击穿代价系数 (× K·C·σ)


def eta_max(sigma):
    return C * sigma


def defender_utility(sigma):
    if sigma <= 0:
        return float("-inf")
    breach = P_BREACH * K * eta_max(sigma)     # 攻击方最优响应隐藏影响
    return P_PRIV * math.log(sigma) - P_ACC * sigma * sigma - breach


def main():
    sigmas = [0.3 + 0.2 * i for i in range(100)]   # 0.3 .. 20.1
    us = [defender_utility(s) for s in sigmas]
    best = max(range(len(us)), key=lambda i: us[i])
    sigma_star = sigmas[best]
    rows = list(zip([round(s, 3) for s in sigmas], [round(u, 4) for u in us]))
    with open(os.path.join(OUTDIR, "stackelberg.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma", "defender_utility"]); w.writerows(rows)

    rep = ["=" * 70,
           " P5 Stackelberg：防御方效用 U(σ) 倒 U 形，内部最优 σ*",
           " η_max=Cσ (C=%.1f, P1), K=%d, 系数 priv=%.2f acc=%.2f breach=%.2f"
           % (C, K, P_PRIV, P_ACC, P_BREACH),
           "=" * 70,
           " σ* (最优隐私) ≈ %.3f" % sigma_star,
           " U(σ*) = %.4f" % us[best],
           " U(0.5)=%.4f  U(4)=%.4f  U(8)=%.4f" % (defender_utility(0.5), defender_utility(4), defender_utility(8)),
           "",
           " 结论：σ 过小→隐私合规不足且攻击也易检(效用低)；σ 过大→精度崩+攻击易隐藏(效用低)；",
           "       内部 σ* 才是均衡，证明'过度 privatize 反而帮攻击者'（T4 反直觉）。"]
    txt = "\n".join(rep)
    with open(os.path.join(OUTDIR, "stackelberg_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return sigma_star, us, sigmas


if __name__ == "__main__":
    main()
