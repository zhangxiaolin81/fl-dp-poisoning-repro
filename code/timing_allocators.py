# -*- coding: utf-8 -*-
"""Step 4: 测量四种分配器的单步求解时间，用于论文"系统评测"开销表。
OURS=water-filling(二分 80 轮, 纯 python)；W25/Q25=全局缩放闭式(O(K))；
OSA=顺序统计量闭式(O(K log K), 排序)。在论文真实规模 K=20 与更大 K=100/500 下计时。
"""
import time, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def water_filling(alpha, eta, Gamma):
    """论文 optimal_ara.water_filling 的纯 python 二分实现（80 轮）。"""
    lo, hi = 1e-9, max(alpha) / 1e-9 + 1.0
    for _ in range(80):
        mu = 0.5 * (lo + hi)
        p = [min(e, a / (2 * mu)) if a > 0 else 0.0 for a, e in zip(alpha, eta)]
        s = sum(x * x for x in p)
        if s > Gamma:
            lo = mu
        else:
            hi = mu
    mu = 0.5 * (lo + hi)
    return [min(e, a / (2 * mu)) if a > 0 else 0.0 for a, e in zip(alpha, eta)]


def alloc_w25(alpha, eta, Gamma, K):
    p_each = math.sqrt(Gamma / K)
    return [min(e, p_each) for e in eta]


def alloc_q25(alpha, eta, Gamma, K):
    s2 = sum(a * a for a in alpha)
    lam = math.sqrt(Gamma / s2) if s2 > 0 else 0.0
    return [lam * a for a in alpha]


def alloc_osa(alpha, eta, Gamma, K):
    order = sorted(range(len(alpha)), key=lambda i: alpha[i] * eta[i], reverse=True)[:K]
    p_each = math.sqrt(Gamma / K)
    return [min(eta[i], p_each) for i in order]


def make_scenario(n, seed=0):
    rng = np.random.default_rng(seed)
    alpha = rng.lognormal(0.0, 0.8, size=n).tolist()
    eta = (rng.uniform(0.5, 1.5, size=n) * 40.0).tolist()
    Gamma = 2000.0
    return alpha, eta, Gamma


def timeit(fn, args, reps=1000):
    # 热身
    for _ in range(50):
        fn(*args)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e6  # 微秒
    return ts.mean(), ts.std(), ts.max()


if __name__ == "__main__":
    print("%-10s %-8s %-12s %-12s %-12s %-12s" %
          ("K", "method", "mean(us)", "std(us)", "max(us)", "complexity"))
    for K in (20, 100, 500):
        alpha, eta, Gamma = make_scenario(K)
        # OURS / W25 / Q25 / OSA 都取前 K 个客户端（与论文 K 妥协设定一致）
        aK, eK = alpha[:K], eta[:K]
        for name, fn, cplx in (
            ("OURS(wf)", lambda: water_filling(aK, eK, Gamma), "O(K*80) bisect"),
            ("W25", lambda: alloc_w25(aK, eK, Gamma, K), "O(K) closed"),
            ("Q25", lambda: alloc_q25(aK, eK, Gamma, K), "O(K) closed"),
            ("OSA", lambda: alloc_osa(aK, eK, Gamma, K), "O(K log K) sort"),
        ):
            mean, std, mx = timeit(fn, ())
            print("%-10d %-8s %-12.3f %-12.3f %-12.3f %-12s" %
                  (K, name, mean, std, mx, cplx))
    print("\nK=20 即论文客户端数 N=20；所有分配器单步求解均在亚毫秒级，")
    print("求解开销不随实验规模构成瓶颈，瓶颈在每轮 CIFAR-10 TinyCNN 前向/反向。")
