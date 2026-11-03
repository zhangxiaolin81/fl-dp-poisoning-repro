# -*- coding: utf-8 -*-
"""诊断：真实管线上 sep 的分解（mean_ben / std / mean_mal）随 σ 与 ρ 变化。"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("exp", os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_fl_exp.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module.__self__ if False else None
# 只取函数，不触发 __main__
import types
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_fl_exp.py")).read()
src = src.replace('if __name__ == "__main__":', 'if False:')
mod = types.ModuleType("expmod")
mod.__dict__["__file__"] = os.path.abspath(__file__)
exec(compile(src, "exp", "exec"), mod.__dict__)

CKPT = os.path.join(mod.ROOT, "results", "real", "warmup.pt")
ck = torch.load(CKPT, weights_only=False)
U0 = ck["U"]; d = ck["d"]; unit = ck["unit_norm"]
mal = list(range(mod.N_CLIENTS - mod.K_MAL, mod.N_CLIENTS))
ben_std_coord = U0.std(dim=0).mean().item()
print("unit_norm=%.3f  benign per-coord std=%.5f  ||b_i-med||=%.3f" % (unit, ben_std_coord, (U0 - U0.median(dim=0).values).norm(dim=1).mean().item()))

g = torch.Generator().manual_seed(7)
for srel in (1.0, 16.0):
    sigma = srel * ben_std_coord
    for rho_u in (0.5, 1.0, 2.0, 4.0):
        s_b, s_m = [], []
        for _ in range(8):
            noise = torch.randn(U0.shape, generator=g) * sigma
            U = U0 + noise
            v = torch.randn(d, generator=g); v = v / v.norm()
            for i in mal:
                U[i] = U[i] + rho_u * unit * v
            med = U.median(dim=0).values
            s = (U - med).norm(dim=1)
            s_b.append(float(torch.cat([s[:mal[0]], s[mal[-1]+1:]]).mean()))
            s_m.append(float(s[mal].mean()))
        mb, mm = sum(s_b)/8, sum(s_m)/8
        print("sigma_rel=%5.1f rho=%4.1fu: mean_ben=%8.3f mean_mal=%8.3f  shift=%7.4f  (sep~shift/std)" % (srel, rho_u, mb, mm, mm-mb))
