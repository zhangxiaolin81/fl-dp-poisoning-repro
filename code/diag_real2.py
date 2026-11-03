# -*- coding: utf-8 -*-
"""诊断2：探测 vs 部署的距离分布对比。"""
import os, sys, math, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "real_fl_exp.py")).read()
src = src.replace('if __name__ == "__main__":', 'if False:')
mod = types.ModuleType("expmod"); mod.__dict__["__file__"] = os.path.abspath(__file__)
exec(compile(src, "exp", "exec"), mod.__dict__)

torch.manual_seed(mod.SEED)
name, shards, val, test = mod.load_data()
ck = torch.load(mod.CKPT, weights_only=False)
U0 = ck["U"]; unit = ck["unit_norm"]; gv = ck["gvec"]
benign_std = U0.std(dim=0).mean().item()
sigma = 1.0 * benign_std
mal = list(range(mod.N_CLIENTS - mod.K_MAL, mod.N_CLIENTS))
v = -gv

print("=== A. 探测设置（U0 池，单恶意注入 ρ=0.71u，方向 -ĝ）===")
g = torch.Generator().manual_seed(7)
i = mal[0]
for trial in range(3):
    noise = torch.randn(U0.shape, generator=g) * sigma
    U = U0 + noise
    U[i] = U[i] + 0.71 * unit * v
    med = U.median(dim=0).values
    s = (U - med).norm(dim=1)
    ben = torch.cat([s[:i], s[i+1:]])
    thr = ben.mean() + 1.645 * ben.std()
    print("  s_mal[%d]=%7.3f  s_ben mean=%6.3f std=%5.3f thr=%6.3f  flagged=%s"
          % (i, s[i], ben.mean(), ben.std(), thr, bool(s[i] > thr)))

print("=== B. 部署设置（6 恶意同时 0.71u，方向 -ĝ，fresh benign updates）===")
gmodel = mod.TinyCNN(); mod.load_flat(gmodel, ck["gmodel"])
g = torch.Generator().manual_seed(11)
for trial in range(3):
    ups = []
    for c in range(mod.N_CLIENTS):
        m = mod.local_train(gmodel, shards[c], mod.LOCAL_EPOCHS)
        ups.append(mod.flat(m) - mod.flat(gmodel))
    U = torch.stack(ups)
    for c in mal:
        U[c] = U[c] + 0.71 * unit * v
    noise = torch.randn(U.shape, generator=g) * sigma
    Un = U + noise
    med = Un.median(dim=0).values
    s = (Un - med).norm(dim=1)
    ben_idx = [c for c in range(mod.N_CLIENTS) if c not in mal]
    sb = s[ben_idx]
    thr = sb.mean() + 1.645 * sb.std()
    print("  s_mal=%s" % ["%.2f" % x for x in s[mal].tolist()])
    print("  s_ben mean=%6.3f std=%5.3f thr=%6.3f   ben_idx flagged=%s"
          % (sb.mean(), sb.std(), thr, [ben_idx[k] for k in (sb > thr).nonzero().flatten().tolist()]))
    # 良性更新沿 -ĝ 的投影分布（共识度）
    proj = (U[ben_idx] @ v)
    print("  benign proj onto -g: mean=%.3f std=%.3f ; mal proj=%s"
          % (proj.mean(), proj.std(), ["%.2f" % x for x in (U[mal] @ v).tolist()]))
