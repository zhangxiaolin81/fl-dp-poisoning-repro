# -*- coding: utf-8 -*-
"""诊断3：带逐轮打印的 FL warmup，定位 acc 不涨的原因。"""
import os, types, torch

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "real_fl_exp.py")).read().replace('if __name__ == "__main__":', 'if False:')
mod = types.ModuleType("expmod"); mod.__dict__["__file__"] = os.path.abspath(__file__)
exec(compile(src, "exp", "exec"), mod.__dict__)

torch.manual_seed(2)
name, shards, val, test = mod.load_data()
sizes = torch.tensor(mod.CLIENT_SIZES, dtype=torch.float)
wts = (sizes / sizes.sum())
gmodel = mod.TinyCNN()
print("init acc %.4f" % mod.evaluate(gmodel, test))
for r in range(6):
    ups = []
    for i in range(mod.N_CLIENTS):
        m = mod.local_train(gmodel, shards[i], mod.LOCAL_EPOCHS)
        ups.append(mod.flat(m) - mod.flat(gmodel))
    U = torch.stack(ups)
    agg = (U * wts.unsqueeze(1)).sum(dim=0)
    print("round %d: |u_i| mean=%.4f std=%.4f |agg|=%.4f  u0[:5]=%s"
          % (r, U.norm(dim=1).mean(), U.norm(dim=1).std(), agg.norm(), ["%.4f" % x for x in agg[:5].tolist()]))
    gmodel2 = mod.TinyCNN(); mod.load_flat(gmodel2, mod.flat(gmodel) + agg)
    gmodel = gmodel2
    print("         acc=%.4f" % mod.evaluate(gmodel, test))
