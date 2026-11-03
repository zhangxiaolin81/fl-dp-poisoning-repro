# -*- coding: utf-8 -*-
"""最小对照：集中式训练 TinyCNN 于同一批 FashionMNIST 数据，验证数据/模型/优化器是否正常。"""
import os, sys, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
spec = importlib.util.spec_from_file_location("expmod", os.path.join(HERE, "real_fl_exp.py"))
mod = importlib.util.module_from_spec(spec)
mod.__dict__["__file__"] = os.path.join(HERE, "real_fl_exp.py")
# 只执行模块定义（不跑 __main__）
import types
src = open(os.path.join(HERE, "real_fl_exp.py")).read().replace('if __name__ == "__main__":', 'if False:')
exec(compile(src, "exp", "exec"), mod.__dict__)

torch.manual_seed(1)
name, shards, val, test = mod.load_data()
xs = torch.cat([s.tensors[0] for s in shards])
ys = torch.cat([s.tensors[1] for s in shards])
print("train pool:", xs.shape, ys.shape, "label min/max:", ys.min().item(), ys.max().item())

m = mod.TinyCNN()
opt = torch.optim.SGD(m.parameters(), lr=0.08)
dl = DataLoader(TensorDataset(xs, ys), batch_size=64, shuffle=True)
for ep in range(3):
    m.train(); tot = corr = 0.0
    for xb, yb in dl:
        opt.zero_grad()
        out = m(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward(); opt.step()
        tot += loss.item() * len(yb); corr += (out.argmax(1) == yb).sum().item()
    print("epoch %d: train loss %.4f  train acc %.4f" % (ep, tot / len(xs), corr / len(xs)))
print("test acc:", mod.evaluate(m, test))
