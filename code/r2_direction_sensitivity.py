# -*- coding: utf-8 -*-
"""
选题三 TODO-1③：Shejwalkar NDSS'21 AGR-tailored 方向敏感性分析
================================================================
目的：把本管线默认的攻击方向（−ĝ，验证集损失梯度上升方向）替换为
      Shejwalkar 在 Median 上的 AGR-tailored 攻击方向（见其官方 notebook
      release-fedsgd-alexnet-median.ipynb, cell 13），在【保持 DP 掩蔽隐蔽上限
      η_max 不变】的前提下重跑 R2，检验双机制（T2 容量约束）与 T3（过 privatize
      助攻击者）结论是否对"攻击方向 sophistication"鲁棒。

Shejwalkar 官方 attack（中位数聚合）：
    deviation = model_re / ||model_re||   (unit_vec, 良性参考方向)
              = sign(model_re)            (sign, 坐标符号)
              = std(all_updates,0)        (std, 良性方差权重)
    mal_update = model_re − λ·deviation   → 聚合位移 = −deviation 方向（对抗良性）
    恶意更新全部相等（集中）；幅度由二分最大化 ‖median − model_re‖ 得到。

本脚本的忠实映射（保持我们的隐蔽上限 η_max，因为 DP 掩蔽 + 鲁棒检测是本文的
约束，Shejwalkar 原攻击无 DP）：
    攻击注入方向 v = −deviation（均指向"增大损失"，与 −ĝ 同性质，但坐标结构不同）
      - grad     : v = −ĝ                       （本管线默认，flat 归一化）
      - shej_unit: v = −mean(U0)/||mean(U0)||     （Shejwalkar unit_vec）
      - shej_sign: v = −sign(mean(U0)) 归一化      （Shejwalkar sign，坐标符号 tailored）
      - shej_std : v = −std(U0,0) 归一化           （Shejwalkar std，方差权重 tailored）
    幅度：沿用 η_max（逐客户端二分隐蔽上限，按各自方向重算）。

结论读法：若 shej_sign/shej_std 在 CIFAR 上把 OURS 推上 stealthy-damage 前沿，
          说明 −ĝ 低估了损伤（tailored 是上界）；若仍被 Q25 占据前沿，则容量受限
          结论对攻击方向鲁棒——这是更强的发表级稳健性证据。
"""
import os, math, time, argparse, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms

torch.manual_seed(20261027)
torch.set_num_threads(8)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "real")
os.makedirs(OUT, exist_ok=True)

# ---------------- 配置（与 real_fl_exp.py 一致） ----------------
N_CLIENTS = 20
K_MAL = 6
VAL_SAMPLES = 2000
WARMUP_ROUNDS = 60
ATTACK_ROUNDS = 5
R2_SEEDS = 10
LOCAL_EPOCHS = 2
ATTACK_LOCAL_EPOCHS = 1
BATCH = 64
LR = 0.08
SEED = 20261027
DATASET = "fmnist"
SIGMA_GRID = [0.5, 1.0, 2.0]
KAPPA_GRID = [0.5, 2.0]            # 松弛 vs 紧张（封顶）两 regime
CLIENT_SIZES = [300, 450, 600, 750, 900] * 4
CIFAR_ROOT = os.environ.get("CIFAR_ROOT", os.path.join(ROOT, "data", "cifar10"))


def _resolve_ckpt(dataset):
    p = os.path.join(OUT, "warmup_%s.pt" % dataset)
    if os.path.exists(p):
        return p
    # FMNIST ckpt 在路径 bug 修复时被复制回为 warmup.pt
    if dataset == "fmnist":
        alt = os.path.join(OUT, "warmup.pt")
        if os.path.exists(alt):
            return alt
    return p


CKPT = _resolve_ckpt(DATASET)
DIRECTION_GRID = ["grad", "shej_unit", "shej_sign", "shej_std"]
STRAT_GRID = ["OURS", "Q25"]


# ---------------- 模型 ----------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def get_model():
    return CIFAR10CNN() if DATASET == "cifar10" else TinyCNN()


# ---------------- 数据 ----------------
def load_fmnist():
    # FashionMNIST 默认由 torchvision 自动下载到 <repo>/data；无网络时请预先放置本地数据。
    tr = torchvision.datasets.FashionMNIST(os.path.join(ROOT, "data"), train=True, download=True)
    te = torchvision.datasets.FashionMNIST(os.path.join(ROOT, "data"), train=False, download=True)
    name = "FashionMNIST"
    x = tr.data.unsqueeze(1).float() / 255.0
    y = tr.targets.long()
    xt = te.data.unsqueeze(1).float() / 255.0
    yt = te.targets.long()
    return name, x, y, xt, yt


def load_cifar():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))])
    tr = torchvision.datasets.CIFAR10(CIFAR_ROOT, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10(CIFAR_ROOT, train=False, download=True, transform=tf)
    x = torch.stack([t[0] for t in tr])
    y = torch.tensor([t[1] for t in tr], dtype=torch.long)
    xt = torch.stack([t[0] for t in te])
    yt = torch.tensor([t[1] for t in te], dtype=torch.long)
    return "CIFAR10", x, y, xt, yt


def load_data():
    name, x, y, xt, yt = load_cifar() if DATASET == "cifar10" else load_fmnist()
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(x), generator=g)
    shards, off = [], 0
    for i in range(N_CLIENTS):
        n_i = CLIENT_SIZES[i]
        s = perm[off:off + n_i]; off += n_i
        shards.append(TensorDataset(x[s], y[s]))
    vt = perm[-VAL_SAMPLES:]
    val = TensorDataset(x[vt], y[vt])
    test = TensorDataset(xt, yt)
    return name, shards, val, test


# ---------------- 工具 ----------------
def flat(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])


def load_flat(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data = vec[i:i + n].view_as(p.data)
        i += n


def local_train(model, ds, epochs=1):
    m = get_model(); load_flat(m, flat(model))
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=LR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
    return m


@torch.no_grad()
def evaluate(model, ds):
    m = model.eval()
    dl = DataLoader(ds, batch_size=512)
    c = t = 0
    for xb, yb in dl:
        c += (m(xb).argmax(1) == yb).sum().item(); t += len(yb)
    return c / t


def coord_median(U):
    return U.median(dim=0).values


def distance_scores(U, med):
    return (U - med).norm(dim=1)


# ---------------- 方向向量 ----------------
def direction_vector(ck, name):
    """返回归一化攻击注入方向 v（指向增大损失）。"""
    U0 = ck["U"]; gv = ck["gvec"]
    if name == "grad":
        return -gv                                       # 本管线默认：−ĝ
    ref = U0.mean(0)
    if name == "shej_unit":
        v = -ref / ref.norm()                            # Shejwalkar unit_vec
    elif name == "shej_sign":
        v = -torch.sign(ref)                             # Shejwalkar sign（坐标符号）
    elif name == "shej_std":
        v = -U0.std(0)                                   # Shejwalkar std（方差权重）
    else:
        raise ValueError(name)
    nrm = v.norm()
    if nrm < 1e-9:                                       # 极端退化保护
        return -gv
    return v / nrm


# ---------------- 逐客户端 η_max（按方向重算） ----------------
def eta_max_per_client(ck, sigma_rel, v):
    U0 = ck["U"]; d = ck["d"]; un = ck["unit_norm"]
    benign_std = U0.std(dim=0).mean().item()
    sigma = sigma_rel * benign_std
    g = torch.Generator().manual_seed(SEED + 99)
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))

    def flag_prob(i, rho):
        cnt = 0
        for _ in range(6):
            noise = torch.randn(U0.shape, generator=g) * sigma
            U = U0 + noise
            U[i] = U[i] + rho * un * v
            med = coord_median(U)
            s = distance_scores(U, med)
            ben = torch.cat([s[:i], s[i + 1:]])
            thr = ben.mean() + 1.645 * ben.std()
            cnt += 1 if s[i] > thr else 0
        return cnt / 6

    etas = []
    for i in mal:
        lo, hi = 0.05, 12.0
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if flag_prob(i, mid) < 0.5:
                lo = mid
            else:
                hi = mid
        etas.append(0.5 * (lo + hi))
    return etas


# ---------------- 配置与攻击 ----------------
def water_filling(alpha, eta, Gamma):
    lo, hi = 1e-9, max(alpha).item() / 1e-9 + 1.0
    for _ in range(80):
        mu = 0.5 * (lo + hi)
        p = torch.clamp(alpha / (2 * mu), max=eta)
        if (p ** 2).sum().item() > Gamma:
            lo = mu
        else:
            hi = mu
    mu = 0.5 * (lo + hi)
    return torch.clamp(alpha / (2 * mu), max=eta)


def alloc(name, alpha, eta, Gamma, K, sigma_rel):
    if name == "OURS":
        return water_filling(alpha, eta, Gamma)
    if name == "Q25":                       # 全局预算分配，不封顶
        lam = math.sqrt(Gamma / (alpha ** 2).sum().item())
        return lam * alpha
    raise ValueError(name)


def run_attack(ck, shards, test, strat, kappa, sigma_rel, eta, v, seed):
    U0 = ck["U"]; alpha = ck["alpha"]; unit = ck["unit_norm"]
    benign_std = U0.std(dim=0).mean().item()
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    g = torch.Generator().manual_seed(SEED + 1000 + seed)
    sigma = sigma_rel * benign_std
    Gamma = kappa * K_MAL * unit ** 2
    p = alloc(strat, alpha[mal].clone(), eta.clone(), Gamma, K_MAL, sigma_rel)
    gm = get_model(); load_flat(gm, ck["gmodel"])
    mal_flags, round_det = 0, 0
    for r in range(ATTACK_ROUNDS):
        ups = []
        for i in range(N_CLIENTS):
            m = local_train(gm, shards[i], ATTACK_LOCAL_EPOCHS)
            ups.append(flat(m) - flat(gm))
        U = torch.stack(ups)
        for k, i in enumerate(mal):
            U[i] = U[i] + p[k].item() * unit * v
        noise = torch.randn(U.shape, generator=g) * sigma
        Un = U + noise
        med0 = coord_median(Un)
        s = distance_scores(Un, med0)
        ben_idx = [i for i in range(N_CLIENTS) if i not in mal]
        thr = s[ben_idx].mean() + 1.645 * s[ben_idx].std()
        flagged = s > thr
        mal_flags += int(flagged[mal].sum())
        if bool(flagged[mal].any()):
            round_det += 1
        keep = ~flagged
        if keep.sum() < 3:
            keep = s <= s.median()
        agg = coord_median(Un[keep])
        load_flat(gm, flat(gm) + agg)
    return evaluate(gm, test), mal_flags / (K_MAL * ATTACK_ROUNDS), round_det / ATTACK_ROUNDS


def main():
    ck = torch.load(CKPT, weights_only=False)
    name, shards, val, test = load_data()        # 只加载一次（避免每轮重读 CIFAR）
    acc_none_list = [run_attack(ck, shards, test, "OURS", 0.0, 1.0,
                              torch.tensor(eta_max_per_client(ck, 1.0, direction_vector(ck, "grad"))),
                              direction_vector(ck, "grad"), s)[0] for s in range(R2_SEEDS)]
    acc_none = sum(acc_none_list) / len(acc_none_list)
    print("[sens] dataset=%s, 无攻击对照 acc=%.4f" % (name, acc_none), flush=True)

    rows, seed_rows, rep = [], [], ["=" * 100,
                     " 方向敏感性分析（Shejwalkar AGR-tailored vs 本管线 −ĝ）",
                     " dataset=%s, K=%d, %d 攻击轮 × %d 种子, 检出即剔除" % (name, K_MAL, ATTACK_ROUNDS, R2_SEEDS),
                     " 方向∈%s；策略∈%s；σ∈%s；κ∈%s" % (DIRECTION_GRID, STRAT_GRID, SIGMA_GRID, KAPPA_GRID),
                     "=" * 100]
    for direction in DIRECTION_GRID:
        v = direction_vector(ck, direction)
        eta_by_sigma = {s: torch.tensor(eta_max_per_client(ck, s, v)) for s in SIGMA_GRID}
        rep.append("")
        rep.append("-" * 100)
        rep.append(" 方向 = %s" % direction)
        for s in SIGMA_GRID:
            rep.append("   σ=%.2f η_i=%s" % (s, ["%.2f" % e for e in eta_by_sigma[s].tolist()]))
            for kappa in KAPPA_GRID:
                for strat in STRAT_GRID:
                    res = [run_attack(ck, shards, test, strat, kappa, s, eta_by_sigma[s], v, sd) for sd in range(R2_SEEDS)]
                    fa = sum(x[0] for x in res) / len(res)
                    mfr = sum(x[1] for x in res) / len(res) * 100
                    rdr = sum(x[2] for x in res) / len(res) * 100
                    drop = acc_none - fa
                    rows.append((direction, strat, s, kappa, fa, drop, mfr, rdr))
                    for sd in range(R2_SEEDS):
                        seed_rows.append((direction, strat, s, kappa, sd, res[sd][0],
                                          acc_none_list[sd] - res[sd][0],
                                          res[sd][1] * 100, res[sd][2] * 100))
                    rep.append("   %-9s σ=%.2f κ=%.2f  final=%.4f drop=%.4f malflg%%=%.1f rdet%%=%.1f"
                               % (strat, s, kappa, fa, drop, mfr, rdr))
                    print("[sens] %-9s %-8s σ=%.2f κ=%.2f drop=%.4f malflg=%.1f"
                          % (direction, strat, s, kappa, drop, mfr), flush=True)
    fn = os.path.join(OUT, "r2_direction_sensitivity_%s.csv" % DATASET)
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "strategy", "sigma_rel", "kappa", "final_acc", "acc_drop_vs_none",
                    "mal_flag_pct", "round_detect_pct"])
        w.writerows(rows)
    fn2 = os.path.join(OUT, "r2_direction_sensitivity_%s_seeds.csv" % DATASET)
    with open(fn2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "strategy", "sigma_rel", "kappa", "seed", "final_acc",
                    "acc_drop_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(seed_rows)
    rep += ["", "输出: %s" % fn]
    with open(os.path.join(OUT, "r2_direction_sensitivity_%s.txt" % DATASET), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("[sens] done ->", fn, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10"])
    a = ap.parse_args()
    DATASET = a.dataset
    CKPT = _resolve_ckpt(DATASET)
    main()
