# -*- coding: utf-8 -*-
"""
选题三 TODO-1 扩规模：真实 FL 管线实验（torch, CPU）
=================================================
支持 MNIST/FashionMNIST 与 CIFAR-10；R2 支持 σ/κ 网格 + 多种子。

三个阶段（CLI --stage）：
  warmup : 干净联邦预训练（加权 FedAvg, n=20 客户端），保存良性更新池、
           各客户端杠杆 α_i（数据量占比）与验证集梯度方向 ĝ。
  r1     : 真实模型上测 η_max(σ)：Fisher 分离度穿越 sep*=2 处即 η_max。
  r2     : 主表：六策略在 σ/κ 网格下攻击，输出最终精度下降、恶意检出率、
           轮次检出率；验证『高损伤 + 低检出』权衡前沿。

运行环境：Python 3.14 + torch 2.14 CPU（机器已装）。
CIFAR-10 数据默认由 torchvision 自动下载至 CIFAR_ROOT（可用环境变量 CIFAR_ROOT 指向本地已有目录）。
产物：results/real/{warmup_<ds>.pt, r1_eta_max_<ds>.csv, r2_main_<ds>.csv, r2_report_<ds>.txt}
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

# ---------------- 配置 ----------------
N_CLIENTS = 20
K_MAL = 6
VAL_SAMPLES = 2000
WARMUP_ROUNDS = 60
ATTACK_ROUNDS = 5
R2_SEEDS = 10
LOCAL_EPOCHS = 2                        # warmup 用
ATTACK_LOCAL_EPOCHS = 1                  # 攻击期本地 epoch（控制单跑耗时，已验证信号清晰）
BATCH = 64
LR = 0.08
SEED = 20261027
DATASET = "fmnist"                      # 运行时由 --dataset 覆盖
SIGMA_GRID = [0.5, 1.0, 2.0]            # σ_rel = σ / 良性每坐标 std
KAPPA_GRID = [0.25, 0.5, 1.0, 2.0]      # Γ = κ · K · unit_norm²
CLIENT_SIZES = [300, 450, 600, 750, 900] * 4   # n=20 异质数据量（真实杠杆来源）

# 本地已有的 CIFAR-10 数据（含 cifar-10-batches-py/）
CIFAR_ROOT = os.environ.get("CIFAR_ROOT", os.path.join(ROOT, "data", "cifar10"))

CKPT = os.path.join(OUT, "warmup_%s.pt" % DATASET)


def ckpt_path():
    """Checkpoint path keyed by dataset and warmup rounds; falls back to the
    legacy ``warmup_<ds>.pt`` for the original 60-round models so existing
    results remain loadable."""
    p = os.path.join(OUT, "warmup_%s_r%d.pt" % (DATASET, WARMUP_ROUNDS))
    if not os.path.exists(p):
        legacy = os.path.join(OUT, "warmup_%s.pt" % DATASET)
        if os.path.exists(legacy):
            return legacy
    return p


def get_model():
    return CIFAR10CNN() if DATASET == "cifar10" else TinyCNN()


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


# ---------------- 数据 ----------------
def load_fmnist():
    try:
        tr = torchvision.datasets.FashionMNIST(os.path.join(ROOT, "data"), train=True, download=True)
        te = torchvision.datasets.FashionMNIST(os.path.join(ROOT, "data"), train=False, download=True)
        name = "FashionMNIST"
    except Exception as e:
        print("[data] FashionMNIST failed (%s), fallback MNIST" % e)
        tr = torchvision.datasets.MNIST(os.path.join(ROOT, "data"), train=True, download=True)
        te = torchvision.datasets.MNIST(os.path.join(ROOT, "data"), train=False, download=True)
        name = "MNIST"
    x = tr.data.unsqueeze(1).float() / 255.0
    y = tr.targets.long()
    xt = te.data.unsqueeze(1).float() / 255.0
    yt = te.targets.long()
    return name, x, y, xt, yt


def load_mnist():
    tr = torchvision.datasets.MNIST(os.path.join(ROOT, "data"), train=True, download=True)
    te = torchvision.datasets.MNIST(os.path.join(ROOT, "data"), train=False, download=True)
    x = tr.data.unsqueeze(1).float() / 255.0
    y = tr.targets.long()
    xt = te.data.unsqueeze(1).float() / 255.0
    yt = te.targets.long()
    return "MNIST", x, y, xt, yt


def load_cifar():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))])
    tr = torchvision.datasets.CIFAR10(CIFAR_ROOT, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10(CIFAR_ROOT, train=False, download=True, transform=tf)
    x = torch.stack([t[0] for t in tr])
    y = torch.tensor([t[1] for t in tr], dtype=torch.long)
    xt = torch.stack([t[0] for t in te])
    yt = torch.tensor([t[1] for t in te], dtype=torch.long)
    return "CIFAR10", x, y, xt, yt


def load_data():
    if DATASET == "cifar10":
        name, x, y, xt, yt = load_cifar()
    elif DATASET == "mnist":
        name, x, y, xt, yt = load_mnist()
    else:
        name, x, y, xt, yt = load_fmnist()
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
def params_of(model):
    return [p for p in model.parameters()]


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


# ---------------- 阶段：warmup ----------------
def stage_warmup():
    t0 = time.time()
    name, shards, val, test = load_data()
    print("[warmup] data:", name)
    sizes = torch.tensor(CLIENT_SIZES, dtype=torch.float)
    wts = sizes / sizes.sum()
    gmodel = get_model()
    acc0 = evaluate(gmodel, test)
    for r in range(WARMUP_ROUNDS):
        ups = []
        for i in range(N_CLIENTS):
            m = local_train(gmodel, shards[i], LOCAL_EPOCHS)
            ups.append(flat(m) - flat(gmodel))
        U = torch.stack(ups)
        agg = (U * wts.unsqueeze(1)).sum(dim=0)      # 加权 FedAvg（warmup 无防御）
        load_flat(gmodel, flat(gmodel) + agg)        # 就地更新（勿重建模型！）
    acc1 = evaluate(gmodel, test)
    alpha = (sizes / sizes.sum()) * N_CLIENTS          # 均值1，保留异质性（FedAvg 杠杆）
    m = gmodel.eval()
    vxl, vyl = next(iter(DataLoader(val, batch_size=VAL_SAMPLES)))
    loss = F.cross_entropy(m(vxl), vyl)
    gparams = torch.autograd.grad(loss, params_of(m))
    gvec = torch.cat([gp.flatten() for gp in gparams]); gvec = gvec / gvec.norm()
    d = U.shape[1]
    unit_norm = U.norm(dim=1).mean().item()
    torch.save({"gmodel": flat(gmodel), "U": U, "alpha": alpha, "gvec": gvec,
                "unit_norm": unit_norm, "d": d, "acc": acc1, "data": name}, ckpt_path())
    print("[warmup] acc %.4f -> %.4f, d=%d, unit_norm=%.3f, alpha[min,max]=[%.2f,%.2f], %.1fs"
          % (acc0, acc1, d, unit_norm, alpha.min().item(), alpha.max().item(), time.time() - t0))


# ---------------- 检测统计 ----------------
def sep_stat(U_noisy, mal_idx, rho_vec, v):
    """Fisher 分离度：恶意与良性到坐标中位数距离的分布间距。"""
    U = U_noisy.clone()
    for k, i in enumerate(mal_idx):
        U[i] = U[i] + rho_vec[k] * v
    med = coord_median(U)
    s = distance_scores(U, med)
    ben = torch.cat([s[:mal_idx[0]], s[mal_idx[-1] + 1:]])
    mal = s[mal_idx[0]:mal_idx[-1] + 1]
    mb, mm = ben.mean(), mal.mean()
    var = ((ben - mb) ** 2).sum().add(((mal - mm) ** 2).sum()).div(len(ben) + len(mal))
    return ((mm - mb) / (var.sqrt() + 1e-12)).item()


def eta_max_at_sigma(ck, sigma_rel, trials=6):
    """给定 σ_rel，扫描 ρ 找 sep 穿越 2.0 的点（ρ 以 unit_norm 为单位）。"""
    U0 = ck["U"]; d = ck["d"]; un = ck["unit_norm"]
    benign_std = U0.std(dim=0).mean().item()
    sigma = sigma_rel * benign_std
    g = torch.Generator().manual_seed(SEED + 7)
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    rho_grid = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0]
    seps = []
    for rho in rho_grid:
        acc = 0.0
        for _ in range(trials):
            noise = torch.randn(U0.shape, generator=g) * sigma
            v = torch.randn(d, generator=g); v = v / v.norm()
            acc += sep_stat(U0 + noise, mal, [rho * un] * K_MAL, v)
        seps.append(acc / trials)
    eta = None
    for k in range(1, len(rho_grid)):
        a, b = seps[k - 1], seps[k]
        if (a <= 2.0 <= b) or (b <= 2.0 <= a):
            r0, r1 = rho_grid[k - 1], rho_grid[k]
            eta = r0 if abs(b - a) < 1e-9 else r0 + (2.0 - a) * (r1 - r0) / (b - a)
            break
    if eta is None:
        eta = rho_grid[-1] if seps[-1] < 2.0 else rho_grid[0]
    return eta, seps, rho_grid


def stage_r1():
    ck = torch.load(ckpt_path(), weights_only=False)
    rows, out = [], []
    for s in SIGMA_GRID:
        t0 = time.time()
        eta, seps, grid = eta_max_at_sigma(ck, s)
        rows.append((s, eta, eta / s))
        out.append(" sigma_rel=%5.2f  eta_max=%.3f unit_norms  eta/sigma=%.3f  (%.1fs)"
                   % (s, eta, eta / s, time.time() - t0))
        print(out[-1])
    with open(os.path.join(OUT, "r1_eta_max_%s.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma_rel", "eta_max_unitnorms", "eta_over_sigma"]); w.writerows(rows)
    ratios = [r[2] for r in rows]
    ok = max(ratios) / max(min(ratios), 1e-9) < 2.0
    with open(os.path.join(OUT, "r1_report_%s.txt" % DATASET), "w", encoding="utf-8") as f:
        f.write("R1 真实管线 η_max(σ)（T1 验证, %s）\n" % DATASET + "\n".join(out) +
                "\n eta/σ 近似常数（比值极差 %.2f）→ η_max ∝ σ 在真实模型上 %s"
                % (max(ratios) / min(ratios), "成立" if ok else "不成立（两段式精化生效）"))


# ---------------- 阶段：r2 主表（网格） ----------------
def eta_max_per_client(ck, sigma_rel):
    """部署判据版探测：逐客户端二分测 η_i（flag-prob 穿越 0.5 的 ρ）。
    方向 = -ĝ（与部署攻击一致）。ρ 以 unit_norm 为单位。"""
    U0 = ck["U"]; d = ck["d"]; gv = ck["gvec"]; un = ck["unit_norm"]
    benign_std = U0.std(dim=0).mean().item()
    sigma = sigma_rel * benign_std
    g = torch.Generator().manual_seed(SEED + 99)
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    v = -gv

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
    if name == "S21":                       # 无 DP 意识：容量按 σ_ref=1 标定（≈ eta/σ_rel）
        return water_filling(alpha, eta / max(sigma_rel, 1.0), Gamma)
    if name == "W25":                       # 噪声自适应全局缩放，按典型 stealthy 容量封顶（数据集无关）
        eta_med = eta.median().item()
        cap = max(sigma_rel, 0.5) * eta_med
        p_each = min(cap, math.sqrt(Gamma / K))
        return torch.full((K,), p_each)
    if name == "H25":                       # 噪声掩蔽启发式：p_i ∝ α_i，按容量封顶、不重定向
        lam = math.sqrt(Gamma / (alpha ** 2).sum().item())
        return torch.clamp(lam * alpha, max=eta)
    if name == "Q25":                       # 全局预算分配，不封顶
        lam = math.sqrt(Gamma / (alpha ** 2).sum().item())
        return lam * alpha
    if name == "UNI":                       # 均匀分配，按容量封顶
        p_each = math.sqrt(Gamma / K)
        return torch.clamp(torch.full((K,), p_each), max=eta)
    raise ValueError(name)


def stage_r2():
    ck = torch.load(ckpt_path(), weights_only=False)
    U0 = ck["U"]; alpha = ck["alpha"]; unit = ck["unit_norm"]; d = ck["d"]
    benign_std = U0.std(dim=0).mean().item()
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    name, shards, val, test = load_data()
    gmodel0 = get_model(); load_flat(gmodel0, ck["gmodel"])
    acc_start = evaluate(gmodel0, test)
    print("[r2] dataset=%s, 起始 acc=%.4f, unit_norm=%.3f, d=%d" % (name, acc_start, unit, d))

    # 预计算每个 σ 的逐客户端 η_i（部署判据）
    eta_by_sigma = {s: torch.tensor(eta_max_per_client(ck, s)) for s in SIGMA_GRID}
    for s in SIGMA_GRID:
        print("   σ=%.2f η_i=%s" % (s, ["%.2f" % e for e in eta_by_sigma[s].tolist()]))

    def run_attack(strat, kappa, sigma_rel, eta, seed):
        """返回 (final_acc, mal_flag_rate, round_detect_rate)。检出即剔除；方向=-ĝ。"""
        g = torch.Generator().manual_seed(SEED + 1000 + seed)
        sigma = sigma_rel * benign_std
        Gamma = kappa * K_MAL * unit ** 2
        p = alloc(strat, alpha[mal].clone(), eta.clone(), Gamma, K_MAL, sigma_rel) if strat != "NONE" else None
        gm = get_model(); load_flat(gm, ck["gmodel"])
        mal_flags, round_det = 0, 0
        for r in range(ATTACK_ROUNDS):
            ups = []
            for i in range(N_CLIENTS):
                m = local_train(gm, shards[i], ATTACK_LOCAL_EPOCHS)
                ups.append(flat(m) - flat(gm))
            U = torch.stack(ups)
            if p is not None:
                v = -ck["gvec"]
                for k, i in enumerate(mal):
                    U[i] = U[i] + p[k].item() * unit * v
            noise = torch.randn(U.shape, generator=g) * sigma
            Un = U + noise
            med0 = coord_median(Un)
            s = distance_scores(Un, med0)
            ben_idx = [i for i in range(N_CLIENTS) if i not in mal]
            thr = s[ben_idx].mean() + 1.645 * s[ben_idx].std()   # ≈5% FPR
            flagged = s > thr
            if p is not None:
                mal_flags += int(flagged[mal].sum())
                if bool(flagged[mal].any()):
                    round_det += 1
            keep = ~flagged if p is not None else torch.ones(N_CLIENTS, dtype=torch.bool)
            if keep.sum() < 3:
                keep = s <= s.median()
            agg = coord_median(Un[keep])
            load_flat(gm, flat(gm) + agg)            # 就地更新（勿重建模型！）
        return evaluate(gm, test), mal_flags / (K_MAL * ATTACK_ROUNDS), round_det / ATTACK_ROUNDS

    rows, rep = [], ["=" * 96,
                     " R2 主表（网格）：dataset=%s, K=%d, %d 攻击轮 × %d 种子, 检出即剔除, 方向=-ĝ"
                     % (name, K_MAL, ATTACK_ROUNDS, R2_SEEDS),
                     " 起始 acc=%.4f (warmup %d 轮)。Γ=κ·K·unit²; σ_rel∈%s, κ∈%s"
                     % (acc_start, WARMUP_ROUNDS, SIGMA_GRID, KAPPA_GRID),
                     "=" * 96]
    # 无攻击对照（每 σ 都相同，逐种子记录供 paired 检验）
    acc_none_list = [run_attack("NONE", 0.0, 1.0, eta_by_sigma[SIGMA_GRID[0]], s)[0] for s in range(R2_SEEDS)]
    acc_none = sum(acc_none_list) / len(acc_none_list)
    seed_rows = []  # (strategy, sigma_rel, kappa, seed, final_acc, acc_drop_vs_none, mal_flag_pct, round_detect_pct)
    rep.append(" NONE（无攻击对照）final_acc=%.4f" % acc_none)
    rep.append("")
    header = " %6s σ=%.2f κ=%.2f %10s %10s %12s %12s" % ("strat", 0, 0, "final", "drop", "malflg%", "rdet%")
    for s in SIGMA_GRID:
        eta = eta_by_sigma[s]
        rep.append("-" * 96)
        rep.append(" σ_rel=%.2f  η_i=%s" % (s, ["%.2f" % e for e in eta.tolist()]))
        sub = []
        for kappa in KAPPA_GRID:
            for strat in ("OURS", "S21", "W25", "H25", "Q25", "UNI"):
                res = [run_attack(strat, kappa, s, eta, sd) for sd in range(R2_SEEDS)]
                fa = sum(x[0] for x in res) / len(res)
                mfr = sum(x[1] for x in res) / len(res) * 100
                rdr = sum(x[2] for x in res) / len(res) * 100
                drop = acc_none - fa
                rows.append((strat, s, kappa, fa, drop, mfr, rdr))
                sub.append((strat, kappa, fa, drop, mfr, rdr))
                for sd in range(R2_SEEDS):
                    seed_rows.append((strat, s, kappa, sd, res[sd][0],
                                      acc_none_list[sd] - res[sd][0],
                                      res[sd][1] * 100, res[sd][2] * 100))
        # 该 σ 下：按 drop 排序，标记低 mal_flag 前沿
        sub_sorted = sorted(sub, key=lambda t: -t[3])
        best = sub_sorted[0]
        rep.append(header)
        for t in sub_sorted:
            mark = " <==" if t is best else ""
            rep.append(" %6s σ=%.2f κ=%.2f %10.4f %10.4f %12.1f %12.1f%s"
                       % (t[0], s, t[1], t[2], t[3], t[4], t[5], mark))
        rep.append("")
        print("[r2] σ=%.2f done (best drop=%.4f by %s)" % (s, best[3], best[0]))
    with open(os.path.join(OUT, "r2_main_%s.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "sigma_rel", "kappa", "final_acc", "acc_drop_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(rows)
    with open(os.path.join(OUT, "r2_main_%s_seeds.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "sigma_rel", "kappa", "seed", "final_acc", "acc_drop_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(seed_rows)
    rep += ["", " 读法：目标=『高 acc_drop + 低 mal_flag%』的可行权衡前沿（每行 σ 下 drop 最大的策略标 <==）。",
            "      κ 增大（封顶起作用）：Q25/W25 大杠杆客户端超 η 被剔（火力丢失），OURS 封顶+重定向保火力。"]
    with open(os.path.join(OUT, "r2_report_%s.txt" % DATASET), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("[r2] done -> results/real/r2_main_%s.csv" % DATASET)


# ---------------- 阶段：r2 后门（P1-9: ASR 度量） ----------------
# 动机（评审 D/E/P1-9）：acc_drop 在噪声底上不可分辨（所有分配器 ≤8e-4），
# 后门目标下 1–2% 扰动即可产生可测的 ASR 差异，让『隐身』与『损害』脱钩。
BD_TARGET = 0            # 后门目标类
BD_N = 1000              # 后门评估样本数
BD_TRIGGER = 3           # trigger 边长（右下角方块，三通道同值）

def make_backdoor_set(test):
    """取测试集前 BD_N 张，右下角 BD_TRIGGER×BD_TRIGGER trigger，标签改为 BD_TARGET。"""
    xs, ys, n = [], [], 0
    for xb, _ in DataLoader(test, batch_size=512):
        for i in range(xb.shape[0]):
            x = xb[i].clone()
            x[..., -BD_TRIGGER:, -BD_TRIGGER:] = 1.0
            xs.append(x); ys.append(BD_TARGET)
            n += 1
            if n >= BD_N: break
        if n >= BD_N: break
    return torch.stack(xs), torch.tensor(ys)

def backdoor_direction(gmodel, bx, by):
    """后门注入方向：使 trigger 样本输出 BD_TARGET 的负梯度（参数空间下降方向）。"""
    m = get_model(); load_flat(m, flat(gmodel))
    loss = F.cross_entropy(m(bx), by)
    gs = torch.autograd.grad(loss, params_of(m))
    v = torch.cat([g.flatten() for g in gs])
    return -v / v.norm()

@torch.no_grad()
def eval_bd(model, bx, by):
    m = model.eval()
    return (m(bx).argmax(1) == by).float().mean().item()

def stage_r2_bd():
    ck = torch.load(ckpt_path(), weights_only=False)
    U0 = ck["U"]; alpha = ck["alpha"]; unit = ck["unit_norm"]; d = ck["d"]
    benign_std = U0.std(dim=0).mean().item()
    mal = list(range(N_CLIENTS - K_MAL, N_CLIENTS))
    name, shards, val, test = load_data()
    gmodel0 = get_model(); load_flat(gmodel0, ck["gmodel"])
    acc_start = evaluate(gmodel0, test)
    bx, by = make_backdoor_set(test)
    v_bd = backdoor_direction(gmodel0, bx, by)
    asr_start = eval_bd(gmodel0, bx, by)
    print("[r2bd] dataset=%s, 起始 acc=%.4f, 起始 ASR=%.4f, unit_norm=%.3f, d=%d"
          % (name, acc_start, asr_start, unit, d))

    eta_by_sigma = {s: torch.tensor(eta_max_per_client(ck, s)) for s in SIGMA_GRID}
    for s in SIGMA_GRID:
        print("   σ=%.2f η_i=%s" % (s, ["%.2f" % e for e in eta_by_sigma[s].tolist()]))

    def run_attack_bd(strat, kappa, sigma_rel, eta, seed):
        """返回 (final_acc, asr, mal_flag_rate, round_detect_rate)。方向=v_bd（后门），其余同 run_attack。"""
        g = torch.Generator().manual_seed(SEED + 1000 + seed)
        sigma = sigma_rel * benign_std
        Gamma = kappa * K_MAL * unit ** 2
        p = alloc(strat, alpha[mal].clone(), eta.clone(), Gamma, K_MAL, sigma_rel) if strat != "NONE" else None
        gm = get_model(); load_flat(gm, ck["gmodel"])
        mal_flags, round_det = 0, 0
        for r in range(ATTACK_ROUNDS):
            ups = []
            for i in range(N_CLIENTS):
                m = local_train(gm, shards[i], ATTACK_LOCAL_EPOCHS)
                ups.append(flat(m) - flat(gm))
            U = torch.stack(ups)
            if p is not None:
                for k, i in enumerate(mal):
                    U[i] = U[i] + p[k].item() * unit * v_bd
            noise = torch.randn(U.shape, generator=g) * sigma
            Un = U + noise
            med0 = coord_median(Un)
            s = distance_scores(Un, med0)
            ben_idx = [i for i in range(N_CLIENTS) if i not in mal]
            thr = s[ben_idx].mean() + 1.645 * s[ben_idx].std()   # ≈5% FPR
            flagged = s > thr
            if p is not None:
                mal_flags += int(flagged[mal].sum())
                if bool(flagged[mal].any()):
                    round_det += 1
            keep = ~flagged if p is not None else torch.ones(N_CLIENTS, dtype=torch.bool)
            if keep.sum() < 3:
                keep = s <= s.median()
            agg = coord_median(Un[keep])
            load_flat(gm, flat(gm) + agg)
        return evaluate(gm, test), eval_bd(gm, bx, by), \
               mal_flags / (K_MAL * ATTACK_ROUNDS), round_det / ATTACK_ROUNDS

    rows, seed_rows = [], []
    rep = ["=" * 96,
           " R2-后门网格（P1-9）：dataset=%s, K=%d, %d 攻击轮 × %d 种子, 检出即剔除, 方向=v_bd(target=%d)"
           % (name, K_MAL, ATTACK_ROUNDS, R2_SEEDS, BD_TARGET),
           " 起始 acc=%.4f, 起始 ASR=%.4f (warmup %d 轮)。Γ=κ·K·unit²; σ_rel∈%s, κ∈%s"
           % (acc_start, asr_start, WARMUP_ROUNDS, SIGMA_GRID, KAPPA_GRID),
           "=" * 96]
    acc_none_list, asr_none_list = [], []
    for s_ in range(R2_SEEDS):
        fa, asr_, mf, rd = run_attack_bd("NONE", 0.0, 1.0, eta_by_sigma[SIGMA_GRID[0]], s_)
        acc_none_list.append(fa); asr_none_list.append(asr_)
    acc_none = sum(acc_none_list) / len(acc_none_list)
    asr_none = sum(asr_none_list) / len(asr_none_list)
    rep.append(" NONE（无攻击对照）final_acc=%.4f  ASR=%.4f" % (acc_none, asr_none))
    rep.append("")
    for s in SIGMA_GRID:
        eta = eta_by_sigma[s]
        rep.append("-" * 96)
        rep.append(" σ_rel=%.2f  η_i=%s" % (s, ["%.2f" % e for e in eta.tolist()]))
        for kappa in KAPPA_GRID:
            sub = []
            for strat in ("OURS", "S21", "W25", "H25", "Q25", "UNI"):
                res = [run_attack_bd(strat, kappa, s, eta, sd) for sd in range(R2_SEEDS)]
                fa = sum(x[0] for x in res) / len(res)
                asr_ = sum(x[1] for x in res) / len(res)
                mfr = sum(x[2] for x in res) / len(res) * 100
                rdr = sum(x[3] for x in res) / len(res) * 100
                rows.append((strat, s, kappa, fa, asr_, asr_ - asr_none, mfr, rdr))
                sub.append((strat, kappa, fa, asr_, asr_ - asr_none, mfr, rdr))
                for sd in range(R2_SEEDS):
                    seed_rows.append((strat, s, kappa, sd, res[sd][0], res[sd][1],
                                      res[sd][1] - asr_none_list[sd],
                                      res[sd][2] * 100, res[sd][3] * 100))
            for t in sorted(sub, key=lambda t: -t[4]):
                rep.append(" %6s σ=%.2f κ=%.2f acc=%.4f ASR=%.4f ΔASR=%+.4f malflg%%=%.1f rdet%%=%.1f"
                           % (t[0], s, t[1], t[2], t[3], t[4], t[5], t[6]))
        rep.append("")
        print("[r2bd] σ=%.2f done" % s)
    with open(os.path.join(OUT, "r2_main_bd_%s.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "sigma_rel", "kappa", "final_acc", "asr", "asr_gain_vs_none",
                    "mal_flag_pct", "round_detect_pct"])
        w.writerows(rows)
    with open(os.path.join(OUT, "r2_main_bd_%s_seeds.csv" % DATASET), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "sigma_rel", "kappa", "seed", "final_acc", "asr",
                    "asr_gain_vs_none", "mal_flag_pct", "round_detect_pct"])
        w.writerows(seed_rows)
    rep += ["", " 读法：后门目标下 ASR_gain 是主度量（acc 基本不动 → 『隐身』与『损害』脱钩）。",
            "      预期：分配器间 ASR 排序在绑定区（κ 大）显著可分辨——这是 acc_drop 给不出的信号。"]
    with open(os.path.join(OUT, "r2_report_bd_%s.txt" % DATASET), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("[r2bd] done -> results/real/r2_main_bd_%s.csv" % DATASET)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["warmup", "r1", "r2"])
    ap.add_argument("--task", default="clean", choices=["clean", "backdoor"],
                    help="r2 only: clean = acc_drop 度量（原 R2）；backdoor = ASR 度量（P1-9）")
    ap.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10", "mnist"])
    ap.add_argument("--warmup-rounds", type=int, default=60,
                    help="warmup rounds (default 60); models are stored per-rounds so "
                         "different values do not overwrite each other")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override R2_SEEDS (default 10); smaller = faster pilot run")
    ap.add_argument("--sigmas", type=float, nargs="+", default=None,
                    help="override SIGMA_GRID, e.g. --sigmas 0.5 1.0 2.0")
    ap.add_argument("--kappas", type=float, nargs="+", default=None,
                    help="override KAPPA_GRID, e.g. --kappas 0.5 2.0")
    a = ap.parse_args()
    DATASET = a.dataset
    WARMUP_ROUNDS = a.warmup_rounds
    if a.seeds is not None:
        R2_SEEDS = a.seeds
    if a.sigmas is not None:
        SIGMA_GRID[:] = list(a.sigmas)
    if a.kappas is not None:
        KAPPA_GRID[:] = list(a.kappas)
    CKPT = os.path.join(OUT, "warmup_%s.pt" % DATASET)
    if a.stage == "warmup":
        stage_warmup()
    elif a.stage == "r1":
        stage_r1()
    elif a.task == "backdoor":
        stage_r2_bd()
    else:
        stage_r2()
