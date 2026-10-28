# -*- coding: utf-8 -*-
"""真·迁移学习实验通道：缺陷图像 + 数据增强，实测 冻结/微调 × 增强/不增强 2×2 对比。

主通道是**真·ResNet**：torch.hub 本地缓存若已有 ImageNet resnet18 权重，就直接读
本地文件加载（零网络调用，符合课堂离线硬约束）——ImageNet 预训练特征 → 小样本
冻结/微调，正是工单说的"站在巨人肩膀上"。缓存里没有权重文件（或加载失败）时，
按"文档化确定性兜底"规则退回**小型本地CNN**：在 400 张合成缺陷图上预训练得到
"底座"再迁移。走没走兜底、为什么兜底，都带实测证据原样写进 report.txt，绝不谎称。

对应工单 M3-1 验收线：良好(C)=准确率≥75%+对照表；优秀(B)=≥80%或增强有效论证；
优(A)=≥85%+训练曲线。本节所有数字都由本模块真实训练/评测产生。
"""
import os

import numpy as np

# 模块加载即预载 torch：pytest 收集阶段先于任何 sklearn 调用，
# 规避本机（macOS+anaconda）实测的 libomp 重复加载段错误（详见 pipeline._load_torch_first）。
try:
    import torch as _torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------- 数据世界
def probe_pretrain_weights():
    """环境也是选型维度：真实 ResNet 权重可用性要实测，不空喊"离线"。

    硬约束是零网络——所以"真·ResNet"的唯一离线路径 = torch.hub 本地缓存里
    是否已躺着一份 ImageNet resnet 权重：有就直接读本地文件（不产生任何请求），
    没有才退回确定性兜底（本地合成图预训练小CNN）。这里探测的就是现场证据。
    """
    info = {"torchvision": "MISS（未安装）", "cache_dir": "", "cached": "",
            "note": ""}
    try:
        import torchvision
        info["torchvision"] = getattr(torchvision, "__version__", "已安装")
    except ImportError:
        info["cache_dir"] = "MISS（torch 不可导入，无缓存目录）"
        info["note"] = "torchvision 缺失 → 兜底口径：本地合成图预训练小CNN"
        return info
    try:
        import torch.hub
        d = torch.hub.get_dir()
    except Exception as e:  # 缓存目录都拿不到：如实记录
        info["cache_dir"] = "探测失败（%s）" % e.__class__.__name__
        info["note"] = "torch.hub 缓存目录不可读 → 兜底口径：本地小CNN"
        return info
    ckpt = os.path.join(d, "checkpoints")
    info["cache_dir"] = ckpt
    found = []
    if os.path.isdir(ckpt):
        for name in sorted(os.listdir(ckpt)):
            if "resnet" in name.lower() and name.endswith((".pth", ".pt")):
                found.append(name)
    if found:
        info["cached"] = "、".join(found)
        info["note"] = ("本地缓存已有 %s → 真·ResNet 通道启用"
                        "（直接读本地文件，零网络）" % info["cached"])
    else:
        info["note"] = ("本地缓存 %s 无 resnet 权重文件，离线约束下不可下载"
                        "（零网络，硬约束）→ 兜底口径：本地合成图预训练小CNN"
                        % ckpt)
    return info


def make_image_world(seed=13, n_big=400, n_small=32, n_test=200, size=32):
    """合成两类缺陷图：1=气孔（亮斑）、0=划痕（暗线），背景为产线灰噪。

    样本间抖动位置/尺度/亮度，逼迫模型学"形状"而非"这张图"（工单B04原理）。
    返回 (big, small, test) 三段，各为 (X, y)，X 形状 (n, size, size)，值域约 [0,1]。
    """
    def batch(rng, n):
        half = n // 2
        X = np.zeros((n, size, size), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        idx = np.arange(n)
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            bg = rng.uniform(0.30, 0.60)  # 整图亮度漂移：模拟光照不均
            X[i] += bg
            if k < half:
                # 气孔：半径 2-3 的小亮斑（对比度 0.18-0.32，刻意接近产线噪声水平）
                cy, cx = rng.randint(6, size - 6, 2)
                r = rng.randint(2, 4)
                ys, xs = np.mgrid[0:size, 0:size]
                mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
                X[i, mask] += rng.uniform(0.18, 0.32)
                y[i] = 1
            else:
                # 划痕：长度 7-12 的暗色斜线（45°±20°，对比度 0.22-0.38）
                length = rng.randint(7, 13)
                ang = np.deg2rad(rng.uniform(25, 65))
                cy, cx = rng.randint(8, size - 8, 2)
                for t in np.linspace(-length / 2, length / 2, length * 2):
                    r0 = int(round(cy + t * np.sin(ang)))
                    c0 = int(round(cx + t * np.cos(ang)))
                    if 0 <= r0 < size and 0 <= c0 < size:
                        X[i, r0, c0] -= rng.uniform(0.22, 0.38)
                y[i] = 0
            X[i] += rng.normal(0.0, 0.08, (size, size)).astype(np.float32)
        return X, y

    rng = np.random.RandomState(seed)
    big = batch(rng, n_big)
    small = batch(rng, n_small)
    test = batch(rng, n_test)
    return big, small, test


# ---------------------------------------------------------------- 数据增强
def hflip(img):
    """水平翻转：缺陷不区分左右朝向（焊缝斜线物理上可镜像），安全增强。"""
    return img[:, ::-1].copy()


def rotate(img, deg):
    """小角度旋转（±15°内）：numpy 逆映射+最近邻，离线确定性，无需 scipy/cv2。"""
    h, w = img.shape
    a = np.deg2rad(deg)
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    dy, dx = ys - cy, xs - cx
    sy = np.rint(dx * np.sin(a) + dy * np.cos(a) + cy).astype(int)
    sx = np.rint(dx * np.cos(a) - dy * np.sin(a) + cx).astype(int)
    return img[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)].copy()


def add_noise(img, sigma, rng):
    """高斯噪声：模拟产线光照/传感器抖动。"""
    return img + rng.normal(0.0, sigma, img.shape).astype(np.float32)


AUG_METHODS = ["水平翻转 hflip", "小角度旋转 ±15°（numpy逆映射最近邻）", "高斯噪声 σ=0.03"]


def augment_dataset(X, y, seed=0, repeats=4, sigma=0.03):
    """训练集静态扩增 ×repeats（原图保留），测试集绝不增强——工单B05防泄漏纪律。

    每轮变体：i=1 翻转；i=2 ±15°旋转(随机符号)；i=3 旋转+翻转+噪声叠加。
    种子固定 → 任意机器重跑数字一致（可复现）。
    """
    rng = np.random.RandomState(seed)
    Xs = [X]
    ys = [y]
    for _ in range(repeats - 1):
        Xi = np.empty_like(X)
        for i in range(len(X)):
            img = hflip(X[i]) if rng.rand() < 0.5 else X[i]
            img = rotate(img, rng.choice([-15.0, 15.0]))
            img = add_noise(img, sigma, rng)
            Xi[i] = img
        Xs.append(Xi)
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# ---------------------------------------------------------------- 小型CNN
def make_net(size=32, n_class=2, seed=0):
    """返回 (model, 预训练底座参数名集合)。底座=两层卷积+池化，头=全连接。"""
    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(16, n_class)

        def forward(self, x):
            z = self.backbone(x)
            return self.head(z.flatten(1))

    torch.manual_seed(seed)
    model = SmallCNN()
    return model


# ---------------------------------------------------------------- 真·ResNet 通道
def load_resnet18_offline(probe_info, seed=0, n_class=2):
    """零网络加载 ImageNet resnet18：只读本地缓存文件，绝不触网。

    工单流程图第一步就是"载ResNet权重"——权重在本地缓存时没有理由不用真的。
    fc 替换为 n_class 类新头（先播种再建，保证可复现）。返回 (net, info)；
    任何一步失败返回 (None, 原因)——由调用方落到兜底口径并写进报告。
    """
    import time

    if not probe_info or not probe_info.get("cached"):
        return None, "本地缓存无 resnet 权重文件"
    try:
        import torch
        import torch.nn as nn
        import torchvision
        path = os.path.join(probe_info["cache_dir"],
                            probe_info["cached"].split("、")[0])
        t0 = time.perf_counter()
        net = torchvision.models.resnet18(weights=None)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        load_sec = time.perf_counter() - t0
        torch.manual_seed(seed)
        in_f = int(net.fc.in_features)
        net.fc = nn.Linear(in_f, n_class)
        info = {"weights": os.path.basename(path), "in_features": in_f,
                "load_sec": round(load_sec, 3)}
        return net, info
    except Exception as e:
        return None, "本地权重加载失败（%s）" % e.__class__.__name__


def resnet_feature_fn(net):
    """冻结底座提特征：conv1→avgpool 一次性前向（no_grad），头训练不再过底座。

    为什么提特征而不是每 epoch 全量前向：底座冻结时输出不变，缓存特征
    数学等价但省 ~90% 算力——课堂分钟级预算下的标准做法（工单"冻结省算力"）。
    """
    import torch

    body = torch.nn.Sequential(*list(net.children())[:-1])
    body.eval()

    def extract(X):
        with torch.no_grad():
            z = body(torch.tensor(to_rgb(X), dtype=torch.float32))
        return z.flatten(1).numpy()

    return extract


def _fit_head(head, FX, y, epochs, lr, bs, seed):
    """只训练分类头的曲线（损失, 训练准确率）——冻结格子的训练口径。"""
    import torch
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(FX, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    curves = []
    for _ in range(epochs):
        head.train()
        tot, nb = 0.0, 0
        idx = np.arange(len(y))
        rng.shuffle(idx)
        for s in range(0, len(idx), bs):
            sel = idx[s:s + bs]
            opt.zero_grad()
            loss = lossf(head(Xt[sel]), yt[sel])
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        head.eval()
        with torch.no_grad():
            acc = float((head(Xt).argmax(1) == yt).float().mean())
        curves.append((tot / max(nb, 1), acc))
    return curves, head


def to_rgb(X):
    """灰度 (N,S,S) → 3通道 (N,3,S,S)：ResNet 的输入口径（灰度复制三份）。

    为什么要复制而不是硬塞 1 通道：resnet18 的 conv1 权重就是 3 通道的，
    复制灰度是让"真·ImageNet 底座"合法吃到缺陷图的最小改动，不改动权重。
    """
    return np.repeat(np.asarray(X)[:, None], 3, axis=1).astype(np.float32)


def _batches(X, y, bs, rng, prep=None):
    idx = np.arange(len(y))
    rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        sel = idx[s:s + bs]
        yield (np.asarray(X[sel])[:, None] if prep is None
               else prep(X[sel])), y[sel]


def _fit(model, X, y, epochs, lr, bs, seed, prep=None):
    """通用训练循环：返回逐 epoch 的 (平均损失, 训练准确率) 曲线。

    prep=None 时按小型CNN口径补 1 通道；ResNet 通道传 to_rgb 补 3 通道。
    """
    import torch
    to4 = prep if prep is not None else lambda a: np.asarray(a)[:, None]
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(to4(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    curves = []
    for _ in range(epochs):
        model.train()
        tot = 0.0
        nb = 0
        for xb, yb in _batches(X, y, bs, rng, prep):
            opt.zero_grad()
            out = model(torch.tensor(xb, dtype=torch.float32))
            loss = lossf(out, torch.tensor(yb, dtype=torch.long))
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        model.eval()
        with torch.no_grad():
            acc = float((model(Xt).argmax(1) == yt).float().mean())
        curves.append((tot / max(nb, 1), acc))
    return curves


def evaluate(model, X, y, prep=None):
    import torch
    to4 = prep if prep is not None else lambda a: np.asarray(a)[:, None]
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(to4(X), dtype=torch.float32))
        pred = pred.argmax(1).numpy()
    return float((pred == y).mean()), pred


# ---------------------------------------------------------------- 2×2 实验
def run_experiment(seed=7, raw=None, out_dir=None):
    """先真·ResNet（本地缓存权重），不可用才兜底小CNN；2×2 → 曲线PNG。

    两条通道共享同一数据世界/增强/评测口径，cells 结构一致，档位判定只看实测。
    raw["use_resnet"]=False 可强制走兜底（单测控制运行时长用），报告中会如实写明。
    """
    import copy
    import time
    import torch

    raw = raw or {}
    pre_epochs = int(raw.get("pretrain_epochs", 6))
    head_epochs = int(raw.get("head_epochs", 40))
    ft_epochs = int(raw.get("finetune_epochs", 25))
    repeats = int(raw.get("aug_repeats", 4))

    big, small, test = make_image_world(seed=seed + 6)
    probe = probe_pretrain_weights() if raw.get("use_resnet", True) else None
    net, rn_info = (None, None)
    if probe and probe["cached"]:
        net, rn_info = load_resnet18_offline(probe, seed=seed)
    pre_curve = pre_acc = None
    t0 = time.perf_counter()
    if net is not None:
        mode = "resnet18-imagenet-local-cache"
        extract = resnet_feature_fn(net)
        Ftest = extract(test[0])
        cells = {}
        for aug in (True, False):
            Xtr, ytr = (augment_dataset(small[0], small[1], seed=seed,
                                        repeats=repeats) if aug else small)
            # 冻结底座+训头：特征只提一次，头每格重新初始化（播种可复现）
            head = copy.deepcopy(net.fc)
            curve, head = _fit_head(head, extract(Xtr), ytr, head_epochs,
                                    lr=5e-3, bs=16, seed=seed + int(aug))
            head.eval()
            with torch.no_grad():
                lp = head(torch.tensor(Ftest, dtype=torch.float32))
                pred = lp.argmax(1).numpy()
            cells[(True, aug)] = {"acc": float((pred == test[1]).mean()),
                                  "curve": curve, "n_train": int(len(Xtr))}
        for aug in (True, False):
            model = copy.deepcopy(net)
            torch.manual_seed(seed + int(aug))
            in_f = int(model.fc.in_features)
            model.fc = torch.nn.Linear(in_f, len(np.unique(test[1])))
            Xtr, ytr = (augment_dataset(small[0], small[1], seed=seed,
                                        repeats=repeats) if aug else small)
            curve = _fit(model, Xtr, ytr, ft_epochs, lr=1e-3, bs=16,
                         seed=seed + int(aug), prep=to_rgb)
            acc, _ = evaluate(model, test[0], test[1], prep=to_rgb)
            cells[(False, aug)] = {"acc": acc, "curve": curve,
                                   "n_train": int(len(Xtr))}
    else:
        # 兜底：小型CNN 在合成 400 张上预训练出底座，再做同样的 2×2
        mode = "smallcnn-fallback"
        base = make_net(seed=seed)
        pre_curve = _fit(base, big[0], big[1], pre_epochs, lr=3e-3, bs=32,
                         seed=seed)
        pre_acc, _ = evaluate(base, test[0], test[1])
        base_state = copy.deepcopy(base.state_dict())
        cells = {}
        for frozen in (True, False):
            for aug in (True, False):
                model = make_net(seed=seed)
                model.load_state_dict(copy.deepcopy(base_state))
                Xtr, ytr = (augment_dataset(small[0], small[1], seed=seed,
                                            repeats=repeats) if aug else small)
                if frozen:
                    # 冻结底座：backbone 参数 requires_grad=False，Adam 自动跳过无梯度参数
                    for p in model.backbone.parameters():
                        p.requires_grad = False
                    epochs = head_epochs
                else:
                    epochs = ft_epochs
                curve = _fit(model, Xtr, ytr, epochs,
                             lr=(5e-3 if frozen else 1e-3), bs=16,
                             seed=seed + int(aug))
                acc, _ = evaluate(model, test[0], test[1])
                cells[(frozen, aug)] = {"acc": acc, "curve": curve,
                                        "n_train": int(len(Xtr))}
    wall_sec = round(time.perf_counter() - t0, 1)

    png = None
    if out_dir:
        curve_path = os.path.join(out_dir, "training_curves.png")
        png = plot_curves(cells, pre_curve, curve_path)

    def g(f, a):
        return cells[(f, a)]["acc"]

    gains = {"frozen": g(True, True) - g(True, False),
             "finetune": g(False, True) - g(False, False)}
    best = max(c["acc"] for c in cells.values())
    return {"cells": cells, "pre_acc": pre_acc, "pre_curve": pre_curve,
            "aug_gains": gains, "best_acc": best, "png": png,
            "mode": mode, "probe": probe, "resnet_info": rn_info,
            "wall_sec": wall_sec,
            "sizes": {"big": len(big[0]), "small": len(small[0]),
                      "test": len(test[0])}}


def plot_curves(cells, pre_curve, path):
    """训练损失曲线落盘 PNG（matplotlib Agg 后端）。

    兜底通道画 5 条（含合成预训练）；真·ResNet 通道底座来自 ImageNet，
    本地没有"预训练损失"可画——如实只画 4 格曲线，绝不编一条假曲线。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    if pre_curve:
        losses = [loss for loss, _ in pre_curve]
        ax.plot(losses, "-x", lw=1.2, label="pretrain backbone")
    styles = {(True, True): "-o", (True, False): "--s",
              (False, True): "-^", (False, False): "--d"}
    for (frozen, aug), c in cells.items():
        name = ("%s/%s" % ("frozen" if frozen else "finetune",
                           "aug" if aug else "noaug"))
        losses = [loss for loss, _ in c["curve"]]
        ax.plot(losses, styles[(frozen, aug)], lw=1.2, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (CE)")
    ax.set_title("M3-1 train curves: frozen/finetune x aug/noaug")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
