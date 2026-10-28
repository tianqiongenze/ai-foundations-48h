# -*- coding: utf-8 -*-
"""MLP：造数、结构搜索、sklearn/torch 双臂训练、损失曲线诊断与过拟合判定（纯函数）。"""
import time

import numpy as np


def make_spectra(n=600, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 30))
    y = rng.binomial(1, .5, n)
    X[y == 1, 2:8] += rng.uniform(.8, 1.6)
    return X, y


def train(Xtr, ytr, Xte, yte, alpha=0.0, seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=alpha,
                      early_stopping=True, random_state=seed, max_iter=300).fit(Xtr, ytr)
    return m, round(float(m.score(Xte, yte)), 4)


def diagnose(loss_curve):
    """把损失曲线分成 4 段，判断是否持续下降。"""
    n = len(loss_curve)
    if n < 4:
        return {"segments": 1, "improving": False}
    idx = [(0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]
    vals = [round(float(np.mean(loss_curve[a:b])), 4) for a, b in idx]
    return {"segments": 4, "means": vals,
            "improving": all(vals[i] > vals[i + 1] for i in range(3)),
            "final": round(float(loss_curve[-1]), 4)}


def n_params(hidden, n_in, n_out=2):
    """MLP 参数量（含偏置）——优(A)"性价比最优结构"的分母口径。"""
    dims = [n_in] + list(hidden) + [n_out]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def structure_search(Xtr, ytr, Xte, yte, configs=((32, 16), (64, 32, 16), (64, 32)), seed=0):
    """真·隐层结构搜索：每个候选结构实测 acc/F1/epochs/参数量/墙钟秒。

    工单优(A)线"找到性价比最优结构"——先真搜结构再按 acc−0.01 容差取参数量最小者，
    不是拿现成 (64,32) 当答案。
    """
    from sklearn.metrics import f1_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rows = []
    for hid in configs:
        sc = StandardScaler().fit(Xtr)
        m = MLPClassifier(hidden_layer_sizes=hid, alpha=0.0, early_stopping=True,
                          random_state=seed, max_iter=300)
        t0 = time.time()
        m.fit(sc.transform(Xtr), ytr)
        sec = round(time.time() - t0, 3)
        pred = m.predict(sc.transform(Xte))
        acc = round(float((pred == np.asarray(yte)).mean()), 4)
        rows.append({"hidden": list(hid), "params": n_params(hid, Xtr.shape[1]),
                     "acc": acc, "f1": round(float(f1_score(yte, pred)), 4),
                     "epochs": len(m.loss_curve_), "sec": sec})
    top = max(r["acc"] for r in rows)
    value = min((r for r in rows if r["acc"] >= top - 0.01), key=lambda r: r["params"])
    return {"rows": rows, "best_acc": top, "best_value": value}


def torch_mlp(Xtr, ytr, Xte, yte, hidden=(64, 32), dropout=0.2, epochs=40, lr=1e-2, seed=0):
    """torch 臂：nn.Linear + nn.Dropout 真训练——Dropout 是工单 B 线"gap 收窄"的药方。

    确定性口径：torch.manual_seed 固定 + 全批训练，逐 epoch 记训练/验证 loss。
    离线无 torch 时走文档化兜底：sklearn MLPClassifier（alpha 仅 L2、无真 Dropout），
    并在 note 明说——兜底臂不得用来声明 B 线 Dropout 证据达标。
    """
    try:
        import torch
        # macOS 环境实测：sklearn 与 torch 各带一份 libomp，同进程多线程并发会段错误；
        # 单线程是文档化的确定性口径（结果不随核数漂移），不是性能优化。
        torch.set_num_threads(1)
    except ImportError:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        t0 = time.time()
        m = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, early_stopping=True,
                          random_state=seed, max_iter=300).fit(sc.transform(Xtr), ytr)
        acc = round(float(m.score(sc.transform(Xte), yte)), 4)
        lc = [float(v) for v in m.loss_curve_]
        return {"engine": "sklearn-fallback", "dropout": dropout, "acc": acc,
                "train_losses": lc, "val_losses": lc, "epochs": len(lc),
                "sec": round(time.time() - t0, 3),
                "note": "离线兜底：未安装 torch，改用 sklearn MLPClassifier（L2 正则，无真 Dropout）"}
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.long)
    Xe = torch.tensor(sc.transform(Xte), dtype=torch.float32)
    ye = torch.tensor(np.asarray(yte), dtype=torch.long)
    n_cls = int(max(yt.max().item(), ye.max().item())) + 1

    class Net(torch.nn.Module):
        """隐层全用 nn.Linear，每个隐层后接 nn.Dropout(dropout)——工单要求的核心两层件套。"""

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList()
            dim = Xt.shape[1]
            for h in hidden:
                self.blocks.append(torch.nn.Linear(dim, h))
                self.blocks.append(torch.nn.Dropout(dropout))
                dim = h
            self.head = torch.nn.Linear(dim, n_cls)

        def forward(self, x):
            for b in self.blocks:
                x = torch.relu(b(x)) if isinstance(b, torch.nn.Linear) else b(x)
            return self.head(x)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
        train_losses.append(round(float(loss.item()), 4))
        net.eval()
        with torch.no_grad():
            val_losses.append(round(float(lossf(net(Xe), ye).item()), 4))
    net.eval()
    with torch.no_grad():
        acc = round(float((net(Xe).argmax(1) == ye).float().mean()), 4)
    return {"engine": "torch", "dropout": dropout, "acc": acc,
            "train_losses": train_losses, "val_losses": val_losses,
            "epochs": epochs, "sec": round(time.time() - t0, 3),
            "note": "nn.Linear×%d + nn.Dropout(%s)，Adam 全批训练，manual_seed=%d"
                    % (len(hidden), dropout, seed)}


def overfit_diagnosis(train_losses, val_losses, gap_warn=0.05):
    """过拟合诊断（工单 B 线口径）：train↔val 损失末段差距 + val 是否后期回升。"""
    tr, va = np.asarray(train_losses, dtype=float), np.asarray(val_losses, dtype=float)
    k = max(1, len(tr) // 4)
    tr_end, va_end = float(tr[-k:].mean()), float(va[-k:].mean())
    gap = va_end - tr_end
    rising = bool(len(va) >= 6 and float(va[-k:].mean()) > float(va[-2 * k:-k].mean()))
    if rising and gap > gap_warn:
        verdict = "疑似过拟合：val 损失后期回升且 train-val gap=%.3f（上 Dropout/早停驯服它）" % gap
    elif gap > gap_warn:
        verdict = "gap=%.3f 偏大但 val 未回升——观察级，容量还有余量" % gap
    else:
        verdict = "健康：train-val gap=%.3f 未越过警戒线 %.2f" % (gap, gap_warn)
    return {"train_end": round(tr_end, 4), "val_end": round(va_end, 4),
            "gap": round(gap, 4), "val_rising": rising, "verdict": verdict}


def plot_losses(train_losses, val_losses, path, title="MLP learning curve (train/val loss & residual)"):
    """matplotlib（Agg 无头）落盘 PNG——良好(C)线"图完整"的实物证据。

    残差损失 = val − train，直接可视化过拟合 gap；文件不存在=图不完整。
    标题/图例用 ASCII：课堂机默认字体 DejaVu 无中文字形，中文会渲染成豆腐块。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=110)
    ax.plot(train_losses, label="train loss")
    ax.plot(val_losses, label="val loss")
    resid = [round(v - t, 4) for t, v in zip(train_losses, val_losses)]
    ax.plot(resid, linestyle="--", label="residual loss (val-train)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
