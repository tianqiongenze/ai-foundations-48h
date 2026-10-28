# -*- coding: utf-8 -*-
"""竞标二选一：图像线（CNN）vs 时序线（LSTM）+ 窗口数据集 + 标签泄漏核查（纯函数）。

审计结论（如实入报）：旧版 make_series 用样本号 i%2 同时决定标签与幅值斜率——
"特征"只是标签的另一种写法（确定性函数），CV≈0.99 是泄漏虚高，不是模型能力。
本版修复为"连续隐退化强度+阈值打标+随机相位噪声"，两类在近阈天然重叠；
旧构造保留在 make_series_leaky 仅用于泄漏演示，禁止进入拟合矩阵。
"""
import time

import numpy as np


def make_images(n=180, seed=7, size=32):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n):
        img = rng.normal(0.5, 0.08, (size, size))
        label = i % 3
        if label == 1:
            r0, c0 = rng.randint(4, size - 4, 2)
            img[r0:r0 + 3, c0:c0 + 3] += 0.5
        if label == 2:
            r0 = rng.randint(6, size - 6)
            img[r0:r0 + 1, 4:size - 4] -= 0.35
        X.append(img)
        y.append(["正常", "气孔", "裂纹"][label])
    return np.array(X), np.array(y)


def img_feats(img):
    """块统计：亮块比例 / 暗线比例 / 最大亮值。"""
    return [float((img > 0.8).mean()), float((img < 0.2).mean()), float(img.max())]


def make_series_leaky(n=180, seed=7):
    """泄漏演示专用（审计留证）：标签 i%2 直接写进幅值斜率——特征=标签的确定性函数。

    此构造的输出**不得**进入任何拟合矩阵；只用于 leakage_audit 量化"泄漏长什么样"。
    """
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n):
        t = np.linspace(0, 2, 64)
        amp = 0.3 + 0.25 * (i % 2) * t / 2
        X.append(amp * np.sin(2 * np.pi * 6 * t) + rng.randn(64) * 0.05)
        y.append(["健康", "退化"][i % 2])
    return np.array(X), np.array(y)


def make_series(n=180, seed=7):
    """时序线（泄漏修复版）：连续隐退化强度 ~U(0.05,0.65)，标签=强度过阈值。

    随机基线/相位/噪声保证近阈样本天然模糊——单特征可分性掉回真实水平，
    CNN/LSTM 臂学到的才是信号规律而不是标签复刻。
    """
    rng = np.random.RandomState(seed)
    X, y = [], []
    for _ in range(n):
        t = np.linspace(0, 2, 64)
        slope = rng.uniform(0.05, 0.65)
        base = rng.uniform(0.25, 0.35)
        phase = rng.uniform(0, 2 * np.pi)
        amp = base + slope * t
        X.append(amp * np.sin(2 * np.pi * 6 * t + phase) + rng.randn(64) * 0.05)
        y.append(["健康", "退化"][int(slope > 0.35)])
    return np.array(X), np.array(y)


def seq_feats(x):
    """时序线：前/后半 RMS 之比（退化斜率指纹）——修复版数据上这是可学信号而非标签复刻。"""
    return [float(np.sqrt(np.mean(x[:32] ** 2)) / (np.sqrt(np.mean(x[32:] ** 2)) + 1e-9))]


def win_feats(w):
    """窗口级特征（传统ML 时序臂用）：短窗前后半 RMS 比 + 窗内均幅。"""
    h = len(w) // 2
    ratio = float(np.sqrt(np.mean(w[:h] ** 2)) / (np.sqrt(np.mean(w[h:] ** 2)) + 1e-9))
    return [ratio, float(np.sqrt(np.mean(w ** 2)))]


def make_windows(X, y, win=16, stride=8):
    """衍生窗口数据集（工单 B 线窗口化口径）：每条序列切重叠窗口，标签随序列隐状态。"""
    xs, ys = [], []
    for seq, lab in zip(X, y):
        for s in range(0, len(seq) - win + 1, stride):
            xs.append(np.asarray(seq)[s:s + win])
            ys.append(lab)
    return np.array(xs), np.array(ys)


def leakage_audit(builder, feat_fn, folds=3, seed=7):
    """量化"特征是不是只是标签的另一种写法"：类间值域重叠度 + 单特征 CV。

    判定口径（写死在代码里、跑出来算数）：overlap≈0 且单特征 CV≈1 → 泄漏实锤；
    修复后的构造必须有可观重叠，指标掉回真实水平才算干净。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    X, y = builder(180, seed=seed)
    y = np.asarray(y)
    F = np.array([feat_fn(x) for x in X], dtype=float).ravel()
    lohi = {a: (float(F[y == a].min()), float(F[y == a].max())) for a in sorted(set(y))}
    classes = sorted(set(y))
    ov = min(lohi[a][1] for a in classes) - max(lohi[a][0] for a in classes)
    rng_ = float(F.max() - F.min())
    overlap_frac = round(max(0.0, ov) / max(rng_, 1e-9), 4)
    single_cv = round(float(cross_val_score(
        RandomForestClassifier(random_state=0, n_estimators=50),
        F.reshape(-1, 1), y, cv=folds).mean()), 4)
    leak = bool(overlap_frac < 0.05 and single_cv > 0.98)
    return {"overlap_frac": overlap_frac, "single_feat_cv": single_cv,
            "class_ranges": {str(a): (round(lohi[a][0], 4), round(lohi[a][1], 4))
                             for a in classes},
            "leak_detected": leak}


def cv(F, y, folds=3):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(RandomForestClassifier(random_state=0), F, y, cv=folds)
    return round(float(s.mean()), 4)


def fit_eval(model, Xtr, ytr, Xte, yte):
    """通用 fit→测试 acc + 墙钟秒——选型矩阵里每个数字都得是真跑出来的。"""
    t0 = time.time()
    model.fit(Xtr, ytr)
    acc = float((model.predict(Xte) == np.asarray(yte)).mean())
    return {"acc": round(acc, 4), "sec": round(time.time() - t0, 3)}


def decision_matrix(arms, weights):
    """选型决策矩阵（工单 A 档）：候选臂 × 权重 × 实测得分 → 加权总分与结论。

    得分全部由实测换算，不引入任何编造数字：
      性能分 = 测试 acc（实测）；
      速度分 = 最快臂训练秒 / 本臂训练秒（实测墙钟秒换算，≤1）；
      环境分 = 按**实测引擎**映射的部署依赖口径：sklearn 无额外框架 1.0、
              torch 需装深度学习框架 0.5、兜底/其他 0.25（口径随矩阵入报告）。
    权重来自 config/rules.yaml 的 tiers.A——改口径只动配置，且和必须为 1
    （tests/test_tiers.py 断言），否则"矩阵"只是把分数揉成玄学。
    """
    if not arms:
        return {"rows": [], "weights": dict(weights),
                "weights_sum": round(sum(weights.values()), 6),
                "top": None, "conclusion": "无可评臂：矩阵空，A 档不达标"}
    best_sec = min(float(a["sec"]) for a in arms)
    rows = []
    for a in arms:
        sec = max(float(a["sec"]), 1e-6)
        s_env = {"sklearn": 1.0, "torch": 0.5}.get(a["engine"], 0.25)
        score = round(weights["acc"] * float(a["acc"])
                      + weights["speed"] * (best_sec / sec)
                      + weights["env"] * s_env, 4)
        rows.append({"line": a["line"], "arm": a["arm"], "engine": a["engine"],
                     "acc": a["acc"], "sec": a["sec"],
                     "s_perf": round(float(a["acc"]), 4),
                     "s_speed": round(best_sec / sec, 4), "s_env": s_env,
                     "score": score})
    rows.sort(key=lambda r: -r["score"])
    top = rows[0]
    runner = rows[1] if len(rows) > 1 else None
    conclusion = ("结论：%s-%s 加权总分 %.4f 居首（性能分 %.4f、速度分 %.4f、"
                  "环境分 %.4f；权重 acc/speed/env=%.1f/%.1f/%.1f，和=%.1f）"
                  % (top["line"], top["arm"], top["score"], top["s_perf"],
                     top["s_speed"], top["s_env"], weights["acc"],
                     weights["speed"], weights["env"],
                     sum(weights.values())))
    if runner is not None:
        conclusion += "；次优 %s-%s（%.4f），差距 %.4f 由实测列可复算" % (
            runner["line"], runner["arm"], runner["score"],
            round(top["score"] - runner["score"], 4))
    return {"rows": rows, "weights": dict(weights),
            "weights_sum": round(sum(weights.values()), 6),
            "top": top, "conclusion": conclusion}


def torch_cnn(Xtr, ytr, Xte, yte, labels, epochs=40, lr=5e-3, seed=0, dropout=0.2):
    """CNN 臂：nn.Conv2d×2 + MaxPool + Dropout——图像线真卷积实测（良好C线 CNN≥75%）。

    离线无 torch → 文档化兜底：sklearn MLPClassifier 吃展平图像，note 明写
    "无真卷积"，兜底数字不得用来声称卷积能力达标。
    """
    try:
        import torch
        # macOS 实测：sklearn 与 torch 各带一份 libomp，同进程并发会段错误（见 m23a 同款）
        torch.set_num_threads(1)
    except ImportError:
        from sklearn.neural_network import MLPClassifier
        t0 = time.time()
        flat_tr = Xtr.reshape(len(Xtr), -1)
        m = MLPClassifier(hidden_layer_sizes=(64,), random_state=seed,
                          max_iter=100).fit(flat_tr, ytr)
        acc = float((m.predict(Xte.reshape(len(Xte), -1)) == np.asarray(yte)).mean())
        return {"engine": "sklearn-fallback", "acc": round(acc, 4),
                "sec": round(time.time() - t0, 3), "epochs": 0,
                "note": "离线兜底：未安装 torch，CNN 臂改用 sklearn MLPClassifier 展平输入"
                        "（无真卷积，此数字不得用于卷积能力达标判定）"}
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(seed)
    idx = {a: i for i, a in enumerate(labels)}
    sc = StandardScaler().fit(Xtr.reshape(len(Xtr), -1))

    def prep(X):
        Z = sc.transform(X.reshape(len(X), -1))
        Z = Z.reshape(len(X), 1, X.shape[1], X.shape[2])
        return torch.tensor(Z, dtype=torch.float32)

    Xt, Xe = prep(Xtr), prep(Xte)
    yt = torch.tensor([idx[a] for a in ytr], dtype=torch.long)
    ye = torch.tensor([idx[a] for a in yte], dtype=torch.long)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.MaxPool2d(4),
        torch.nn.Conv2d(8, 16, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveMaxPool2d(1), torch.nn.Flatten(),
        torch.nn.Dropout(dropout), torch.nn.Linear(16, len(labels)))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(Xe).argmax(1) == ye).float().mean())
    return {"engine": "torch", "acc": round(acc, 4),
            "sec": round(time.time() - t0, 3), "epochs": epochs,
            "note": "nn.Conv2d(1→8)+MaxPool+nn.Conv2d(8→16)+Dropout(%.1f)×%d epochs"
                    % (dropout, epochs)}


def torch_lstm(Xw, yw, Xw_te, yw_te, labels, epochs=60, lr=1e-2, seed=0, hidden=32):
    """LSTM 臂：nn.LSTM(1→32) 吃 (N,16,1) 窗口——时序线长程依赖建模（窗口化=工单口径）。

    离线无 torch → 文档化兜底：sklearn RF 吃窗口统计特征，note 明写"无真 LSTM"。
    """
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        Ftr = np.array([win_feats(w) for w in Xw])
        Fte = np.array([win_feats(w) for w in Xw_te])
        r = fit_eval(RandomForestClassifier(random_state=0), Ftr, yw, Fte, yw_te)
        return {"engine": "sklearn-fallback", "acc": r["acc"], "sec": r["sec"],
                "epochs": 0,
                "note": "离线兜底：未安装 torch，LSTM 臂改用 sklearn RandomForest 窗口统计"
                        "（无真 LSTM，此数字不得用于时序建模能力达标判定）"}
    torch.manual_seed(seed)
    idx = {a: i for i, a in enumerate(labels)}
    Xt = torch.tensor(np.asarray(Xw, dtype=np.float32).reshape(len(Xw), -1, 1))
    Xe = torch.tensor(np.asarray(Xw_te, dtype=np.float32).reshape(len(Xw_te), -1, 1))
    yt = torch.tensor([idx[a] for a in yw], dtype=torch.long)
    ye = torch.tensor([idx[a] for a in yw_te], dtype=torch.long)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, hidden, batch_first=True)
            self.drop = torch.nn.Dropout(0.2)
            self.head = torch.nn.Linear(hidden, len(labels))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :]))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        acc = float((net(Xe).argmax(1) == ye).float().mean())
    return {"engine": "torch", "acc": round(acc, 4),
            "sec": round(time.time() - t0, 3), "epochs": epochs,
            "note": "nn.LSTM(1→%d) 取末步隐态 + Dropout0.2×%d epochs（窗口 win=16/stride=8）"
                    % (hidden, epochs)}
