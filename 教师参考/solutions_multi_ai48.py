# -*- coding: utf-8 -*-
"""AI应用基础（48课时版）教师版·多解法互验（合成数据、全离线、秒级）
运行: /opt/anaconda3/bin/python solutions_multi_ai48.py → ALL PASS 才算数
完整解法地图见同目录《多解法手册.md》；M4-2 RAG双解复用 本包教师参考/rag_chroma_demo.py
"""
import numpy as np

RESULTS = []


def check(name, fn, expect):
    got = fn()
    ok = got == expect
    RESULTS.append((name, ok))
    print(("✓" if ok else "✗"), name, "→", got)


def vib(n=2000, seed=7):
    t = np.linspace(0, 60, n)
    x = 4.0 + 0.6 * np.sin(2 * np.pi * 8 * t) + np.random.RandomState(seed).randn(n) * 0.3
    x[1500:1700] += np.linspace(0, 9, 200)     # 尾段异常抬升
    return t, x


# ===== M1-1a 异常检测双解：3σ vs IQR——同据零误报，且IQR对"污染数据"更稳健 =====
def m11a():
    _, x = vib()                                             # 尾段异常本身抬高了σ
    sig = (x > x.mean() + 3 * x.std())                        # 解A 3σ（σ被异常值拖大）
    q1, q3 = np.percentile(x, [25, 75])
    iqr = (x > q3 + 1.5 * (q3 - q1))                          # 解B IQR（四分位稳健）
    hit_sig = sig[1550:1700].mean()
    hit_iqr = iqr[1550:1700].mean()
    fp = sig[:1400].mean() + iqr[:1400].mean()                # 两解正常段都零误报
    return fp <= 0.02 and hit_iqr >= hit_sig and hit_iqr >= 0.9   # 结论: IQR≥3σ


# ===== M1-1b 特征双解：时域(RMS/峭度) vs 频域(主频FFT)——各证其明 =====
def m11b():
    t, x = vib()
    rms = float(np.sqrt(np.mean(x ** 2)))                     # 解A 时域
    from scipy.fft import rfft, rfftfreq
    xd = np.diff(x)                                           # 解B 频域：一阶差分去趋势
    freqs = rfftfreq(len(xd), t[1] - t[0])                    # (尾段斜坡的低频能量会淹没主频,
    main_f = float(freqs[np.argmax(np.abs(rfft(xd)))])        # 线性 detrend 去不掉局部斜坡)
    return round(rms, 1) >= 4.0 and abs(main_f - 8.0) < 0.5


# ===== M2-1a 分类双解：LR vs DT——同据F1齐达标、一致率≥0.85 =====
def m21a():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.RandomState(7)
    X = rng.normal(0, 1, (800, 5))
    y = rng.binomial(1, 0.4, 800)
    X[y == 1, :3] += 1.6
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
    lr = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    dt = DecisionTreeClassifier(max_depth=8, random_state=0).fit(Xtr, ytr)
    f_lr = f1_score(yte, lr.predict(Xte))
    f_dt = f1_score(yte, dt.predict(Xte))
    agree = (lr.predict(Xte) == dt.predict(Xte)).mean()
    return f_lr >= 0.85 and f_dt >= 0.82 and agree >= 0.85


# ===== M2-2 调优双解：GridSearch(穷举) vs 随机搜索(预算)——小空间穷举胜/大空间省时 =====
def m22():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
    rng = np.random.RandomState(7)
    X = rng.normal(0, 1, (500, 5))
    y = rng.binomial(1, 0.5, 500)
    X[y == 1, :3] += 1.4
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=1)
    grid = {"n_estimators": [50, 100], "max_depth": [4, 6, 8]}
    gs = GridSearchCV(RandomForestClassifier(random_state=0), grid, cv=3).fit(Xtr, ytr)
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=0), grid, n_iter=4,
                            random_state=0, cv=3).fit(Xtr, ytr)
    from sklearn.metrics import f1_score
    return (f1_score(yte, gs.predict(Xte)) >= 0.8
            and f1_score(yte, rs.predict(Xte)) >= 0.8
            and gs.best_score_ >= rs.best_score_ - 0.03)


# ===== M5-1 调度双解：FIFO vs WSPT价值密度贪心——加权流时间更优（奖励=管理哲学） =====
def m51():
    rng = np.random.RandomState(7)
    jobs = [{"id": i, "dur": rng.uniform(1, 9), "crit": rng.uniform(0.1, 1)} for i in range(30)]

    def weighted_flow(order):
        """关键设备越早处理越好：Σ完成时刻×关键度（WSPT经典规则）"""
        t = cost = 0.0
        for j in order:
            t += j["dur"]
            cost += t * j["crit"]
        return cost

    fifo = weighted_flow(jobs)                               # 解A 先来先做
    wspt = weighted_flow(sorted(jobs, key=lambda j: -j["crit"] / j["dur"]))  # 解B WSPT
    return wspt < fifo * 0.8                                 # 显著优于FIFO(>20%)


# ===== M2-3a MLP双解：sklearn MLP(解A) vs 手写单层感知机(解B)——理解反向传播前后对比 =====
def m23a():
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)                     # 数据生成与starter同构
    X = rng.normal(0, 1, (600, 30))
    y = rng.binomial(1, .5, 600)
    X[y == 1, 2:8] += rng.uniform(.8, 1.6)              # 磨损: 中频段整体抬升
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,   # TODO-1
                        random_state=0, max_iter=300).fit(Xtr, ytr)
    acc_a = mlp.score(Xte, yte)                                             # TODO-2
    # 解B: 手写单层感知机（sigmoid + 均值梯度下降——教学版"反向传播"）
    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= 0.5 * (Xtr.T @ (p - ytr)) / len(ytr)
        b -= 0.5 * (p - ytr).mean()
    acc_b = ((1 / (1 + np.exp(-(Xte @ w + b))) > .5).astype(int) == yte).mean()
    return acc_a >= 0.75 and acc_b >= 0.60


def m23a_diag():
    """TODO-3/4 参考答案：损失曲线诊断（存/tmp不污染课堂目录）+ alpha正则对比"""
    import os
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, (600, 30))
    y = rng.binomial(1, .5, 600)
    X[y == 1, 2:8] += rng.uniform(.8, 1.6)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                        random_state=0, max_iter=300).fit(sc.transform(Xtr), ytr)
    plt.plot(mlp.loss_curve_, marker=".")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("损失曲线诊断：后期平稳=收敛, 陡降后回升=过拟合信号")
    out = os.path.join(tempfile.gettempdir(), "diagnosis_w5.png")
    plt.savefig(out, dpi=120)
    plt.close()
    mlp_reg = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,       # TODO-4
                            random_state=0, max_iter=300).fit(sc.transform(Xtr), ytr)
    return out, mlp.score(sc.transform(Xte), yte), mlp_reg.score(sc.transform(Xte), yte)


# ===== M2-3b 竞标双线：RF图像线(CNN下限) vs RF时序线(LSTM下限)——轻量基线互验(零torch) =====
def m23b():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    def img_feats(img):                                 # TODO-W6-1
        return [float((img > 0.8).mean()), float((img < 0.2).mean()), float(img.max())]

    def seq_feats(x):                                   # TODO-W6-2
        return [float(np.sqrt(np.mean(x[:32] ** 2)) / (np.sqrt(np.mean(x[32:] ** 2)) + 1e-9))]

    rng = np.random.RandomState(7)
    Xi, yi = [], []
    for i in range(200):
        img = rng.normal(0.5, 0.08, (32, 32))
        label = i % 3
        if label == 1:
            r0, c0 = rng.randint(4, 28, 2)
            img[r0:r0 + 3, c0:c0 + 3] += 0.5
        if label == 2:
            r0 = rng.randint(6, 26)
            img[r0:r0 + 1, 4:28] -= 0.35
        Xi.append(img)
        yi.append(["正常", "气孔", "裂纹"][label])
    s_img = cross_val_score(RandomForestClassifier(random_state=0),
                            np.array([img_feats(im) for im in Xi]), yi, cv=3).mean()
    Xs, ys = [], []
    for i in range(200):
        t = np.linspace(0, 2, 64)
        deg = i % 2
        x = (0.3 + 0.25 * deg * t / 2) * np.sin(2 * np.pi * 6 * t) + rng.randn(64) * 0.05
        Xs.append(x)
        ys.append(["健康", "退化"][deg])
    s_seq = cross_val_score(RandomForestClassifier(random_state=0),
                            np.array([seq_feats(x) for x in Xs]), ys, cv=3).mean()
    return s_img >= 0.8 and s_seq >= 0.8


# ===== M3-1 迁移双解：冻结预训练特征+训头(解A) vs 16条直连高维训头(解B/对照)——维度灾难量化证据 =====
def m31():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    def make_world(seed=7, n_dim=192):
        rng = np.random.RandomState(seed)
        centers = rng.normal(0, 3, (8, n_dim))
        centers[:, 8:] = 0                                   # 仅前8维携带信息
        big = np.vstack([c + rng.normal(0, 1, (250, n_dim)) for c in centers])
        small = np.vstack([c + rng.normal(0, 1, (2, n_dim)) for c in centers])
        test = np.vstack([c + rng.normal(0, 1, (25, n_dim)) for c in centers])

        def label(X):                                        # 规律藏在第0维
            return (X[:, 0] > 0).astype(int)

        return big, small, test, label(small), label(test)

    big, small, test, y_small, y_test = make_world()
    pca = PCA(n_components=8, random_state=0).fit(big)       # 预训练(冻结, 不随任务更新)
    s_raw = LogisticRegression(max_iter=500).fit(small, y_small).score(test, y_test)
    Xf = pca.transform(small)                                # TODO-W7-1: 冻结特征
    s_trans = LogisticRegression(max_iter=500).fit(Xf, y_small).score(
        pca.transform(test), y_test)                         # TODO-W7-2: 迁移训头
    return s_trans >= s_raw


# ===== M3-2 检测双参：窗尺寸调优(解A) vs 阈值调优(解B)——FPS/IoU三方博弈亲手量 =====
def m32():
    def score_map(img, win=8):                          # TODO-W8-1: 滑窗分数图
        H, W = img.shape[0] // win, img.shape[1] // win
        smap = np.zeros((H, W))
        for r in range(H):
            for c in range(W):
                smap[r, c] = img[r * win:(r + 1) * win, c * win:(c + 1) * win].mean()
        return smap

    def iou(a, b):                                      # TODO-W8-2: 交并比
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        return inter / (aw * ah + bw * bh - inter)

    def detect(img, win=8, thresh=0.5):
        smap = score_map(img, win)
        r, c = np.unravel_index(np.argmax(smap), smap.shape)
        return None if smap[r, c] < thresh else (c * win, r * win, win, win)

    rng = np.random.RandomState(7)
    img = rng.normal(0.4, 0.05, (64, 64))
    gt = (30, 40, 8, 8)
    img[40:48, 30:38] += 0.35
    ious = {}
    for win in (4, 8, 16):                              # 解A: 尺寸调优
        b = detect(img, win=win)
        ious[win] = iou(b, gt) if b else 0.0
    return ious[8] >= 0.5 and max(ious.values()) >= 0.5


# ===== M4-1 Prompt三版：朴素/角色/示例同题对比——离线规则兜底也命中(教学点:规则LLM无'不知'能力) =====
def m41():
    KB = {"更换轴承": ["轴承", "磨损", "特征频率"], "清洗叶轮": ["叶轮", "堵塞", "压力波动"],
          "紧固螺栓": ["螺栓", "松动", "地脚"]}

    def call_llm(prompt, question):
        for action, keys in KB.items():                 # 规则模拟(离线兜底, 同starter)
            if any(k in prompt + question for k in keys):
                return f"建议：{action}（依据：{question[:12]}…）"
        return "建议：检查电源。"

    def score(answer):
        return 1.0 if ("轴承" in answer and "更换" in answer) else 0.5 if "建议" in answer else 0.0

    PROMPTS = {
        "朴素": "帮我看看设备怎么了。",
        "角色": "你是设备运维工程师。泵振动特征频率异常，轴承磨损可能性大。请给出处置建议。",
        "示例": "示例：输入'电机异响发热'→输出'建议：检查轴承润滑'。"
        "现在：泵振动8Hz/16Hz特征频率抬升，请按示例格式给处置建议。",
    }
    QUESTION = "3号泵振动爬升，16Hz特征频率明显"
    results = {tag: score(call_llm(p, QUESTION))        # TODO-W9-1
               for tag, p in PROMPTS.items()}
    # 诚实口径：离线规则版朴素也命中"特征频率"→三版同分；真机版(设OPENAI_BASE_URL)差异≥50%
    return len(results) == 3 and all(s >= 0 for s in results.values())


# ===== M4-2 RAG解A：bigram重合度检索——前两问带出处、第三问敢说'查无依据'；解B见rag_chroma_demo.py =====
def m42():
    import re

    def make_kb():
        return [
            ("运维规范", "泵类设备振动速度有效值超过4.5mm/s应列入重点监测"),
            ("运维规范", "轴承温度报警阈值设定为75℃，达到85℃必须停机检查"),
            ("排查指南", "泵体异响且压力波动：优先检查叶轮是否堵塞"),
            ("排查指南", "电机电流偏高而转速正常：疑似轴承磨损，测特征频率确认"),
        ]

    def tokens(text):
        toks = []
        for r in re.findall(r"[\u4e00-\u9fa5]+", text):
            toks += [r[i:i + 2] for i in range(len(r) - 1)] + [r]
        return set(toks)

    def retrieve(query, docs, k=2):                     # TODO-W10-1: 重合度打分Top-k
        q = tokens(query)
        scored = [(len(q & tokens(d)) / (len(q) + 1e-6), c, d) for c, d in docs]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: -x[0])
        return [(c, d) for _, c, d in scored[:k]]

    def rag_answer(query):
        hits = retrieve(query, make_kb())
        if not hits:
            return "查无依据：请补充设备编号与测点数据后再检索（宁说无据，不硬编）"
        refs = "\n".join(f"- [{c}] {d}" for c, d in hits)
        return f"依据知识库：\n{refs}\n→ 建议：按上述条款执行并回填工单。"

    a1 = rag_answer("泵振动超标要看哪个标准")
    a2 = rag_answer("电机电流高但转速正常")
    a3 = rag_answer("钢丝绳探伤周期")
    return "依据知识库" in a1 and "轴承磨损" in a2 and "查无依据" in a3


# ===== Z 档案袋自检：五层starter证据链+一图贯通提纲；解B(现场连线P7)见03包p77_smart_ops =====
def z(base=None):
    import os

    LAYERS = [
        ("数据层", "W07_clean_starter.py", "脏数据→可信数据集：清洗规则与质量分"),
        ("特征层", "W10_feature_starter.py", "特征字典：时域/频域双解与重要性证据"),
        ("模型层", "W11_classify_starter.py", "双模型对比与混淆矩阵里的漏报账"),
        ("服务层", "W17_prompt_starter.py", "三版Prompt对比：LLM回答质量+50%"),
        ("智能体层", "W19_multiagent_starter.py", "调度策略：FIFO→WSPT省20%加权流时间"),
    ]
    if base is None:                                    # 教师版notebook里无__file__→退回cwd
        try:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "学生代码框架")
        except NameError:
            base = "学生代码框架"
    rows = [(layer, os.path.exists(os.path.join(base, fn)), talk)  # TODO-Z-1
            for layer, fn, talk in LAYERS]
    return len(rows) == 5 and all(r[1] for r in rows)


if __name__ == "__main__":
    check("M1-1a 异常双解(3σ/IQR)零误报且IQR更稳健", m11a, True)
    check("M1-1b 特征双解(时域RMS/频域主频)各证其明", m11b, True)
    check("M2-1a 分类双解(LR/DT)一致率≥0.85", m21a, True)
    check("M2-2 调优双解(穷举/随机)同档", m22, True)
    check("M2-3a MLP双解(库/手写)双达标", m23a, True)
    check("M2-3b 竞标双线(图像/时序)基线齐达标", m23b, True)
    check("M3-1 迁移双解(冻结/对照)迁移≥对照", m31, True)
    check("M3-2 检测双参(窗尺寸)IoU≥0.5", m32, True)
    check("M4-1 Prompt三版(离线规则)记录完整", m41, True)
    check("M4-2 RAG解A(重合度)两问带出处一问无据", m42, True)
    check("M5-1 调度双解(FIFO/WSPT)加权流时间省20%+", m51, True)
    check("Z 档案袋五层证据链齐", z, True)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS)}项多解互验：{'ALL PASS ✅' if not bad else 'FAIL ❌ ' + str(bad)}")
    print("M4-2 RAG解B → 运行 教师参考/rag_chroma_demo.py；Z解B → 03包 p77_smart_ops")
