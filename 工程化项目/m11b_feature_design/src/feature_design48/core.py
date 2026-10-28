# -*- coding: utf-8 -*-
"""特征设计：13 维时域 + 5 维频域 = 18 个特征，全部登记在特征字典里（纯函数）。

对齐工单 M1-1b 的四档验收：
  D 5 特征表能跑通（`FEAT_NAMES[:5]` 子集照样出指标）；
  C ≥10 特征且字典完整（名称/公式/物理含义/单位/取值范围五栏齐全，见 FEATURE_DICT）；
  B 特征 −30% 性能不降（`ablate()`：按重要性砍掉尾部 30%，CV 指标不塌）；
  A 新特征进 Top3 且有物理解释（`since` 标记 v1.1 新增的频域特征）。

工业口径（工单知识卡）：
  B01 特征=体检指标：模型吃统计量不吃原始波形，时域+频域双视角；
  B02 特征字典=户口本：五列成文，新人不看代码也懂；
  B03 相关性初筛：|r|>0.95 的冗余对只留一个；
  B04 特征重要性：让树模型投票，与物理直觉交叉验证；
  B05 好特征胜过换模型：A/B 特征集对比表最有说服力；
  B06 泄漏预警：只用 t 时刻之前可得的窗口统计量，不许出现事后结论列。
"""
import math

import numpy as np

FS_DEFAULT = 100.0            # 采样率 Hz（400 点 / 4 秒）
BASE_HZ = 8.0                 # 转频（基频）
WEAR_HZ = 16.0                # 二倍转频——轴承磨损的典型指纹
HF_BAND = (12.0, 30.0)        # 高频带（冲击调制产生的边带落在这里）
REDUNDANT_R = 0.95            # 相关性初筛阈值（工单 B03）

# --------------------------------------------------------------- 特征字典
# 五栏：name / formula / meaning / unit / range（工单 B02 的"户口本"格式）
FEATURE_DICT = [
    {"name": "t_mean", "domain": "时域", "formula": "mean(x)",
     "meaning": "波形直流偏置（安装应力/静态载荷水平）", "unit": "-",
     "range": "-0.5~0.5", "since": "v1.0"},
    {"name": "t_std", "domain": "时域", "formula": "std(x)",
     "meaning": "总体波动强度，与振动能量同源", "unit": "-", "range": "0.1~5",
     "since": "v1.0"},
    {"name": "t_rms", "domain": "时域", "formula": "sqrt(mean(x^2))",
     "meaning": "有效值——ISO 振动烈度的标准量", "unit": "mm/s", "range": "0.1~5",
     "since": "v1.0"},
    {"name": "t_ptp", "domain": "时域", "formula": "max(x)-min(x)",
     "meaning": "峰峰值——间隙/撞击类故障敏感", "unit": "-", "range": "0.5~20",
     "since": "v1.0"},
    {"name": "t_max", "domain": "时域", "formula": "max(|x|)",
     "meaning": "峰值——单次冲击的幅度上界", "unit": "-", "range": "0.2~10",
     "since": "v1.0"},
    {"name": "t_absmean", "domain": "时域", "formula": "mean(|x|)",
     "meaning": "整流平均值，正态下 ≈0.798σ", "unit": "-", "range": "0.1~4",
     "since": "v1.0"},
    {"name": "t_kurt", "domain": "时域", "formula": "mean((x-mu)^4)/sigma^4",
     "meaning": "峭度——冲击性故障（点蚀/剥落）第一指标", "unit": "-",
     "range": "3~12", "since": "v1.0"},
    {"name": "t_skew", "domain": "时域", "formula": "mean((x-mu)^3)/sigma^3",
     "meaning": "偏度——波形不对称，提示单向摩擦", "unit": "-", "range": "-1~1",
     "since": "v1.0"},
    {"name": "t_crest", "domain": "时域", "formula": "max(|x|)/rms",
     "meaning": "峰值因子——早期点蚀最灵的时域量", "unit": "-", "range": "2~8",
     "since": "v1.1"},
    {"name": "t_shape", "domain": "时域", "formula": "rms/mean(|x|)",
     "meaning": "波形因子——谐波成分多少", "unit": "-", "range": "1.2~2.5",
     "since": "v1.0"},
    {"name": "t_impulse", "domain": "时域", "formula": "max(|x|)/mean(|x|)",
     "meaning": "脉冲因子——冲击成分占比", "unit": "-", "range": "2.5~15",
     "since": "v1.0"},
    {"name": "t_clearance", "domain": "时域", "formula":
     "max(|x|)/(mean(sqrt(|x|)))^2",
     "meaning": "裕度因子——对极窄冲击比峰值因子更敏感", "unit": "-",
     "range": "3~25", "since": "v1.1"},
    {"name": "t_zcr", "domain": "时域", "formula": "mean(sign(x[1:])!=sign(x[:-1]))",
     "meaning": "过零率——频率成分变密的粗代理", "unit": "次/样本",
     "range": "0~0.5", "since": "v1.0"},
    {"name": "f_a8", "domain": "频域", "formula": "|FFT(x)| at 8Hz",
     "meaning": "基频（转频）幅值——不平衡主指标", "unit": "-", "range": "0~60",
     "since": "v1.0"},
    {"name": "f_a16", "domain": "频域", "formula": "|FFT(x)| at 16Hz",
     "meaning": "二倍转频幅值——轴承磨损/不对中指纹", "unit": "-", "range": "0~60",
     "since": "v1.0"},
    {"name": "f_ratio2x", "domain": "频域", "formula": "|FFT|@16Hz / |FFT|@8Hz",
     "meaning": "二倍频比——去掉整机转速大小后只看'磨损比例'", "unit": "-",
     "range": "0~3", "since": "v1.1"},
    {"name": "f_centroid", "domain": "频域", "formula":
     "sum(f*|X|)/sum(|X|)",
     "meaning": "谱质心——能量重心频率，松动/摩擦上移", "unit": "Hz",
     "range": "0~50", "since": "v1.0"},
    {"name": "f_hf_ratio", "domain": "频域", "formula":
     "sum(|X|^2 in 12~30Hz) / sum(|X|^2)",
     "meaning": "高频带能量占比——磨损冲击调制出的边带能量份额", "unit": "-",
     "range": "0~1", "since": "v1.1"},
]
FEAT_NAMES = [d["name"] for d in FEATURE_DICT]
TIME_FEATS = [d["name"] for d in FEATURE_DICT if d["domain"] == "时域"]
FREQ_FEATS = [d["name"] for d in FEATURE_DICT if d["domain"] == "频域"]
NEW_FEATS = [d["name"] for d in FEATURE_DICT if d["since"] == "v1.1"]


def dict_missing_columns():
    """字典完整性自检：五栏（名称/公式/含义/单位/范围）一个都不能空。"""
    need = ("name", "formula", "meaning", "unit", "range")
    bad = []
    for d in FEATURE_DICT:
        for k in need:
            if not str(d.get(k, "")).strip():
                bad.append("%s.%s" % (d.get("name", "?"), k))
    return bad


# --------------------------------------------------------------- 信号与特征
def make_signal(kind, n=400, seed=7, noise=0.4, fs=FS_DEFAULT):
    """合成 8Hz 转频振动；工况幅值与本底噪声逐窗随机（车间真实情形）。

    关键设计：**磨损只体现在"二倍频/基频之比"上**，而整机振幅随转速与
    负载逐窗变化。于是 t_std / t_rms / t_max 这类"绝对幅值"特征只跟工况
    走、不跟故障走——模型想分开两类就必须学会用比值型特征。
    这正是工单 B01/B05 想让学生踩的坑：特征选错，换模型也白搭。
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n) / fs
    amp = rng.uniform(0.20, 0.60)                    # 逐窗工况（转频幅值）
    floor = noise * rng.uniform(0.6, 1.4)            # 逐窗本底噪声
    x = amp * np.sin(2 * np.pi * BASE_HZ * t) + rng.randn(n) * floor
    # 二倍频/基频 之比：正常 0.00~0.25，磨损 0.45~0.95
    ratio = rng.uniform(0.00, 0.25) if kind == "normal" else rng.uniform(0.45, 0.95)
    if kind == "wear":
        x = x + amp * ratio * np.sin(2 * np.pi * WEAR_HZ * t)
        # 磨损偶发弱冲击（滚道点蚀）——只给时域峭度/裕度因子一点点信息
        hits = rng.choice(n, 4, replace=False)
        x[hits] += amp * rng.uniform(0.3, 0.7, len(hits))
    else:
        x = x + amp * ratio * np.sin(2 * np.pi * WEAR_HZ * t)
    return t, x


def feats_time(x):
    """13 维时域统计量（工单 B01：把一段波形压成'体检指标'）。"""
    x = np.asarray(x, float)
    n = len(x)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    rms = math.sqrt(float(np.mean(x ** 2)))
    abs_x = np.abs(x)
    absmean = float(np.mean(abs_x))
    mx = float(np.max(abs_x))
    root = float(np.mean(np.sqrt(abs_x))) ** 2
    zc = float(np.mean(np.sign(x[1:]) != np.sign(x[:-1])))
    return {
        "t_mean": mu,
        "t_std": sd,
        "t_rms": rms,
        "t_ptp": float(np.max(x) - np.min(x)),
        "t_max": float(np.max(x)),
        "t_absmean": absmean,
        "t_kurt": float(np.mean((x - mu) ** 4)) / (sd ** 4 + 1e-12),
        "t_skew": float(np.mean((x - mu) ** 3)) / (sd ** 3 + 1e-12),
        "t_crest": mx / (rms + 1e-12),
        "t_shape": rms / (absmean + 1e-12),
        "t_impulse": mx / (absmean + 1e-12),
        "t_clearance": mx / (root + 1e-12),
        "t_zcr": zc,
    }


def feats_freq(x, fs=FS_DEFAULT):
    """5 维频域特征：基频/二倍频幅值、二倍频比、谱质心、高频带能量占比。"""
    x = np.asarray(x, float)
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    p = spec ** 2

    def amp(hz):
        return float(spec[np.argmin(np.abs(f - hz))])

    a8, a16 = amp(BASE_HZ), amp(WEAR_HZ)
    centroid = float(np.sum(f * spec) / (np.sum(spec) + 1e-12))
    band = (f >= HF_BAND[0]) & (f <= HF_BAND[1])
    hf_ratio = float(np.sum(p[band]) / (np.sum(p) + 1e-12))
    return {"f_a8": a8, "f_a16": a16, "f_ratio2x": a16 / (a8 + 1e-12),
            "f_centroid": centroid, "f_hf_ratio": hf_ratio}


def extract(x, fs=FS_DEFAULT):
    """一行取全 18 个特征（窗口化流水线的标准环节）。"""
    ft, ff = feats_time(x), feats_freq(x, fs)
    return [ft[k] for k in TIME_FEATS] + [ff[k] for k in FREQ_FEATS]


def build_dataset(n_win=60, seed=7, noise=0.4, fs=FS_DEFAULT):
    """多随机实现 → 稳定重要性；行序 = 类别无关的随机洗牌前次序。"""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for lab, kind in ((0, "normal"), (1, "wear")):
        for _ in range(n_win):
            _, x = make_signal(kind, seed=int(rng.randint(0, 100000)), noise=noise, fs=fs)
            X.append(extract(x, fs))
            y.append(lab)
    return np.array(X, float), np.array(y, int)


def subset(X, names):
    """按特征名取列（A/B 特征集对比与消融都用它）。"""
    cols = [FEAT_NAMES.index(n) for n in names]
    return X[:, cols]


# --------------------------------------------------------------- 评价与筛选
def _n_jobs(n_tasks):
    """臂级并行度：留 2 核给系统，任务少就不开那么多进程（省 worker 启动税）。"""
    from multiprocessing import cpu_count
    return max(1, min(n_tasks, cpu_count() - 2))


def _eval_cols(X, y, names, folds, seeds):
    """并行工作函数：按需取特征列再跑多种子 CV——与串行同一条代码路径。"""
    return evaluate_multi(X if names is None else subset(X, names),
                          y, folds, seeds)


def evaluate(X, y, folds=4, seed=0):
    """交叉验证 acc / macro-F1（小样本随机切一次全靠运气，必须 K 折）。

    单个森林保持单线程：并行单位上移到"实验臂"（ablate / feature_set_ab /
    rank_stable 用 joblib 跨臂并行）。课堂数据每臂只有几百棵树的小森林，
    进程内并行只会反复付池调度税；臂级并行一次开池、多臂复用，墙钟从
    ~6 分钟压进 1 分钟内——sklearn 每棵树的种子由 random_state 按树序推导，
    结果与调度方式无关，所以提速是无损的。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    cvk = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=cvk, scoring="accuracy")
    f1 = cross_val_score(clf, X, y, cv=cvk, scoring="f1_macro")
    return {"acc": round(float(acc.mean()), 4),
            "acc_std": round(float(acc.std()), 4),
            "f1": round(float(f1.mean()), 4),
            "f1_std": round(float(f1.std()), 4), "folds": folds}


def evaluate_multi(X, y, folds=4, seeds=(0, 1, 2)):
    """多种子 CV 取均值——120 样本上一次划分差一个点就是 0.8%，
    不加重复的话"性能降没降"根本没法下结论（消融实验的可信度门槛）。"""
    runs = [evaluate(X, y, folds, s) for s in seeds]
    return {"acc": round(float(np.mean([r["acc"] for r in runs])), 4),
            "f1": round(float(np.mean([r["f1"] for r in runs])), 4),
            "acc_spread": round(float(np.max([r["acc"] for r in runs])
                                      - np.min([r["acc"] for r in runs])), 4),
            "seeds": list(seeds), "folds": folds}


def rank(X, y, seed=0):
    """单次 RandomForest 重要性榜（工单 B04：让数据投票，与物理直觉交叉验证）。

    单线程同 evaluate：种子榜的并行在 rank_stable 的臂（种子）级做，结果不变。
    """
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(X, y)
    order = np.argsort(clf.feature_importances_)[::-1]
    return [(FEAT_NAMES[i], round(float(clf.feature_importances_[i]), 4)) for i in order], clf


def _rank_board(X, y, s):
    """并行工作函数：只回传榜单，不跨进程 pickle 整个森林。"""
    ranked, _clf = rank(X, y, s)
    return ranked


def rank_stable(X, y, seeds=(0, 1, 2)):
    """多种子重要性榜取均值 + 名次波动——榜单自己都要有置信度，
    不然"新特征进 Top3"可能只是随机种子运气（工单 B04 的严谨口径）。"""
    from joblib import Parallel, delayed
    boards = [{n: imp for n, imp in rk} for rk in Parallel(
        n_jobs=_n_jobs(len(seeds)))(
        delayed(_rank_board)(X, y, s) for s in seeds)]
    imp_mean = {n: round(float(np.mean([b[n] for b in boards])), 4) for n in FEAT_NAMES}
    ranked = sorted(imp_mean.items(), key=lambda kv: -kv[1])
    orders = []
    for b in boards:
        o = sorted(b.items(), key=lambda kv: -kv[1])
        orders.append({n: i + 1 for i, (n, _) in enumerate(o)})
    pos = {n: [o[n] for o in orders] for n in FEAT_NAMES}
    return [{"name": n, "importance": v, "domain": _dom(n), "since": _since(n),
             "rank_seed_min": min(pos[n]), "rank_seed_max": max(pos[n])}
            for n, v in ranked]


_DOM = {d["name"]: d["domain"] for d in FEATURE_DICT}
_SINCE = {d["name"]: d["since"] for d in FEATURE_DICT}


def _dom(n):
    return _DOM.get(n, "?")


def _since(n):
    return _SINCE.get(n, "?")


def correlation_screen(X, y=None, threshold=REDUNDANT_R):
    """相关性初筛（工单 B03）：|r|>阈值的冗余对只留榜单里更靠前的那个。"""
    if X.shape[1] < 2:
        return [], list(range(X.shape[1]))
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    pairs = []
    drop = set()
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            r = float(C[i, j])
            if abs(r) > threshold:
                pairs.append({"a": FEAT_NAMES[i], "b": FEAT_NAMES[j],
                              "r": round(r, 4),
                              "keep": FEAT_NAMES[i],
                              "reason": "信息重复，保留先出现的 %s" % FEAT_NAMES[i]})
                drop.add(j)
    keep = [i for i in range(X.shape[1]) if i not in drop]
    return pairs, keep


def ablate(X, y, ranked, cut_frac=0.30, folds=4, seeds=(0, 1, 2),
           n_random=5, max_drop=0.03):
    """消融实验（工单 B05：一次只改一处，是归因的黄金标准）。

    三种砍法各跑一遍，用来区分"砍谁"：
      tail   —— 砍掉重要性榜尾部 cut_frac（应几乎不降，证明榜可信）
      random —— 随机砍同数量，重复 n_random 组取均值（对照）
      head   —— 砍掉榜头部 cut_frac（反向对照，应明显塌，否则榜是假的）
    随机砍单跑一次会和尾部砍打平（120 样本一票之差 0.8%），
    所以这里重复多组 + 多种子 CV，"降没降"才叫有统计意义。
    判定：尾部砍相对全集的准确率下降 ≤ max_drop（默认 3 个百分点）→ B 档通过。

    各臂（全集/随机×n/尾部/头部）用 joblib 臂级并行：每臂的种子都是
    显式常量，并行不改变任何数字——纯粹省墙钟时间。
    """
    from joblib import Parallel, delayed
    names = [n for n, _ in ranked]
    k = int(math.ceil(cut_frac * len(names)))   # 向上取整：保证真的砍够 30%
    tail, head = names[:-k], names[k:]
    # 随机臂：n_random 组不同的随机子集，每组再跑多种子 CV
    rnd_names_sets = []
    for s in range(n_random):
        rng = np.random.RandomState(1000 + s)
        pick = list(rng.permutation(len(names))[: len(names) - k])
        pick.sort()
        rnd_names_sets.append([names[i] for i in pick])
    jobs = [None] + rnd_names_sets + [tail, head]
    runs = Parallel(n_jobs=_n_jobs(len(jobs)))(
        delayed(_eval_cols)(X, y, c, folds, seeds) for c in jobs)
    base = runs[0]
    rnd_runs = []
    for c, r in zip(rnd_names_sets, runs[1:1 + n_random]):
        rr = dict(r)
        rr["set"] = c
        rnd_runs.append(rr)
    tail_m = runs[1 + n_random]
    head_m = runs[2 + n_random]
    rnd = {"acc": round(float(np.mean([r["acc"] for r in rnd_runs])), 4),
           "f1": round(float(np.mean([r["f1"] for r in rnd_runs])), 4),
           "acc_spread": round(float(np.max([r["acc"] for r in rnd_runs])
                                     - np.min([r["acc"] for r in rnd_runs])), 4),
           "n_sets": n_random}
    drop = round(base["acc"] - tail_m["acc"], 4)
    return {
        "n_total": len(names), "n_cut": k, "keep": len(names) - k,
        "cut_frac": cut_frac, "max_drop": max_drop,
        "cut_pct_actual": round(100.0 * k / max(len(names), 1), 2),
        "base": base,
        "tail_set": tail, "tail": tail_m, "tail_drop": drop,
        "tail_drop_pct": round(100.0 * drop / base["acc"], 2),
        "random": rnd, "random_drop": round(base["acc"] - rnd["acc"], 4),
        "head_set": head, "head": head_m,
        "head_drop": round(base["acc"] - head_m["acc"], 4),
        "pass": drop <= max_drop + 1e-9,
        "tail_better_than_random": tail_m["acc"] >= rnd["acc"],
        "head_collapses": (base["acc"] - head_m["acc"]) > max_drop + 1e-9,
    }


def set_tags(topk=8, n_screened=None):
    """A/B 各臂的显示名（数量随特征字典长度自动跟随，避免文档与代码打架）。"""
    tags = {"min5": "D档最小集(前%d特征)" % 5,
            "time": "A·仅时域(%d)" % len(TIME_FEATS),
            "freq": "B·仅频域(%d)" % len(FREQ_FEATS),
            "full": "C·全集(%d)" % len(FEAT_NAMES),
            "topN": "D·Top%d 精选" % topk,
            "top3": "E·Top3 极限集"}
    if n_screened is not None:
        tags["screened"] = "F·相关性初筛后(%d)" % n_screened
    return tags


def feature_set_ab(X, y, topk=8, folds=4, seeds=(0, 1, 2), ranked=None):
    """A/B 特征集对比（工单 B05：同模型只换特征，归因明确）。

    七个臂：D 档最小集 / 仅时域 / 仅频域 / 全集 / Top 精选 / Top3 极限集 /
    相关性初筛后精简集——同模型 RandomForest，只换特征列。
    臂与臂之间用 joblib 并行：每臂独立种子链，结果与串行完全一致。
    """
    from joblib import Parallel, delayed
    if ranked is None:
        ranked, _ = rank(X, y, seeds[0])
    top = [n for n, _ in ranked[:topk]]
    pairs, keep_idx = correlation_screen(X)
    screened = [FEAT_NAMES[i] for i in keep_idx]
    tg = set_tags(topk, len(screened))
    sets = {
        tg["min5"]: FEAT_NAMES[:5],
        tg["time"]: TIME_FEATS,
        tg["freq"]: FREQ_FEATS,
        tg["full"]: FEAT_NAMES,
        tg["topN"]: top,
        tg["top3"]: [n for n, _ in ranked[:3]],
        tg["screened"]: screened,
    }
    items = list(sets.items())
    ms = Parallel(n_jobs=_n_jobs(len(items)))(
        delayed(evaluate_multi)(subset(X, names), y, folds, seeds)
        for _tag, names in items)
    out = {}
    for (tag, names), m in zip(items, ms):
        out[tag] = {"id": [k for k, v in tg.items() if v == tag][0],
                    "n_features": len(names), "features": names, **m}
    best = max(out.items(), key=lambda kv: (kv[1]["f1"], kv[1]["acc"]))
    return out, best[0], top


def new_feature_check(ranked, topn=3):
    """A 档判定：v1.1 新增特征是否挤进重要性榜 TopN（工单要求"进 Top3 且有物理解释"）。"""
    top = ranked[:topn]
    hit = [(n, s) for n, s in top if n in NEW_FEATS]
    explained = [d for d in FEATURE_DICT if d["name"] in NEW_FEATS]
    return {"top%d" % topn: top, "new_in_top": hit,
            "n_new_in_top": len(hit), "pass": len(hit) > 0,
            "meanings": {d["name"]: d["meaning"] for d in explained}}
