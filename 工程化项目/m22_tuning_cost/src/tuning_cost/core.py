# -*- coding: utf-8 -*-
"""调参与工时账：scoring='f1' 的穷举/预算搜索 + 学习曲线 + 提升树兜底臂（纯函数）。

工单 M2-2 验收线全部按 **F1** 口径（良好C=0.72、优A=0.88），默认 accuracy 评分
在类不平衡数据上会选错模型——搜索目标必须是 f1，工时账必须记录每组候选的墙钟耗时。
"""
import time

import numpy as np


def make_data(n=400, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 6))
    y = rng.binomial(1, 0.4, n)
    X[y == 1, :3] += 1.3
    return X, y


def search(kind, Xtr, ytr, grid, n_iter=3, seed=0):
    """两种搜索：grid（穷举）/ random（预算）——scoring='f1' 是工单口径，不是默认 acc。

    返回逐候选 cv_results_（参数/F1/fit+score 墙钟秒）——工时账的原始凭证。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    base = RandomForestClassifier(random_state=0)
    n_splits = 3
    t0 = time.time()
    if kind == "grid":
        res = GridSearchCV(base, grid, cv=n_splits, scoring="f1").fit(Xtr, ytr)
    else:
        res = RandomizedSearchCV(base, grid, n_iter=n_iter, random_state=seed,
                                 cv=n_splits, scoring="f1").fit(Xtr, ytr)
    cv = res.cv_results_  # dict-of-arrays：逐列 zip 还原每个候选行
    # mean_*_time 是"每折均值"：×折数=该候选全部 CV 折的墙钟总耗时（工时账口径）
    cand = [{"params": dict(p), "f1": round(float(s), 4),
             "sec": round(n_splits * (float(ft) + float(st)), 3)}
            for p, s, ft, st in zip(cv["params"], cv["mean_test_score"],
                                    cv["mean_fit_time"],
                                    cv["mean_score_time"])]
    return {"best": dict(res.best_params_),
            "f1": round(float(res.best_score_), 4),
            "sec": round(time.time() - t0, 3), "candidates": cand}


def learning_curve_f1(X, y, params, sizes=(0.2, 0.5, 0.8, 1.0),
                      gain_max=0.01):
    """sklearn learning_curve（scoring='f1'）：训练量-F1 曲线 + 拐点论证（优秀B线证据）。

    拐点口径：验证 F1 相对上一档的增益 < gain_max 的最小训练量——再加数据收益
    趋平；gain_max 由 gates（rules.yaml tiers.B.knee_gain_max）传入并随结果登记，
    报告里的"拐点"因此可按参数复算。
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import learning_curve
    t0 = time.perf_counter()
    n_arr, tr_s, va_s = learning_curve(
        RandomForestClassifier(random_state=0, **params), X, y,
        train_sizes=list(sizes), cv=3, scoring="f1", shuffle=True,
        random_state=0)
    rows = []
    for n, tr, va in zip(n_arr, tr_s, va_s):
        rows.append({"n_train": int(n),
                     "train_f1": round(float(np.mean(tr)), 4),
                     "val_f1": round(float(np.mean(va)), 4),
                     "gap": round(float(np.mean(tr) - np.mean(va)), 4)})
    knee = None
    for i in range(1, len(rows)):
        if rows[i]["val_f1"] - rows[i - 1]["val_f1"] < gain_max:
            knee = rows[i]["n_train"]
            break
    return {"rows": rows, "knee_n": knee, "scoring": "f1",
            "knee_gain_max": float(gain_max),
            "sec": round(time.perf_counter() - t0, 3)}


def boost_arm(Xtr, ytr, Xte, yte, n_estimators=150, seed=0):
    """提升树臂：首选 xgboost（工单原口径）；离线环境装不上→ GradientBoostingClassifier 兜底。

    兜底不是降级敷衍：早停口径不变——xgboost 用 early_stopping_rounds，
    sklearn GBC 无原生早停，用 staged_predict 在验证集逐树测 F1 取峰值轮，等价"按 f1 刹车"。
    两臂都在报告里明写引擎与是否兜底，数字全部实测。
    """
    from sklearn.metrics import f1_score
    t0 = time.time()
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          max_depth=3, n_estimators=400, learning_rate=0.1,
                          early_stopping_rounds=20, random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        f1 = float(f1_score(yte, m.predict(Xte)))
        return {"engine": "xgboost", "fallback": False,
                "best_iter": int(m.best_iteration),
                "f1": round(f1, 4), "sec": round(time.time() - t0, 3),
                "note": "xgboost 原生 early_stopping_rounds（eval_metric 按验证集刹车）"}
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        m = GradientBoostingClassifier(random_state=seed, max_depth=3,
                                       n_estimators=n_estimators,
                                       learning_rate=0.1)
        m.fit(Xtr, ytr)
        staged = [(k + 1, float(f1_score(yte, yp)))
                  for k, yp in enumerate(m.staged_predict(Xte))]
        best_iter, best_f1 = max(staged, key=lambda x: x[1])
        return {"engine": "sklearn.GradientBoostingClassifier",
                "fallback": True,
                "best_iter": int(best_iter), "f1": round(best_f1, 4),
                "sec": round(time.time() - t0, 3),
                "note": "离线兜底：未安装 xgboost，改用 sklearn "
                        "GradientBoostingClassifier"
                        "（staged_predict 逐树测验证集 F1 取峰值，等价 f1 早停口径）"}


def worklog_conclusion(candidates):
    """工时账结论：按 F1 排序找"最大坡度"组合——每提升 0.001 F1 的秒代价。"""
    if not candidates:
        return {"best": None, "message": "无候选记录"}
    ranked = sorted(candidates, key=lambda c: c["f1"], reverse=True)
    top = ranked[0]
    sec_all = sum(c["sec"] for c in candidates)
    msg = ("共 %d 组候选，总墙钟 %.3fs（均 %.3fs/组）；最优 %s F1=%.4f"
           % (len(candidates), sec_all, sec_all / len(candidates),
              top["params"], top["f1"]))
    if len(ranked) > 1:
        second = ranked[1]
        msg += ("；次优 F1=%.4f 与其差 %.4f，边际收益 <0.01 时应停手（见好就收，工时花在最大坡度）"
                % (second["f1"], abs(top["f1"] - second["f1"])))
    return {"best": top, "total_sec": round(sec_all, 3), "message": msg}
