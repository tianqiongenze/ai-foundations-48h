# -*- coding: utf-8 -*-
"""生成报告全部图表 -> assets/*.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

C = {"blue": "#2563EB", "cyan": "#0891B2", "green": "#059669", "orange": "#EA580C",
     "purple": "#7C3AED", "red": "#DC2626", "gray": "#6B7280", "yellow": "#CA8A04"}

# ============ 图1: Python 在 AI 技术栈中的位置 ============
fig, ax = plt.subplots(figsize=(9, 5.6))
layers = [
    ("应用层：ChatGPT / 智能客服 / 推荐系统 / 计算机视觉", C["purple"]),
    ("模型层：Scikit-learn / PyTorch / TensorFlow / XGBoost", C["blue"]),
    ("数据层：NumPy / Pandas / Matplotlib", C["green"]),
    ("语言层：Python（胶水语言，贯穿所有层）", C["orange"]),
    ("硬件层：CPU / GPU / TPU 算力", C["gray"]),
]
h = 0.72
for i, (label, color) in enumerate(layers):
    box = FancyBboxPatch((0.5, 0.5 + i * (h + 0.22)), 9, h,
                         boxstyle="round,pad=0.12", fc=color, ec="none", alpha=0.92)
    ax.add_patch(box)
    ax.text(5, 0.5 + i * (h + 0.22) + h / 2, label, ha="center", va="center",
            fontsize=12.5, color="white", fontweight="bold")
ax.annotate("", xy=(10.15, 0.5 + 3 * (h + 0.22) + h / 2), xytext=(10.15, 0.5 + h / 2),
            arrowprops=dict(arrowstyle="-|>", lw=2.5, color=C["red"]))
ax.text(10.45, 0.5 + 1.5 * (h + 0.22), "开发者\n主要工作层", ha="left", va="center",
        fontsize=11, color=C["red"], fontweight="bold")
ax.set_xlim(0, 12); ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("图1  Python 在 AI 技术栈中的位置（自下而上）", fontsize=15, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig("assets/chart_stack.png", bbox_inches="tight")
plt.close(fig)

# ============ 图2: 学习路线图 ============
fig, ax = plt.subplots(figsize=(11, 4.6))
stages = [
    ("阶段一\n语言基础", "变量·类型·字符串\n控制流·函数·类", "第1-2周", C["green"]),
    ("阶段二\n数据三件套", "NumPy·Pandas\nMatplotlib", "第3-4周", C["blue"]),
    ("阶段三\n机器学习入门", "sklearn·线性回归\nKNN·模型评估", "第5-6周", C["orange"]),
    ("阶段四\nAI 实战项目", "端到端数据科学项目\n深度学习拓展", "第7-8周", C["purple"]),
]
for i, (title, content, week, color) in enumerate(stages):
    x = 1 + i * 2.8
    box = FancyBboxPatch((x, 1.0), 2.3, 2.2, boxstyle="round,pad=0.15",
                         fc=color, ec="none", alpha=0.93)
    ax.add_patch(box)
    ax.text(x + 1.15, 2.55, title, ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")
    ax.text(x + 1.15, 1.75, content, ha="center", va="center", fontsize=10.5, color="white")
    ax.text(x + 1.15, 0.62, week, ha="center", va="center", fontsize=11,
            color=color, fontweight="bold")
    if i < 3:
        ax.annotate("", xy=(x + 2.72, 2.1), xytext=(x + 2.42, 2.1),
                    arrowprops=dict(arrowstyle="-|>", lw=3, color=C["gray"]))
ax.set_xlim(0.3, 12.4); ax.set_ylim(0.2, 3.6)
ax.axis("off")
ax.set_title("图2  Python for AI 八周学习路线图", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("assets/chart_roadmap.png", bbox_inches="tight")
plt.close(fig)

# ============ 图3: 数据类型体系图 ============
fig, ax = plt.subplots(figsize=(10, 5.6))
root = (5, 4.9)
ax.annotate("", xy=(5, 4.35), xytext=root, arrowprops=dict(arrowstyle="-", lw=2, color=C["gray"]))
leaf_pos = [1.4, 3.2, 5, 6.8, 8.6]
leafs = [("数字类型", "int 整数: 42\nfloat 浮点: 3.14\n用途: 损失值·学习率", C["blue"]),
         ("字符串 str", "'hello' / f-string\n用途: Prompt·文本预处理", C["green"]),
         ("布尔 bool", "True / False\n用途: 条件判断·掩码筛选", C["orange"]),
         ("列表 list", "[1, 2, 3] 有序可变\n用途: 数据批量·loss历史", C["purple"]),
         ("字典 dict", "{'k': v} 键值对\n用途: 配置·词频统计", C["red"])]
for x, (t, d, c) in zip(leaf_pos, leafs):
    ax.annotate("", xy=(x, 3.6), xytext=(5, 4.35), arrowprops=dict(arrowstyle="-", lw=1.5, color=C["gray"]))
    box = FancyBboxPatch((x - 0.92, 1.7), 1.84, 1.9, boxstyle="round,pad=0.12",
                         fc=c, ec="none", alpha=0.92)
    ax.add_patch(box)
    ax.text(x, 3.15, t, ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.text(x, 2.35, d, ha="center", va="center", fontsize=9.5, color="white")
box = FancyBboxPatch((3.6, 4.35), 2.8, 1.0, boxstyle="round,pad=0.12",
                     fc="#111827", ec="none", alpha=0.95)
ax.add_patch(box)
ax.text(5, 4.85, "Python 核心数据类型", ha="center", va="center", fontsize=14, color="white", fontweight="bold")
ax.set_xlim(0, 10); ax.set_ylim(1.2, 5.8)
ax.axis("off")
ax.set_title("图3  Python 核心数据类型体系与 AI 应用场景", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("assets/chart_types.png", bbox_inches="tight")
plt.close(fig)

# ============ 图4: 控制流流程图 ============
fig, ax = plt.subplots(figsize=(8.5, 6.2))
def diamond(cx, cy, w, h, text, color=C["blue"]):
    ax.add_patch(plt.Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                 fc=color, ec="none", alpha=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
def rect(cx, cy, w, h, text, color):
    box = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.08", fc=color, ec="none", alpha=0.9)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2, color=C["gray"]))
    if label:
        ax.text((x1+x2)/2 + 0.12, (y1+y2)/2, label, fontsize=10, color=C["red"], fontweight="bold")

arrow(4.5, 5.6, 4.5, 5.0)
diamond(4.5, 4.4, 2.6, 1.1, "if score >= 90 ?", C["blue"])
arrow(5.8, 4.4, 7.6, 4.4, "是")
rect(8.6, 4.4, 1.7, 0.8, "grade = A", C["green"])
arrow(4.5, 3.85, 4.5, 3.3, "否")
diamond(4.5, 2.7, 2.9, 1.1, "elif score >= 60 ?", C["orange"])
arrow(5.95, 2.7, 7.6, 2.7, "是")
rect(8.6, 2.7, 1.7, 0.8, "grade = C", C["yellow"])
arrow(4.5, 2.15, 4.5, 1.6, "否")
rect(4.5, 1.1, 2.2, 0.8, "grade = F", C["red"])
ax.text(4.5, 5.9, "for s in scores: 循环取每个分数", ha="center", fontsize=11.5,
        color="#111827", fontweight="bold")
ax.set_xlim(1, 11); ax.set_ylim(0.4, 6.4)
ax.axis("off")
ax.set_title("图4  条件分支控制流：成绩等级判定流程", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("assets/chart_flow.png", bbox_inches="tight")
plt.close(fig)

# ============ 图5: NumPy 广播示意图 ============
fig, ax = plt.subplots(figsize=(9.5, 4.8))
def draw_matrix(x, y, mat, title, cmap_vals, vmin=0, vmax=3):
    n, m = mat.shape
    for i in range(n):
        for j in range(m):
            v = cmap_vals[i, j]
            color = plt.cm.Blues((v - vmin) / (vmax - vmin) * 0.75 + 0.15)
            ax.add_patch(plt.Rectangle((x + j*0.62, y - i*0.62), 0.58, 0.58, fc=color, ec="white", lw=2))
            ax.text(x + j*0.62 + 0.29, y - i*0.62 + 0.29, str(mat[i, j]),
                    ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(x + m*0.31, y + 0.35, title, ha="center", fontsize=11.5, fontweight="bold")

A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
Bb = np.array([[10, 20], [10, 20]])
R = A + Bb
draw_matrix(0.5, 2.4, A, "矩阵 A (2×2)", A)
ax.text(2.3, 1.7, "+", fontsize=20, ha="center", fontweight="bold")
b2 = b.reshape(1, -1)
draw_matrix(2.7, 2.4, b2, "向量 b (1×2)", b2, 0, 30)
ax.text(4.55, 1.7, "=", fontsize=20, ha="center", fontweight="bold")
draw_matrix(5.0, 2.4, Bb, "b 自动复制扩展为 (2×2)", Bb, 0, 30)
ax.annotate("", xy=(5.0, 1.5), xytext=(3.3, 1.5), arrowprops=dict(arrowstyle="-|>", lw=2, color=C["red"]))
ax.text(4.1, 1.15, "广播 broadcast", fontsize=10.5, color=C["red"], ha="center", fontweight="bold")
draw_matrix(7.9, 2.4, R, "结果 A + b (2×2)", R, 10, 30)
ax.set_xlim(0, 11); ax.set_ylim(0.4, 3.4)
ax.axis("off")
ax.set_title("图5  NumPy 广播机制：不同形状数组间的向量化运算", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("assets/chart_numpy.png", bbox_inches="tight")
plt.close(fig)

# ============ 图6: Pandas 数据清洗流程 ============
fig, ax = plt.subplots(figsize=(11, 3.8))
steps = [("原始数据", "缺失值·异常值\n重复行·类型混乱", C["gray"]),
         ("读取", "read_csv\nread_excel", C["blue"]),
         ("探查", "head / info\ndescribe / isnull", C["cyan"]),
         ("清洗", "fillna 填充\ndropna 剔除\nastype 转型", C["orange"]),
         ("加工", "特征列构造\ngroupby 分组\nmerge 合并", C["green"]),
         ("输出", "to_csv\n交给模型训练", C["purple"])]
for i, (t, d, c) in enumerate(steps):
    x = 0.4 + i * 2.05
    box = FancyBboxPatch((x, 0.8), 1.8, 2.0, boxstyle="round,pad=0.12", fc=c, ec="none", alpha=0.93)
    ax.add_patch(box)
    ax.text(x + 0.9, 2.4, t, ha="center", fontsize=13, color="white", fontweight="bold")
    ax.text(x + 0.9, 1.55, d, ha="center", va="center", fontsize=9.5, color="white")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 1.98, 1.8), xytext=(x + 1.75, 1.8),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=C["gray"]))
ax.set_xlim(0.1, 12.6); ax.set_ylim(0.3, 3.4)
ax.axis("off")
ax.set_title("图6  Pandas 数据处理标准流程（数据清洗占 AI 项目 60%+ 工作量）", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("assets/chart_pandas.png", bbox_inches="tight")
plt.close(fig)

# ============ 图7: Matplotlib 四联图 ============
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
x = np.linspace(0, 10, 100)
ax = axes[0, 0]
ax.plot(x, np.sin(x), lw=2.5, color=C["blue"], label="sin(x)")
ax.plot(x, np.cos(x), lw=2.5, color=C["orange"], label="cos(x)", ls="--")
ax.set_title("折线图：训练曲线趋势", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
models = ["KNN", "逻辑回归", "决策树", "随机森林", "XGBoost"]
accs = [0.93, 0.90, 0.87, 0.95, 0.97]
bars = ax.bar(models, accs, color=[C["cyan"], C["green"], C["yellow"], C["orange"], C["red"]], alpha=0.9)
ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.set_ylim(0.8, 1.0)
ax.set_title("柱状图：模型准确率对比", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelsize=9, rotation=15)

ax = axes[1, 0]
rng = np.random.default_rng(0)
ax.scatter(rng.normal(3, 1, 60), rng.normal(3, 1, 60), s=30, alpha=0.7, color=C["blue"], label="类别A")
ax.scatter(rng.normal(6, 1, 60), rng.normal(6, 1, 60), s=30, alpha=0.7, color=C["red"], label="类别B")
ax.set_title("散点图：样本分布可视化", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]
data = rng.normal(70, 12, 500)
ax.hist(data, bins=25, color=C["purple"], alpha=0.85, edgecolor="white")
ax.axvline(data.mean(), color=C["red"], ls="--", lw=2, label=f"均值={data.mean():.1f}")
ax.set_title("直方图：数据分布与均值", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
fig.suptitle("图7  Matplotlib 四种最常用图表（AI 数据分析标配）", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig("assets/chart_data_4panel.png", bbox_inches="tight")
plt.close(fig)

# ============ 图8: 线性回归训练过程 ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
rng = np.random.default_rng(42)
x = rng.uniform(0, 5, 100)
y = 2 * x + 1 + rng.normal(0, 0.5, 100)
w, b, lr = 0.0, 0.0, 0.05
losses, ws, bs = [], [], []
for epoch in range(500):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss); ws.append(w); bs.append(b)
    grad_w = 2 * np.mean((y_pred - y) * x)
    grad_b = 2 * np.mean(y_pred - y)
    w -= lr * grad_w; b -= lr * grad_b

ax = axes[0]
ax.plot(range(1, 501), losses, lw=2.5, color=C["red"])
ax.set_xlabel("训练轮次 epoch"); ax.set_ylabel("损失 loss (MSE)")
ax.set_title("损失随训练下降（梯度下降）", fontsize=12.5, fontweight="bold")
ax.grid(alpha=0.3)
ax.annotate(f"初始 loss={losses[0]:.1f}", xy=(10, losses[5]), xytext=(60, 12),
            fontsize=10, color=C["gray"], arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.annotate(f"最终 loss={losses[-1]:.3f}", xy=(450, losses[-1]), xytext=(260, 5),
            fontsize=10, color=C["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["green"]))

ax = axes[1]
ax.scatter(x, y, s=22, alpha=0.6, color=C["blue"], label="真实样本")
xs = np.linspace(0, 5, 50)
ax.plot(xs, 0.0 * xs + 0.0, lw=2, color=C["gray"], ls=":", label="初始模型 w=0, b=0")
ax.plot(xs, w * xs + b, lw=2.5, color=C["red"], label=f"训练后 y={w:.2f}x+{b:.2f}")
ax.plot(xs, 2 * xs + 1, lw=1.5, color=C["green"], ls="--", label="真实关系 y=2x+1")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("模型拟合结果 vs 真实关系", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
fig.suptitle("图8  从零实现线性回归：损失下降与拟合效果（报告代码12的真实运行结果）", fontsize=14.5, fontweight="bold")
fig.tight_layout()
fig.savefig("assets/chart_lr.png", bbox_inches="tight")
plt.close(fig)

# ============ 图9: 技能金字塔 ============
fig, ax = plt.subplots(figsize=(9, 5.8))
levels = [("第4层  AI 实战", "端到端项目：数据→特征→模型→评估→部署", C["purple"], 1.2),
          ("第3层  机器学习", "sklearn：回归·分类·聚类·模型评估", C["orange"], 2.2),
          ("第2层  数据三件套", "NumPy 数值计算 · Pandas 数据清洗 · Matplotlib 可视化", C["blue"], 3.2),
          ("第1层  语言基础", "变量·控制流·函数·类·文件操作（本报告重点）", C["green"], 4.2)]
for t, d, c, y in levels:
    wdt = 2.4 + (5 - y) * 1.35
    box = FancyBboxPatch((5 - wdt/2, y - 0.42), wdt, 0.86, boxstyle="round,pad=0.1", fc=c, ec="none", alpha=0.93)
    ax.add_patch(box)
    ax.text(5, y + 0.12, t, ha="center", va="center", fontsize=12.5, color="white", fontweight="bold")
    ax.text(5, y - 0.2, d, ha="center", va="center", fontsize=9.5, color="white")
arrow_y = [1.72, 2.72, 3.72]
for y in arrow_y:
    ax.annotate("", xy=(5, y + 0.06), xytext=(5, y - 0.14), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=C["gray"]))
ax.text(9.6, 4.2, "← 打地基\n最优先", fontsize=11, color=C["green"], fontweight="bold", va="center")
ax.set_xlim(0.5, 11); ax.set_ylim(0.5, 5.0)
ax.axis("off")
ax.set_title("图9  Python for AI 技能金字塔（自下而上逐层进阶）", fontsize=15, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("assets/chart_pyramid.png", bbox_inches="tight")
plt.close(fig)

# ============ 图10: 鸢尾花数据集散点图 ============
from sklearn.datasets import load_iris
iris = load_iris()
fig, ax = plt.subplots(figsize=(8.5, 4.6))
colors = [C["blue"], C["orange"], C["green"]]
names = iris.target_names
for i, (c, name) in enumerate(zip(colors, names)):
    mask = iris.target == i
    ax.scatter(iris.data[mask, 0], iris.data[mask, 1], s=38, alpha=0.8, color=c, label=name)
ax.set_xlabel("花萼长度 (cm)"); ax.set_ylabel("花萼宽度 (cm)")
ax.legend(title="鸢尾花品种", fontsize=10)
ax.grid(alpha=0.3)
ax.set_title("图10  鸢尾花数据集：150个样本·4个特征·3个品种（KNN分类案例）", fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig("assets/chart_iris.png", bbox_inches="tight")
plt.close(fig)

print("charts done")
