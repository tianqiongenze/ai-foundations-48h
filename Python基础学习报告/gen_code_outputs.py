# -*- coding: utf-8 -*-
"""运行报告中全部代码示例，捕获真实输出 -> outputs.json"""
import io, json, contextlib, sys

SNIPPETS = {}

SNIPPETS["hello_ai"] = '''
# 1. 第一个程序：向 AI 世界问好
print("Hello, AI World!")
print("人生苦短，我用 Python")
'''

SNIPPETS["variables"] = '''
# 2. 变量与四大基本类型
age = 20              # int 整数
score = 92.5          # float 浮点数
name = "小明"          # str 字符串
is_student = True     # bool 布尔值

print(age, type(age))
print(score, type(score))
print(name, type(name))
print(is_student, type(is_student))
'''

SNIPPETS["string_ops"] = '''
# 3. 字符串操作：AI 提示词(Prompt)拼接实战
model = "GPT-4"
task = "情感分类"
prompt = f"请你扮演NLP专家，使用{model}完成{task}任务"
print(prompt)

text = "  Python是AI时代的必修语言  "
print("原始长度:", len(text))
print("去空格:", text.strip())
print("替换:", text.strip().replace("必修", "核心"))
print("分割:", "loss,accuracy,f1".split(","))
'''

SNIPPETS["list_ops"] = '''
# 4. 列表：批量管理数据
losses = [0.9, 0.6, 0.4, 0.3, 0.25]

print("训练损失:", losses)
print("列表长度:", len(losses))
print("最后一个epoch损失:", losses[-1])
print("前3个epoch:", losses[:3])
losses.append(0.22)          # 尾部追加
print("追加后:", losses)
print("最小损失:", min(losses), "平均损失:", sum(losses)/len(losses))
'''

SNIPPETS["dict_count"] = '''
# 5. 字典：AI 文本词频统计实战
text = "AI 改变世界 Python 支撑 AI AI 依赖数据 数据 驱动 AI"
words = text.split()

freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1

for word, count in sorted(freq.items(), key=lambda x: -x[1]):
    print(f"{word}: {count}次")
'''

SNIPPETS["control_flow"] = '''
# 6. 条件与循环：批量评定学生等级
scores = [95, 82, 67, 58, 74]

for s in scores:
    if s >= 90:
        grade = "优秀(A)"
    elif s >= 80:
        grade = "良好(B)"
    elif s >= 60:
        grade = "及格(C)"
    else:
        grade = "不及格(F)"
    print(f"分数 {s} -> 等级 {grade}")
'''

SNIPPETS["comprehension"] = '''
# 7. 列表推导式：一行代码完成特征变换
temps_c = [15.2, 18.6, 21.3, 24.8, 27.1]

temps_f = [c * 9 / 5 + 32 for c in temps_c]        # 摄氏 -> 华氏
high_days = [c for c in temps_c if c > 20]          # 筛选高温日

print("华氏温度:", [round(f, 1) for f in temps_f])
print("高于20度的天数:", high_days)
'''

SNIPPETS["function_demo"] = '''
# 8. 函数：封装可复用的模型评估逻辑
def evaluate(y_true, y_pred, verbose=True):
    """计算准确率"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)
    if verbose:
        print(f"样本数={len(y_true)}, 预测正确={correct}, 准确率={acc:.1%}")
    return acc

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]
acc = evaluate(y_true, y_pred)
print("返回值:", acc)
'''

SNIPPETS["class_demo"] = '''
# 9. 类与对象：一个极简"线性模型"
class LinearModel:
    def __init__(self, w=0.0, b=0.0):   # 构造方法
        self.w = w
        self.b = b

    def predict(self, x):               # 实例方法
        return self.w * x + self.b

model = LinearModel(w=2.0, b=1.0)
print("x=3 时的预测值:", model.predict(3))
print("x=10 时的预测值:", model.predict(10))
'''

SNIPPETS["numpy_demo"] = '''
# 10. NumPy：AI 数值计算的基石
import numpy as np

a = np.array([1, 2, 3, 4])
A = np.array([[1, 2], [3, 4]])
B = np.ones((2, 2))

print("一维数组:", a, " 形状:", a.shape)
print("批量运算 a*2+1:", a * 2 + 1)        # 向量化，无需for循环
print("矩阵 A:")
print(A)
print("矩阵乘法 A@B:")
print(A @ B)
print("均值/最大值:", a.mean(), a.max())
'''

SNIPPETS["pandas_demo"] = '''
# 11. Pandas：结构化数据处理实战
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六"],
    "学习时长(h)": [10, 25, np.nan, 40],
    "测验成绩": [65, 88, 72, 95],
})
print("原始数据(含缺失值):")
print(df)

df["学习时长(h)"] = df["学习时长(h)"].fillna(df["学习时长(h)"].mean())
df["是否达标"] = df["测验成绩"] >= 80
print("\\n填充缺失值并添加达标标签后:")
print(df)
print("\\n按是否达标统计平均成绩:")
print(df.groupby("是否达标")["测验成绩"].mean())
'''

SNIPPETS["linear_regression"] = '''
# 12. 完整案例：用梯度下降训练线性回归 y = 2x + 1（纯 NumPy 实现）
import numpy as np

np.random.seed(42)
x = np.random.uniform(0, 5, 100)             # 随机生成100个样本
y = 2 * x + 1 + np.random.normal(0, 0.5, 100) # 加噪声的真实关系

w, b, lr = 0.0, 0.0, 0.05                    # 初始化参数与学习率
for epoch in range(1, 501):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)        # 均方误差 MSE
    grad_w = 2 * np.mean((y_pred - y) * x)   # 对 w 的梯度
    grad_b = 2 * np.mean(y_pred - y)         # 对 b 的梯度
    w -= lr * grad_w                          # 沿负梯度更新
    b -= lr * grad_b
    if epoch in (1, 100, 300, 500):
        print(f"epoch {epoch:3d} | loss = {loss:.4f} | w = {w:.4f} | b = {b:.4f}")

print(f"\\n训练结束: 学到的模型 y = {w:.3f}x + {b:.3f}（真实关系 y = 2.000x + 1.000）")
'''

SNIPPETS["sklearn_demo"] = '''
# 13. 机器学习初体验：sklearn 三行代码训练鸢尾花分类器
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)                  # 训练
acc = model.score(X_test, y_test)            # 评估
print(f"鸢尾花数据集: 训练样本 {len(X_train)} 条, 测试样本 {len(X_test)} 条")
print(f"KNN(k=3) 测试集准确率: {acc:.1%}")
'''

SNIPPETS["matplotlib_demo"] = '''
# Matplotlib 基础绘图三步：数据 -> 定制 -> 保存
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False   # 解决负号显示为方块的问题

epochs = [1, 2, 3, 4, 5]
train_loss = [0.90, 0.55, 0.35, 0.27, 0.24]
val_loss = [0.95, 0.62, 0.45, 0.38, 0.36]

plt.figure(figsize=(6, 3.5))
plt.plot(epochs, train_loss, marker="o", label="训练损失")
plt.plot(epochs, val_loss, marker="s", label="验证损失")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("模型训练/验证损失曲线")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("assets/mini_loss_curve.png", dpi=120, bbox_inches="tight")
print("图表已保存: mini_loss_curve.png")
print("最终训练损失:", train_loss[-1], "| 验证损失:", val_loss[-1])
'''

results = {}
for sid, code in SNIPPETS.items():
    buf = io.StringIO()
    g = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
        results[sid] = {"code": code.strip("\\n"), "output": buf.getvalue().rstrip("\\n"), "error": None}
        print(f"[OK] {sid}")
    except Exception as e:
        results[sid] = {"code": code.strip("\\n"), "output": buf.getvalue(), "error": f"{type(e).__name__}: {e}"}
        print(f"[FAIL] {sid}: {e}")

with open("outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("done:", len(results), "snippets")
