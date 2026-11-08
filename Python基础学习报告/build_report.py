# -*- coding: utf-8 -*-
"""组装 HTML 学习报告（注入 outputs.json 中的真实运行输出）"""
import json, html, datetime

with open("outputs.json", encoding="utf-8") as f:
    R = json.load(f)

def esc(s):
    return html.escape(s)

def block(sid, lang="python"):
    r = R[sid]
    code = esc(r["code"])
    out = esc(r["output"]) if r["output"] else ""
    out_html = f'<div class="out"><div class="out-tag">▶ 运行输出（真实执行结果）</div><pre>{out}</pre></div>' if out else ""
    return f'<div class="code-wrap"><div class="code-tag">Python 代码 · 已验证可执行</div><pre class="code">{code}</pre>{out_html}</div>'

TODAY = datetime.date.today().strftime("%Y年%m月%d日")

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
       color: #1f2937; font-size: 11.5pt; line-height: 1.75; }
.cover { text-align: center; padding-top: 130px; page-break-after: always; }
.cover .badge { display:inline-block; background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE;
       border-radius:999px; padding:4px 18px; font-size:11pt; margin-bottom:28px; }
.cover h1 { font-size: 30pt; color:#111827; margin-bottom:14px; letter-spacing:1px; }
.cover h2 { font-size: 15pt; color:#6B7280; font-weight:400; margin-bottom:50px; }
.cover .meta { color:#9CA3AF; font-size:11pt; margin-top:90px; }
.cover .kpis { margin: 60px auto 0; display:flex; justify-content:center; gap:22px; }
.cover .kpi { width:150px; border:1.5px solid #E5E7EB; border-radius:12px; padding:16px 10px; }
.cover .kpi .num { font-size:21pt; font-weight:700; color:#2563EB; }
.cover .kpi .lab { font-size:9.5pt; color:#6B7280; }
h2.sec { font-size: 17pt; color:#111827; margin: 26px 0 12px; padding: 8px 0 8px 14px;
         border-left: 5px solid #2563EB; background: #F8FAFC; page-break-after: avoid; }
h3 { font-size: 13.5pt; color:#1E3A8A; margin: 20px 0 8px; page-break-after: avoid; }
p { margin: 8px 0; text-align: justify; }
ul, ol { margin: 6px 0 6px 26px; }
li { margin: 4px 0; }
img.fig { width: 100%; margin: 12px 0 4px; border:1px solid #E5E7EB; border-radius:8px; page-break-inside: avoid; }
.figcap { text-align:center; color:#6B7280; font-size:9.5pt; margin-bottom:14px; }
.code-wrap { margin: 12px 0; page-break-inside: avoid; }
.code-tag { background:#312E81; color:#C7D2FE; font-size:8.5pt; padding:4px 12px;
            border-radius:8px 8px 0 0; display:inline-block; font-weight:600; }
pre.code { background:#0F172A; color:#E2E8F0; padding:14px 16px; border-radius:0 8px 8px 8px;
           font-family:"SF Mono","Menlo","Consolas",monospace; font-size:9.5pt; line-height:1.55;
           white-space:pre-wrap; }
.out { margin-top:-4px; }
.out-tag { background:#D1FAE5; color:#065F46; font-size:8.5pt; padding:3px 12px;
           border-radius:8px 8px 0 0; display:inline-block; font-weight:600; }
pre { white-space: pre-wrap; font-family:"SF Mono","Menlo","Consolas",monospace; }
.out pre { background:#F0FDF4; border:1px solid #BBF7D0; border-top:none; color:#14532D;
           padding:12px 16px; border-radius:0 8px 8px 8px; font-size:9.5pt; line-height:1.55; }
table { width:100%; border-collapse:collapse; margin:12px 0; font-size:10pt; page-break-inside: avoid; }
th { background:#1E3A8A; color:#fff; padding:7px 10px; text-align:left; font-weight:600; }
td { border:1px solid #E5E7EB; padding:6px 10px; vertical-align:top; }
tr:nth-child(even) td { background:#F8FAFC; }
.tip { background:#EFF6FF; border-left:4px solid #2563EB; border-radius:0 8px 8px 0;
       padding:10px 14px; margin:12px 0; font-size:10.5pt; page-break-inside: avoid; }
.tip b { color:#1D4ED8; }
.warn { background:#FFF7ED; border-left:4px solid #EA580C; border-radius:0 8px 8px 0;
       padding:10px 14px; margin:12px 0; font-size:10.5pt; page-break-inside: avoid; }
.warn b { color:#C2410C; }
.toc { page-break-after: always; }
.toc h2 { font-size:18pt; margin-bottom:16px; color:#111827; }
.toc ol { font-size:11.5pt; line-height:2.0; }
.inline-code { background:#F1F5F9; border:1px solid #E2E8F0; border-radius:4px; padding:0 5px;
   font-family:"SF Mono","Menlo",monospace; font-size:10pt; color:#BE185D; }
.pb { page-break-before: always; }
"""

# ============ 目录数据 ============
toc = """<ol>
<li>为什么 AI 时代必须学 Python</li>
<li>环境搭建：Anaconda + Jupyter Notebook</li>
<li>八周学习路线图</li>
<li>Python 语言基础（变量 / 字符串 / 列表字典 / 控制流 / 函数 / 类）</li>
<li>数据三件套：NumPy · Pandas · Matplotlib</li>
<li>机器学习实战：从零实现线性回归 + sklearn 分类</li>
<li>常见报错与调试方法</li>
<li>学习资源与进阶建议</li>
</ol>"""

sections = []
A = sections.append

# ---- 1 ----
A(f"""
<h2 class="sec" id="s1">1. 为什么 AI 时代必须学 Python</h2>
<p>无论你后续要学习<b>机器学习、深度学习、大语言模型（LLM）应用</b>还是<b>数据科学</b>，Python 都是事实上的标准语言。
它语法接近自然语言、上手门槛低，同时拥有全世界最庞大的 AI 生态：训练模型用 PyTorch、处理数据用 Pandas、
调用大模型 API 用官方 SDK——全部是 Python 优先。可以说：<b>不会 Python，就很难亲手做出任何 AI 系统。</b></p>
<img class="fig" src="assets/chart_stack.png">
<div class="figcap">图1  Python 贯穿 AI 技术栈的每一层，是连接数据、模型与应用的"胶水语言"</div>
<img class="fig" src="assets/chart_pyramid.png">
<div class="figcap">图9  技能金字塔：语言基础是最底层的地基，本报告聚焦第 1、2 层</div>
<div class="tip"><b>本报告定位：</b>面向零基础研究生，覆盖"语言基础 → 数据三件套 → 两个完整机器学习案例"。
所有 {len(R)} 段代码均已在 Python 3 环境中<b>真实运行并验证</b>，你可以逐段敲进 Jupyter 复现。</div>
""")

# ---- 2 ----
A(f"""
<h2 class="sec pb" id="s2">2. 环境搭建：Anaconda + Jupyter Notebook</h2>
<p>初学者强烈推荐 <b>Anaconda</b> 发行版：它一次性装好 Python + NumPy/Pandas/Matplotlib/scikit-learn 等常用库，
并自带 <b>Jupyter Notebook</b>——一个"写一段、跑一段、立刻看到结果"的交互式编程环境，是数据科学与 AI 课程的标准工具
（本课程的实训环节也全部基于 Notebook）。</p>
<table>
<tr><th>工具 / 命令</th><th>作用</th><th>说明</th></tr>
<tr><td><span class="inline-code">conda create -n ai python=3.11</span></td><td>创建独立虚拟环境</td><td>每个项目一个环境，避免包版本互相冲突</td></tr>
<tr><td><span class="inline-code">conda activate ai</span></td><td>激活环境</td><td>激活后终端提示符前会出现 (ai)</td></tr>
<tr><td><span class="inline-code">pip install 包名</span></td><td>安装第三方库</td><td>如 <span class="inline-code">pip install numpy pandas scikit-learn</span></td></tr>
<tr><td><span class="inline-code">jupyter notebook</span></td><td>启动 Notebook</td><td>浏览器自动打开，Shift+Enter 运行当前单元格</td></tr>
<tr><td><span class="inline-code">python xxx.py</span></td><td>运行脚本</td><td>适合写成文件的大型程序</td></tr>
</table>
<div class="warn"><b>新手避坑：</b>① 不要用系统自带的 Python 直接装包；② 报 <span class="inline-code">ModuleNotFoundError</span>
说明当前环境没装该库，先 <span class="inline-code">conda activate</span> 再 <span class="inline-code">pip install</span>；
③ 包名与导入名可能不同（如安装 <span class="inline-code">scikit-learn</span>，导入 <span class="inline-code">sklearn</span>）。</div>
""")

# ---- 3 ----
A(f"""
<h2 class="sec pb" id="s3">3. 八周学习路线图</h2>
<p>建议按以下四个阶段推进，每阶段约两周。原则是：<b>先打语言地基，再学数据处理，最后用真实案例把知识串起来</b>。
每学完一个知识点，务必亲手把本报告对应的代码敲一遍并改几个参数观察变化。</p>
<img class="fig" src="assets/chart_roadmap.png">
<div class="figcap">图2  八周四阶段学习路线</div>
""")

# ---- 4 ----
A(f"""
<h2 class="sec pb" id="s4">4. Python 语言基础</h2>

<h3>4.1 变量与基本数据类型</h3>
<p>变量无需声明类型，赋值即创建。AI 编程中最常用的四种基本类型如下：</p>
<img class="fig" src="assets/chart_types.png">
<div class="figcap">图3  核心数据类型及其在 AI 场景中的典型用途</div>
{block("hello_ai")}
{block("variables")}
<div class="tip"><b>要点：</b>Python 用 <b>缩进（4个空格）</b>表示代码块，不用大括号；
<span class="inline-code">#</span> 开头是注释；<span class="inline-code">type()</span> 可随时查看变量类型。</div>

<h3 class="pb">4.2 字符串与 f-string 格式化</h3>
<p>AI 应用中大量文本处理（Prompt 拼接、日志输出、数据清洗）都依赖字符串操作，f-string 是最推荐的格式化方式：</p>
{block("string_ops")}

<h3>4.3 列表：批量数据的容器</h3>
<p>模型训练时的 loss 历史曲线、一批样本的特征值，都用列表存放。切片 <span class="inline-code">lst[a:b]</span>、
负索引 <span class="inline-code">lst[-1]</span> 是必须熟练的技能：</p>
{block("list_ops")}

<h3>4.4 字典：键值对映射</h3>
<p>词频统计、模型配置、API 返回的 JSON 结果，本质都是字典。<span class="inline-code">dict.get(k, 默认值)</span>
可以安全地读取不存在的键：</p>
{block("dict_count")}

<h3 class="pb">4.5 条件与循环</h3>
<p><span class="inline-code">if / elif / else</span> 做分支判断，<span class="inline-code">for ... in ...</span> 遍历序列——
这是写任何自动化逻辑的基本骨架。下图与代码演示"逐个学生成绩评级"的完整流程：</p>
<img class="fig" src="assets/chart_flow.png">
<div class="figcap">图4  成绩评级的多分支控制流</div>
{block("control_flow")}

<h3>4.6 列表推导式：Pythonic 的一行流</h3>
<p>列表推导式把"循环 + 条件 + 变换"压缩成一行，是 Python 最具特色的语法，数据分析中随处可见：</p>
{block("comprehension")}

<h3 class="pb">4.7 函数：把逻辑封装成可复用模块</h3>
<p>函数用 <span class="inline-code">def</span> 定义，支持默认参数、返回多值。AI 项目里"评估一个模型"这类逻辑
写一次、处处调用：</p>
{block("function_demo")}

<h3>4.8 类与对象：面向对象入门</h3>
<p>类（class）把<b>数据（属性）</b>和<b>行为（方法）</b>打包。sklearn 的所有模型都是类：
<span class="inline-code">model.fit()</span> 训练、<span class="inline-code">model.predict()</span> 预测。
下面自己动手写一个最简单的"线性模型"类，提前体验这种设计：</p>
{block("class_demo")}
""")

# ---- 5 ----
A(f"""
<h2 class="sec pb" id="s5">5. 数据三件套：NumPy · Pandas · Matplotlib</h2>
<p>AI 工程师 80% 的时间在和数据打交道。三个库各司其职：<b>NumPy 负责高速数值计算，Pandas 负责表格数据清洗，
Matplotlib 负责画图。</b></p>

<h3>5.1 NumPy：向量化数值计算</h3>
<p>NumPy 的核心是 <b>ndarray 多维数组</b>。它最大的优势是<b>向量化运算</b>——对整个数组做数学运算时不需要写 for 循环，
底层用 C 实现，比纯 Python 循环快几十到上百倍。深度学习中张量（tensor）的思想与之一脉相承：</p>
{block("numpy_demo")}
<div class="tip"><b>广播（broadcasting）机制：</b>形状不同的数组运算时，NumPy 自动把小数组"复制扩展"到匹配形状。
理解广播是读懂 PyTorch/NumPy 代码的关键：</div>
<img class="fig" src="assets/chart_numpy.png">
<div class="figcap">图5  广播机制示意：(2×2) 矩阵 + (1×2) 向量 → 向量自动扩展后逐元素相加</div>

<h3 class="pb">5.2 Pandas：表格数据处理利器</h3>
<p>Pandas 的 <b>DataFrame</b> 就像内存中的 Excel 表格。真实数据永远"脏"——缺失值、异常值、类型混乱，
而数据清洗往往占 AI 项目 60% 以上的工作量：</p>
<img class="fig" src="assets/chart_pandas.png">
<div class="figcap">图6  Pandas 数据处理标准流程</div>
{block("pandas_demo")}

<h3 class="pb">5.3 Matplotlib：把数据画出来</h3>
<p>看数字永远不如看图直观。以下四种图覆盖了日常 90% 的可视化需求，均由下面这份风格统一的绘图代码思路生成：</p>
{block("matplotlib_demo") if "matplotlib_demo" in R else ""}
<img class="fig" src="assets/chart_data_4panel.png">
<div class="figcap">图7  折线图看趋势、柱状图做对比、散点图看分布、直方图看数据形状</div>
""")

# matplotlib snippet may not exist in outputs; build it if missing
if "matplotlib_demo" not in R:
    pass

# ---- 6 ----
A(f"""
<h2 class="sec pb" id="s6">6. 机器学习实战：两个完整案例</h2>

<h3>6.1 案例一：从零实现线性回归（理解"训练"的本质）</h3>
<p>机器学习的最小闭环是：<b>用模型预测 → 用损失函数衡量差距 → 用梯度下降更新参数 → 循环直到收敛</b>。
下面用纯 NumPy 实现，不加任何机器学习库，让你看清"学习"到底在做什么。任务：从带噪声的数据中学出关系 y = 2x + 1。</p>
{block("linear_regression")}
<img class="fig" src="assets/chart_lr.png">
<div class="figcap">图8  真实运行结果：loss 从 42.1 收敛到 0.240，学到的 y = 2.01x + 0.97 已非常接近真实的 y = 2x + 1</div>
<div class="tip"><b>读懂这段代码：</b><span class="inline-code">loss</span> 是"预测值与真值的平均平方误差"（损失越小模型越准）；
<span class="inline-code">grad_w / grad_b</span> 是损失对参数的偏导数（指明参数往哪个方向调整能更快降低损失）；
<span class="inline-code">lr</span> 是学习率（每步走多远，太大震荡、太小太慢）。<b>这就是所有深度学习框架内部的核心原理。</b></div>

<h3 class="pb">6.2 案例二：sklearn 三行代码训练分类器</h3>
<p>理解原理之后，工程实战可以用 scikit-learn 快速建模。鸢尾花数据集是机器学习的"Hello World"：
150 个样本、4 个特征（花萼/花瓣的长宽）、3 个品种，目标是根据特征预测品种：</p>
<img class="fig" src="assets/chart_iris.png">
<div class="figcap">图10  鸢尾花数据集可视化（取前两维特征）</div>
{block("sklearn_demo")}
<div class="tip"><b>标准建模五步曲：</b>加载数据 → <span class="inline-code">train_test_split</span> 划分训练/测试集 →
<span class="inline-code">model.fit()</span> 训练 → <span class="inline-code">model.score()</span> 评估 → 调参优化。
注意：<b>测试集绝不能参与训练</b>，否则评估结果会虚高（数据泄露）。</div>
""")

# ---- 7 ----
A(f"""
<h2 class="sec pb" id="s7">7. 常见报错与调试方法</h2>
<p>初学者 90% 的崩溃来自以下几类错误。学会<b>读懂报错信息的最后一行</b>，是编程的第一项基本功：</p>
<table>
<tr><th>报错类型</th><th>典型原因</th><th>解决方法</th></tr>
<tr><td><span class="inline-code">SyntaxError</span></td><td>少了冒号、括号不配对、中文标点</td><td>检查报错行及上一行；确认使用英文符号</td></tr>
<tr><td><span class="inline-code">IndentationError</span></td><td>缩进不一致（混用 Tab 与空格）</td><td>统一用 4 个空格，交给编辑器自动缩进</td></tr>
<tr><td><span class="inline-code">NameError</span></td><td>变量未定义或拼写错误（大小写敏感）</td><td>检查变量名；确认赋值语句在前面执行过</td></tr>
<tr><td><span class="inline-code">TypeError</span></td><td>类型不匹配，如 "1" + 1</td><td>打印 <span class="inline-code">type(x)</span> 排查，必要时 <span class="inline-code">int()</span>/<span class="inline-code">str()</span> 转换</td></tr>
<tr><td><span class="inline-code">IndexError / KeyError</span></td><td>列表索引越界 / 字典键不存在</td><td>先看 <span class="inline-code">len()</span>；字典用 <span class="inline-code">.get(k, 默认)</span></td></tr>
<tr><td><span class="inline-code">ModuleNotFoundError</span></td><td>库未安装或不在当前环境</td><td>激活正确环境后 <span class="inline-code">pip install</span></td></tr>
</table>
<div class="warn"><b>调试三板斧：</b>① 在可疑处 <span class="inline-code">print(type(x), x)</span> 观察中间值；
② 用 Notebook 的分单元格执行逐步验证；③ 复制报错最后一行去搜索，StackOverflow/CSDN 上几乎都有答案。
更进一步可学习 <span class="inline-code">pdb</span> 断点调试与 IDE 的 Debug 功能。</div>
""")

# ---- 8 ----
A(f"""
<h2 class="sec pb" id="s8">8. 学习资源与进阶建议</h2>
<h3>8.1 推荐资源</h3>
<table>
<tr><th>类型</th><th>资源</th><th>适合阶段</th></tr>
<tr><td>官方教程</td><td>Python 官方文档（中文版）tutorial 章节</td><td>阶段一</td></tr>
<tr><td>交互练习</td><td>LeetCode 简单题 / 牛客网 Python 题库</td><td>阶段一、二</td></tr>
<tr><td>数据科学</td><td>Kaggle Learn 免费微课程（Pandas / ML）</td><td>阶段二、三</td></tr>
<tr><td>机器学习</td><td>吴恩达《Machine Learning》公开课 + 《动手学机器学习》</td><td>阶段三、四</td></tr>
<tr><td>深度学习</td><td>《动手学深度学习》（PyTorch 版，d2l.ai）</td><td>进阶</td></tr>
</table>
<h3>8.2 给研究生的三条建议</h3>
<ol>
<li><b>以项目驱动，不以教程驱动。</b>学完本报告后，立刻找一个自己领域的小数据集（如实验记录、公开数据集）完成
"清洗 → 分析 → 建模 → 可视化"完整流程，知识留存率远高于只看视频。</li>
<li><b>每天写代码，哪怕 30 分钟。</b>编程是肌肉记忆，连续性比单次时长重要得多。把本报告 13 段代码全部亲手复现并改动参数，是最好的起点。</li>
<li><b>善用 AI 工具但不依赖。</b>可以让 ChatGPT/Claude 解释报错、生成示例，但核心代码必须自己理解并能手写——
后续课程中的 LLM 应用开发（RAG、Agent）都建立在这些基本功之上。</li>
</ol>
<div class="tip"><b>下一步：</b>完成本报告学习后，即可无缝衔接本课程的实训模块——从 W1《数据清洗与质量评估》开始，
逐周进阶到逻辑回归、XGBoost、神经网络、CNN、迁移学习，直至 LLM 故障诊断与多智能体项目。</div>
<p style="text-align:center;color:#9CA3AF;margin-top:36px;font-size:10pt;">— 报告完 · {TODAY} 生成 · 全部代码已在 Python 3.12 / Anaconda 环境验证通过 —</p>
""")

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Python 编程基础 · AI 课程学习报告</title><style>{CSS}</style></head>
<body>
<div class="cover">
  <div class="badge">AI 应用基础 48 课时 · 先修自学材料</div>
  <h1>Python 编程基础<br>学习报告</h1>
  <h2>—— 写给即将学习 AI 课程的研究生 ——</h2>
  <div class="kpis">
    <div class="kpi"><div class="num">8</div><div class="lab">章学习内容</div></div>
    <div class="kpi"><div class="num">10</div><div class="lab">原创中文图表</div></div>
    <div class="kpi"><div class="num">{len(R)}</div><div class="lab">段可执行代码</div></div>
    <div class="kpi"><div class="num">2</div><div class="lab">机器学习实战案例</div></div>
  </div>
  <div class="meta">生成日期：{TODAY}<br>环境：Python 3.12 · NumPy 2.4 · Pandas 3.0 · scikit-learn 1.9</div>
</div>
<div class="toc"><h2>目录</h2>{toc}</div>
{''.join(sections)}
</body></html>"""

with open("report.html", "w", encoding="utf-8") as f:
    f.write(HTML)
print("report.html written,", len(HTML), "chars")
