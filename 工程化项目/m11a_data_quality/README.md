# 数据清洗与质量评估（05-AI应用基础48课时版 · W07 第1次课）

> **第三种代码形态：Python 工程化项目** —— 多文件、规则外置、测试齐全、文档完备，
> 符合 GB/T 8566（软件生存周期过程）/ GB/T 9385（需求规格）/ GB/T 15532（测试规范）/
> GB/T 8567（文档编制），并以 TOGAF 四域描述架构。

## 一、30 秒跑起来
```bash
cd m11a_data_quality
python run.py                 # 输出到 output/ 并打印摘要
python -m pytest -q           # 全量测试（应全绿）
```

## 二、目录结构
```
m11a_data_quality/
├── run.py                  # 唯一入口（argparse CLI）
├── pyproject.toml          # 包元数据 + pytest 配置
├── requirements.txt        # 依赖清单
├── config/rules.yaml       # 规则外置：改配置不改代码
├── src/data_quality48/
│   ├── config.py           # YAML 加载 + Settings（含 yaml_lite 兜底）
│   ├── core.py             # 纯函数领域逻辑（零 IO，可单测）
│   ├── io_utils.py         # CSV/JSON/原子写
│   ├── report.py           # 报告渲染
│   ├── pipeline.py         # load → process → evaluate → report
│   └── cli.py              # 命令入口（[project.scripts] 真实落点，run.py 委托至此）
├── tests/
│   ├── test_core.py        # 正常/边界/异常三档
│   ├── test_config.py      # 配置加载与"配置真被读取"
│   └── test_tiers.py       # 工单 D/C/B/A 四档数值门禁
└── docs/                   # 01~08 全套过程文档（见下）
```

## 三、过程文档（GB/T 8567 文档编制）
| 文档 | 对应标准/方法 |
|---|---|
| [01 需求规格](docs/01-需求规格.md) | GB/T 9385 |
| [02 架构说明](docs/02-架构说明.md) | TOGAF 四域 |
| [03 设计说明](docs/03-设计说明.md) | 模块设计与异常策略 |
| [04 测试计划与报告](docs/04-测试计划与报告.md) | GB/T 15532 |
| [05 运行手册](docs/05-运行手册.md) | GB/T 8566 运行过程 |
| [06 配置管理与变更记录](docs/06-配置管理与变更记录.md) | GB/T 8566 配置管理 |
| [07 ADR-001](docs/07-ADR-001-架构决策记录.md) | 架构决策记录 |
| [08 国标与TOGAF映射](docs/08-国标与TOGAF映射.md) | GB/T 8566 过程组落地 |

## 四、三种代码形态的定位
| 形态 | 目录 | 定位 |
|---|---|---|
| ① starter | `../学生代码框架/` | 学生填空起步（TODO 骨架） |
| ② 教师参考 | `../教师参考/` | 完成态参考（单文件示意） |
| ③ **工程化项目（本目录）** | `../工程化项目/` | **企业级工程结构 + 全套流程资料** |

## 五、验收口径（对齐工单 M1-1a 四档）
| 档 | 工单标准 | 本项目实测（见 `output/report.txt`） |
|---|---|---|
| 及格 D | 温度列缺失率降为 0 且行数不变 | 9.00% → 0.00%，行数 300 → 300（缺失阶段零删行） |
| 良好 C | 缺失处理率 ≥ 95% + 三清单齐全 | 100.00%；`missing_list/anomaly_list/duplicate_list.csv` |
| 优秀 B | 质量分 ≥ 85 且理由完整、决策记录完整 | 93.60 → 99.07（0.4/0.4/0.2 加权，三率明细 + 逐列理由） |
| 优 A | 预测填充优于均值且论证清楚 | MAE 0.6749℃ < 均值 2.6774℃（改善 74.79%，逐格胜率 86.1%） |

工程口径：
- `python run.py` 生成 `output/` 七件产物（report.txt、result.json、三清单 CSV、
  audit_log.csv、cleaned.csv）；`python -m pytest -q` 全绿（39 用例，含四档门禁）
- 配置修改不需改代码；二次运行结果一致（种子固定）
- `pip install -e .` 后可用 `m11a_data_quality` 命令，与 `run.py` 同一入口实现
- 过程文档 8 份齐全，`docs/01` 第 7 节为工单↔需求↔测试追踪矩阵
