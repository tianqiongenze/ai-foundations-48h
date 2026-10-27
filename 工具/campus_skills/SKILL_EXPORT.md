# campus_skills 导出说明（OpenAI tools协议）

> 本目录3个Skill（课表/DDL/专注）按OpenAI tools协议实现，可导入扣子(Coze)/Dify等平台。

## 导出清单
| Skill | 文件 | 功能 | 协议 |
|---|---|---|---|
| skill_schedule | skill_schedule.py | 课表查询（读my_schedule.json） | OpenAI tools function calling |
| skill_deadline | skill_deadline.py | DDL排序（WSPT调度，与M5-1同源） | 同上 |
| skill_focus | skill_focus.py | 专注时长记录 | 同上 |

## 验证
`python verify_skills.py` → 7项ALL PASS（教师版目录内已验）

## 平台导入
1. 各平台"创建函数/工具"→粘贴对应py的函数定义（docstring即schema）
2. my_schedule.json/my_ddl.json为数据文件，随Skill上传或平台变量托管
