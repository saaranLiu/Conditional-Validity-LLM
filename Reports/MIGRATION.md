# 2026-08-18 目录重组

原始调查收进 `Survey/`，实验按 Testing 分目录，不再使用 `Phase1_experiments/`、`Phase2_experiments/`、`Stage-Restoration_1/`。分析与独立文档在 `Reports/`，环境与模型权重在 `Env/`。

- 臂名对照：[`Testing/NAMING.md`](Testing/NAMING.md)
- 路径解析：[`Testing/scripts/paths.py`](Testing/scripts/paths.py)
- 实验登记：[`Reports/EXPERIMENT_REGISTRY.md`](Reports/EXPERIMENT_REGISTRY.md)
- 分析与论文：[`Reports/`](Reports/)
- 环境：[`Env/`](Env/)

每条生成臂只物理存放一次（主 Testing）；其它 Testing 用相对符号链接引用。清洗 gold 在 `Survey/{survey}/cleaned/`，不进 Testing。Add Health 只在 Survey，未建 Testing。

本地备份：

- `/home/bld/data/data2/jiaxin/simulation_backup_20260813T220610`
- `/home/bld/data/data2/jiaxin/simulation_backup_20260818T171600`

未重跑模型；`prompt_text` 未改，只改了 `task`、`input_id` 与配置路径元数据。
