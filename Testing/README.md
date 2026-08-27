# Testing

实验按判据 T1–T6 组织。脚本按 Testing 分类存放；共享基础设施在 `common/`。

```text
Testing/
  common/                    # 数据清洗、输入构建、生成器、批量 launch、paths
  T1_association/scripts/    # State eval 1–3；UCNets network eval
  T2_information_use/scripts/  # S2 条件化 state（launch_s2, build_s2_cleaned）
  T3_interval_following/scripts/  # 补充 network 生成 launch；Signos pilot
  T4_individual_resample/scripts/  # R1 输入构建与 eval；T4 launch
  T5_additivity/scripts/     # N4/N5/N6 additivity launch
  T6_cross_population/scripts/  # 跨人群 eval 与 ISSP launch
```

批量评估入口：`Testing/common/run_model_evaluation.sh`（依次调用 T1 / T3 / T4 / T6 下各 eval 脚本）。

本地完整实验还需 `Survey/` 数据与 `Testing/configs/`；本仓库仅公开脚本与论文说明。
