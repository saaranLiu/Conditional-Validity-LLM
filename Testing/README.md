# Testing

实验按判据（Testing 1–6）组织，不再按 Phase1 / Phase2 / Stage-Restoration。

```text
Testing/
  NAMING.md
  configs/state/{dataset}.yaml
  configs/network/{dataset}.yaml
  scripts/
  T1_association/{dataset}/{input,output,eval}
  T2_information_use/...
  T3_interval_following/...
  T4_individual_resample/...
  T5_additivity/...
  T6_cross_population/...
  logs/
```

原始调查与清洗 gold 在 `Survey/{gss,ucnets,nshap,signos,issp,add_health}/`。路径解析见 `scripts/paths.py`。

一条生成臂只落在它的主 Testing 下；其它 Testing 通过符号链接引用，不要复制 jsonl。

分析、论文与实验登记见 [`../Reports/`](../Reports/)。
