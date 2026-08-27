# ISSP 2017（T6 跨国）

没有提名法，永远没有 Network 阶段。靶量是 `resource_summary`，配置 `roster: none`。

提示模板固定，只改国家：80 人 × 30 国 = 2400 条，臂名 `t6_country_resources`。

```bash
python Testing/scripts/build_network_inputs.py --dataset issp_2017
bash Testing/scripts/launch_t6_issp_gpu1.sh
```

Gold：[`Survey/issp/cleaned/`](../../../Survey/issp/cleaned/)。
