# Input samples

每个实验臂 3 条真实输入，从该臂全量 jsonl 的首条、中位、末条截取。全量文件仍在本地 `Testing/*/input/`，不进仓库。`manifest.json` 记录来源路径和行号。

Network / T6 样本的答案在记录自己的 `metadata.gold_target` 里。State 样本不含答案，对应标签在 `gold/`，按 `respondent_id` 对齐。`s0` 与 `s0b`（以及 `s1` 与 `s1b`）抽到的是同一批人，所以 gold 不重复放。

这些记录含调查微数据的派生字段，只用于查看 prompt 格式，不能当作完整复现数据。
