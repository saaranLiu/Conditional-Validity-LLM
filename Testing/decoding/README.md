# Native-path decoding redesign

This directory contains the redesigned decoding experiment. The previous
forced-choice experiment, including its scripts, inputs, outputs, logs, and
README, is preserved unchanged under [`temp/`](temp/).

## Why this redesign

The legacy A/B/C probe and the original full-JSON experiment did not use the
same measurement path. They differed in model precision, prompt, response
format, sample design, and estimand. Consequently, their probability spans
could not identify "structural generation attenuation."

The redesign aligns the following:

- one local HuggingFace `Meta-Llama-3.1-8B-Instruct` BF16 model;
- the original full network prompt and one chat template;
- temperature `0.7` for native sequence scores and JSON sampling;
- the same 400 GSS respondents in both T3 tasks;
- all three counterfactual buckets for every respondent;
- targets recomputed from generated alters/ties rather than self-reported
  composition fields.

## Stages

1. **Native JSON slot score**
   - `sex_homophily`: score `male` vs `female` at five JSON `gender` slots.
   - `closure`: score the three allowed `tie_strength` strings at all ten dyads.
   - `size_bucket`: score every integer from 1 through 30 at
     `composition_summary.num_alters`.
   - Candidates are scored as complete token sequences. No A/B/C labels are
     used.

2. **Matched full JSON generation**
   - Generate all three bucket interventions for each ego on the same HF stack.
   - T3 uses three independent samples per ego and bucket.
   - The size positive control uses one sample per ego and bucket.

3. **Paired summary**
   - Report condition means, parse/target-valid rates, interval compliance,
     and ego-paired endpoint contrasts with bootstrap confidence intervals.

## Layout

```text
Testing/decoding/
├── input/<run>/counterfactual_inputs.jsonl
├── output/<run>/native_slot_scores.jsonl
├── output/<run>/full_json_samples.jsonl
├── output/<run>/summary.json
├── logs/<run>/
├── scripts/run_native_path.py
├── run_gpu0.sh
└── temp/                         # archived legacy experiment
```

## Run

Four-GPU sharded run (one model replica per GPU):

```bash
for i in 0 1 2 3; do
  GPU_ID="$i" SHARD_INDEX="$i" \
    nohup bash Testing/decoding/run_gpu_shard.sh \
    > "Testing/decoding/logs/main/shard_${i}/launcher.log" 2>&1 &
done
```

All three counterfactual conditions for an ego/task are assigned to the same
shard. Shard outputs live under `output/main/shard_0` through `shard_3`.

Single-GPU fallback:

```bash
bash Testing/decoding/run_gpu0.sh
```

The runner is resumable: existing `decision_id` and `sample_id` records are
skipped. For a smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 /home/bld/anaconda3/envs/sociosim/bin/python \
  Testing/decoding/scripts/run_native_path.py \
  --run-name smoke --max-egos 1 --native-batch-size 4 \
  --generation-batch-size 1 --t3-reps 1 --size-reps 1
```
