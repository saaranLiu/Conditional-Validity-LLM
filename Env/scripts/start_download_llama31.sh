#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/bld/data/data2/jiaxin/simulation
ENV=$ROOT/Env
LOG=$ENV/logs/download_llama31_instruct.log
PIDFILE=$ENV/logs/download_llama31_instruct.pid
PY=/home/bld/anaconda3/envs/sociosim/bin/python
SCRIPT=$ENV/scripts/download_llama31_instruct.py

mkdir -p "$ENV/logs"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

# stop leftovers
pkill -f "$SCRIPT" 2>/dev/null || true
sleep 1

rm -f "$ENV/models/modelscope_cache/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master/original/consolidated.00.pth.incomplete" || true

echo "===== restart $(date -Is) =====" >> "$LOG"
nohup "$PY" "$SCRIPT" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "started pid=$(cat "$PIDFILE") log=$LOG"
sleep 5
ps -p "$(cat "$PIDFILE")" -o pid,etime,cmd || true
tail -c 600 "$LOG" | tr '\r' '\n' | tail -20 || true
