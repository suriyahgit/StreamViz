#!/usr/bin/env bash
set -euo pipefail

export TZ="Europe/Rome"
export LD_LIBRARY_PATH="/home/sdhinakaran/micromamba/envs/StreamViz/lib:$LD_LIBRARY_PATH"

# === CONFIG ===
ENV_PY="/home/sdhinakaran/micromamba/envs/StreamViz/bin/python"
JOB_PATHS=(
  "/home/sdhinakaran/task/StreamViz/streamer/ecmwf_single_worker.py"
  "/home/sdhinakaran/task/StreamViz/streamer/ecmwf_pressure_worker.py" 
  "/home/sdhinakaran/task/StreamViz/streamer/meps_single_worker.py"
  "/home/sdhinakaran/task/StreamViz/streamer/meps_pressure_worker.py"
)

#JOB_PATHS=(
#  "/home/sdhinakaran/task/StreamViz/streamer/dummy.py"
#  "/home/sdhinakaran/task/StreamViz/streamer/dummy_1.py" 
#)
TIMES=("11:00" "16:00" "23:00" "04:30")
#TIMES=("18:07" "18:08" "18:09" "18:10")
STATE_FILE=".last_run_stamp"

# sanity check - verify all files exist
for job_path in "${JOB_PATHS[@]}"; do
  if [[ ! -f "$job_path" ]]; then
    echo "[scheduler] ERROR: File not found: $job_path" >&2
    exit 127
  fi
done

if [[ ! -x "$ENV_PY" ]]; then
  echo "[scheduler] ERROR: Python not found at $ENV_PY" >&2
  exit 127
fi

echo "[scheduler] up. TZ=$TZ times=${TIMES[*]}"
echo "[scheduler] using python: $ENV_PY"
echo "[scheduler] will run files: ${JOB_PATHS[*]}"

while true; do
  now_hm="$(date +%H:%M)"
  for idx in "${!TIMES[@]}"; do
    if [[ "$now_hm" == "${TIMES[$idx]}" ]]; then
      stamp="${now_hm}-$(date +%Y-%m-%d)"
      last="$(cat "$STATE_FILE" 2>/dev/null || true)"
      if [[ "$stamp" != "$last" ]]; then
        echo "$stamp" > "$STATE_FILE"

        case "$idx" in
          0) export DATE="$(date +%F)";                export TIME="00" ;; # 11:00
          1) export DATE="$(date +%F)";                export TIME="06" ;; # 16:00
          2) export DATE="$(date +%F)";                export TIME="12" ;; # 23:00
          3) export DATE="$(date -d 'yesterday' +%F)"; export TIME="18" ;; # 04:30
        esac

        echo "[scheduler] $(date) running jobs with DATE=$DATE TIME=$TIME"
        
        # Run all three Python files sequentially
        for job_path in "${JOB_PATHS[@]}"; do
          echo "[scheduler] executing $job_path"
          if ! "$ENV_PY" "$job_path"; then
            echo "[scheduler] $job_path failed (exit $?)"
          fi
        done
        
      fi
    fi
  done
  sleep 20
done

