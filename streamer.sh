#!/usr/bin/env bash
set -euo pipefail

export TZ="Europe/Rome"
export LD_LIBRARY_PATH="/home/sdhinakaran/micromamba/envs/StreamViz/lib:$LD_LIBRARY_PATH"

# === CONFIG ===
ENV_PY="/home/sdhinakaran/micromamba/envs/StreamViz/bin/python"

# Separate job paths for ECMWF (4-hourly) and MEPS (3-hourly)
ECMWF_PATHS=(
  "/home/sdhinakaran/task/StreamViz/streamer/ecmwf_single_worker.py"
  "/home/sdhinakaran/task/StreamViz/streamer/ecmwf_pressure_worker.py" 
)

MEPS_PATHS=(
  "/home/sdhinakaran/task/StreamViz/streamer/meps_single_worker.py"
  "/home/sdhinakaran/task/StreamViz/streamer/meps_pressure_worker.py"
)

# ECMWF times (existing four-hourly schedule)
ECMWF_TIMES=("11:00" "16:00" "23:00" "04:30")

# MEPS times (every 3 hours starting from 3AM)
MEPS_TIMES=("03:00" "06:00" "09:00" "12:00" "15:00" "18:00" "21:00" "00:00")

STATE_FILE=".last_run_stamp"
MEPS_STATE_FILE=".meps_last_run_stamp"

# sanity check - verify all files exist
for job_path in "${ECMWF_PATHS[@]}" "${MEPS_PATHS[@]}"; do
  if [[ ! -f "$job_path" ]]; then
    echo "[scheduler] ERROR: File not found: $job_path" >&2
    exit 127
  fi
done

if [[ ! -x "$ENV_PY" ]]; then
  echo "[scheduler] ERROR: Python not found at $ENV_PY" >&2
  exit 127
fi

echo "[scheduler] up. TZ=$TZ"
echo "[scheduler] ECMWF times: ${ECMWF_TIMES[*]}"
echo "[scheduler] MEPS times: ${MEPS_TIMES[*]}"
echo "[scheduler] using python: $ENV_PY"

while true; do
  now_hm="$(date +%H:%M)"
  current_hour=$(date +%H)
  current_minute=$(date +%M)
  
  # Process ECMWF jobs (existing four-hourly schedule)
  for idx in "${!ECMWF_TIMES[@]}"; do
    if [[ "$now_hm" == "${ECMWF_TIMES[$idx]}" ]]; then
      stamp="${now_hm}-ecmwf-$(date +%Y-%m-%d)"
      last="$(cat "$STATE_FILE" 2>/dev/null || true)"
      if [[ "$stamp" != "$last" ]]; then
        echo "$stamp" > "$STATE_FILE"

        case "$idx" in
          0) export DATE="$(date +%F)";                export TIME="00" ;; # 11:00
          1) export DATE="$(date +%F)";                export TIME="06" ;; # 16:00
          2) export DATE="$(date +%F)";                export TIME="12" ;; # 23:00
          3) export DATE="$(date -d 'yesterday' +%F)"; export TIME="18" ;; # 04:30
        esac

        echo "[scheduler] $(date) running ECMWF jobs with DATE=$DATE TIME=$TIME"
        
        # Run ECMWF Python files
        for job_path in "${ECMWF_PATHS[@]}"; do
          echo "[scheduler] executing $job_path"
          if ! "$ENV_PY" "$job_path"; then
            echo "[scheduler] $job_path failed (exit $?)"
          fi
        done
      fi
    fi
  done
  
  # Process MEPS jobs (every 3 hours with specific DATE/TIME assignments)
  for meps_time in "${MEPS_TIMES[@]}"; do
    if [[ "$now_hm" == "$meps_time" ]]; then
      stamp="${now_hm}-meps-$(date +%Y-%m-%d)"
      last="$(cat "$MEPS_STATE_FILE" 2>/dev/null || true)"
      if [[ "$stamp" != "$last" ]]; then
        echo "$stamp" > "$MEPS_STATE_FILE"
        
        # Determine DATE and TIME for MEPS based on the specific rules
        case "$meps_time" in
          "03:00")
            export DATE="$(date +%F)"
            export TIME="00"
            ;;
          "06:00")
            export DATE="$(date +%F)"
            export TIME="03"
            ;;
          "09:00")
            export DATE="$(date +%F)"
            export TIME="06"
            ;;
          "12:00")
            export DATE="$(date +%F)"
            export TIME="09"
            ;;
          "15:00")
            export DATE="$(date +%F)"
            export TIME="12"
            ;;
          "18:00")
            export DATE="$(date +%F)"
            export TIME="15"
            ;;
          "21:00")
            export DATE="$(date +%F)"
            export TIME="18"
            ;;
          "00:00")
            export DATE="$(date -d 'yesterday' +%F)"
            export TIME="21"
            ;;
        esac

        echo "[scheduler] $(date) running MEPS jobs with DATE=$DATE TIME=$TIME"
        
        # Run MEPS Python files
        for job_path in "${MEPS_PATHS[@]}"; do
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