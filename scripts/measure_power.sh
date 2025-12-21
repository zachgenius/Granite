#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/measure_power.sh [--label <name>] [--out <path>] -- <command> [args...]

Runs a command and reports basic battery/power information.

Options:
  --label <name>   Label for the run (default: command)
  --out <path>     Write raw output to a file

Notes:
  - On macOS, this uses `pmset -g batt` before/after the command.
  - If `powermetrics` is available and the script is run with sudo, it captures
    SMC power samples while the command runs.
EOF
}

label=""
out_path=""
cmd=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      label="$2"
      shift 2
      ;;
    --out)
      out_path="$2"
      shift 2
      ;;
    --)
      shift
      cmd=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#cmd[@]} -eq 0 ]]; then
  echo "Missing command to run."
  usage
  exit 1
fi

if [[ -z "$label" ]]; then
  label="${cmd[0]}"
fi

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

read_batt_percent_macos() {
  pmset -g batt | awk -F';' 'NR==2 {gsub(/[^0-9]/,"",$2); print $2}'
}

read_batt_percent_linux() {
  if [[ -r /sys/class/power_supply/BAT0/capacity ]]; then
    cat /sys/class/power_supply/BAT0/capacity
  fi
}

start_ts=$(timestamp)
start_epoch=$(date +%s)
start_batt=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  start_batt=$(read_batt_percent_macos || true)
elif [[ -f /sys/class/power_supply/BAT0/capacity ]]; then
  start_batt=$(read_batt_percent_linux || true)
fi

powermetrics_pid=""
powermetrics_file=""
if [[ "$(uname -s)" == "Darwin" && -x /usr/bin/powermetrics ]]; then
  if sudo -n true 2>/dev/null; then
    powermetrics_file=$(mktemp -t granite_powermetrics.XXXXXX)
    sudo -n powermetrics --samplers smc -i 1000 > "$powermetrics_file" 2>/dev/null &
    powermetrics_pid=$!
  else
    echo "powermetrics available but requires sudo; run with sudo for power sampling."
  fi
fi

echo "Running: ${cmd[*]}"
echo "Start: $start_ts"
"${cmd[@]}"

if [[ -n "$powermetrics_pid" ]]; then
  sudo -n kill "$powermetrics_pid" 2>/dev/null || true
  wait "$powermetrics_pid" 2>/dev/null || true
fi

end_ts=$(timestamp)
end_epoch=$(date +%s)
duration=$((end_epoch - start_epoch))

end_batt=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  end_batt=$(read_batt_percent_macos || true)
elif [[ -f /sys/class/power_supply/BAT0/capacity ]]; then
  end_batt=$(read_batt_percent_linux || true)
fi

echo "End:   $end_ts"
echo "Label: $label"
echo "Duration: ${duration}s"

if [[ -n "$start_batt" && -n "$end_batt" ]]; then
  delta=$((start_batt - end_batt))
  echo "Battery: ${start_batt}% -> ${end_batt}% (delta: ${delta}%)"
fi

if [[ -n "$powermetrics_file" ]]; then
  echo "powermetrics output: $powermetrics_file"
fi

if [[ -n "$out_path" ]]; then
  {
    echo "Label: $label"
    echo "Start: $start_ts"
    echo "End: $end_ts"
    echo "Duration: ${duration}s"
    if [[ -n "$start_batt" && -n "$end_batt" ]]; then
      echo "Battery: ${start_batt}% -> ${end_batt}%"
    fi
    if [[ -n "$powermetrics_file" ]]; then
      echo ""
      echo "--- powermetrics ---"
      if [[ -s "$powermetrics_file" ]]; then
        cat "$powermetrics_file"
      else
        echo "powermetrics output file is empty"
      fi
    fi
  } > "$out_path"
  echo "Saved power trace to: $out_path"
fi
