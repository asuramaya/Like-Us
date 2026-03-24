#!/bin/bash
# Session F remaining experiments
# 1. Reroute on 7B (coherent text — the real behavioral test)
# 2. Llama 3.1 8B behavioral via ollama (cross-arch behavioral)
# 3. Information-theoretic measurements on 7B

set -e
export PYTHONUNBUFFERED=1
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$BENCH_DIR/logs/session_f"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1" | tee -a "$LOG_DIR/master.log"; }

run_experiment() {
    local name="$1"
    shift
    log "START: $name"
    local start_time=$(date +%s)
    if PYTHONUNBUFFERED=1 python3 "$@" 2>&1 | tee "$LOG_DIR/${name}.log"; then
        local end_time=$(date +%s)
        local duration=$(( end_time - start_time ))
        log "DONE:  $name (${duration}s)"
    else
        local end_time=$(date +%s)
        local duration=$(( end_time - start_time ))
        log "FAIL:  $name (${duration}s)"
    fi
}

log "=== REMAINING EXPERIMENTS ==="

# 1. Information-theoretic measurements on 7B
run_experiment "7b_info_theory" \
    "$BENCH_DIR/info_theory.py" --model Qwen/Qwen2.5-7B-Instruct --scenarios 5

# 2. Reroute on 7B — the critical test with coherent text
run_experiment "7b_reroute" \
    "$BENCH_DIR/reroute.py" --model Qwen/Qwen2.5-7B-Instruct --scenarios 3

# 2. Wait for 7B behavioral to finish (check if still running)
if ps aux | grep "ollama_behavioral.*qwen2.5:7b" | grep -v grep > /dev/null 2>&1; then
    log "7B behavioral still running — waiting..."
    while ps aux | grep "ollama_behavioral.*qwen2.5:7b" | grep -v grep > /dev/null 2>&1; do
        sleep 60
    done
    log "7B behavioral finished"
fi

# 3. Llama 3.1 8B behavioral via ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "Starting ollama..."
    ollama serve &
    sleep 3
fi

run_experiment "llama8b_behavioral" \
    "$BENCH_DIR/ollama_behavioral.py" --model llama3.1:8b --scenarios 10 --turns 6

log "=== REMAINING EXPERIMENTS DONE ==="
log "Final data inventory:"
ls -la "$BENCH_DIR/neuron_data/" | wc -l | tee -a "$LOG_DIR/master.log"
