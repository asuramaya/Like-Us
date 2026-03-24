#!/bin/bash
# Cross-architecture experiments — runs in PARALLEL with behavioral
# Tests: is the MLP-only finding Qwen-specific?

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

log "=== CROSS-ARCHITECTURE (parallel with behavioral) ==="

# Try models in order of likelihood to work without auth
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-2b-it"
    "meta-llama/Llama-3.1-8B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
    log "Trying: $MODEL"

    # Quick load test (1 scenario, controls-only)
    if PYTHONUNBUFFERED=1 python3 "$BENCH_DIR/run_controls.py" --model "$MODEL" --scenarios 1 --controls-only 2>&1 | tee "$LOG_DIR/${MODEL_TAG}_test.log" | grep -q "Loaded"; then
        log "SUCCESS: $MODEL loads"

        # Full controls
        run_experiment "${MODEL_TAG}_controls" \
            "$BENCH_DIR/run_controls.py" --model "$MODEL"

        # Causal patching (the critical test)
        run_experiment "${MODEL_TAG}_patching" \
            "$BENCH_DIR/patch_all_layers.py" --model "$MODEL" --scenarios 5

        # Exhaustion
        run_experiment "${MODEL_TAG}_exhaust" \
            "$BENCH_DIR/exhaust_small.py" --model "$MODEL"

        # Token sweep
        run_experiment "${MODEL_TAG}_token_sweep" \
            "$BENCH_DIR/single_token_sweep.py" --model "$MODEL"

        log "CROSS-ARCH COMPLETE: $MODEL"
        break
    else
        log "FAIL to load: $MODEL"
        continue
    fi
done

log "=== CROSS-ARCHITECTURE DONE ==="
