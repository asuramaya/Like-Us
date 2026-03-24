#!/bin/bash
# Session F: Full falsification battery
# M3 Max 36GB — everything that fits
# Each experiment logs stdout+stderr to bench/logs/session_f/

set -e
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
    if python3 "$@" 2>&1 | tee "$LOG_DIR/${name}.log"; then
        local end_time=$(date +%s)
        local duration=$(( end_time - start_time ))
        log "DONE:  $name (${duration}s)"
    else
        local end_time=$(date +%s)
        local duration=$(( end_time - start_time ))
        log "FAIL:  $name (${duration}s) — check $LOG_DIR/${name}.log"
    fi
}

# ================================================================
# PHASE 1: Qwen 7B via TransformerLens
# The critical scale question: does the gap close?
# ================================================================

log "=== PHASE 1: QWEN 7B VIA TRANSFORMERLENS ==="

# 1a. Full matrix + controls (17 scenarios × 16 conditions = 272 forward passes)
run_experiment "7b_controls" \
    "$BENCH_DIR/run_controls.py" --model Qwen/Qwen2.5-7B-Instruct

# 1b. Causal patching: MLP vs attention at every layer (the MLP-only question)
run_experiment "7b_patching" \
    "$BENCH_DIR/patch_all_layers.py" --model Qwen/Qwen2.5-7B-Instruct --scenarios 5

# 1c. Exhaustion: cosine, attention flow, logit lens, clustering, heads
run_experiment "7b_exhaust" \
    "$BENCH_DIR/exhaust_small.py" --model Qwen/Qwen2.5-7B-Instruct

# 1d. Token sweep + superadditivity (does the inversion hold at 7B?)
run_experiment "7b_token_sweep" \
    "$BENCH_DIR/single_token_sweep.py" --model Qwen/Qwen2.5-7B-Instruct

# 1e. Saturation curves
run_experiment "7b_saturation" \
    "$BENCH_DIR/saturation_test.py" --model Qwen/Qwen2.5-7B-Instruct

# 1f. Degradation: all conditions × 10 scenarios × 8 turns
run_experiment "7b_degradation" \
    "$BENCH_DIR/degradation_extended.py" --model Qwen/Qwen2.5-7B-Instruct --scenarios 10 --turns 8

log "=== PHASE 1 COMPLETE ==="

# ================================================================
# PHASE 2: Qwen 7B via ollama (behavioral ground truth)
# ================================================================

log "=== PHASE 2: QWEN 7B BEHAVIORAL (OLLAMA) ==="

# Check ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "Starting ollama..."
    ollama serve &
    sleep 3
fi

# Check 7B is available
if ! ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    log "Pulling qwen2.5:7b..."
    ollama pull qwen2.5:7b
fi

# Full behavioral: 17 scenarios × 4 conditions × 6 turns
run_experiment "7b_behavioral" \
    "$BENCH_DIR/ollama_behavioral.py" --model qwen2.5:7b --scenarios 17 --turns 6

log "=== PHASE 2 COMPLETE ==="

# ================================================================
# PHASE 3: Cross-architecture (Mistral 7B or Llama)
# Is the MLP-only finding Qwen-specific?
# ================================================================

log "=== PHASE 3: CROSS-ARCHITECTURE ==="

# Try Mistral 7B first (no auth needed), fall back to others
CROSS_MODELS=("mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-3.1-8B-Instruct")

for CROSS_MODEL in "${CROSS_MODELS[@]}"; do
    MODEL_TAG=$(echo "$CROSS_MODEL" | tr '/' '_')
    log "Trying cross-arch: $CROSS_MODEL"

    # Test if model loads (quick check with 1 scenario)
    if python3 "$BENCH_DIR/run_controls.py" --model "$CROSS_MODEL" --scenarios 1 2>&1 | head -20 | grep -q "Loaded"; then
        log "Model loaded: $CROSS_MODEL"

        # Full matrix
        run_experiment "${MODEL_TAG}_controls" \
            "$BENCH_DIR/run_controls.py" --model "$CROSS_MODEL"

        # Causal patching
        run_experiment "${MODEL_TAG}_patching" \
            "$BENCH_DIR/patch_all_layers.py" --model "$CROSS_MODEL" --scenarios 3

        # Exhaustion
        run_experiment "${MODEL_TAG}_exhaust" \
            "$BENCH_DIR/exhaust_small.py" --model "$CROSS_MODEL"

        log "Cross-arch complete: $CROSS_MODEL"
        break
    else
        log "Failed to load $CROSS_MODEL, trying next..."
    fi
done

# Also run cross-arch behavioral via ollama
for OLLAMA_MODEL in "llama3.1:8b" "mistral:7b"; do
    if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
        run_experiment "${OLLAMA_MODEL//[:.]/_}_behavioral" \
            "$BENCH_DIR/ollama_behavioral.py" --model "$OLLAMA_MODEL" --scenarios 10 --turns 6
        break
    else
        log "Pulling $OLLAMA_MODEL for behavioral..."
        if ollama pull "$OLLAMA_MODEL" 2>&1; then
            run_experiment "${OLLAMA_MODEL//[:.]/_}_behavioral" \
                "$BENCH_DIR/ollama_behavioral.py" --model "$OLLAMA_MODEL" --scenarios 10 --turns 6
            break
        else
            log "Failed to pull $OLLAMA_MODEL"
        fi
    fi
done

log "=== PHASE 3 COMPLETE ==="

# ================================================================
# SUMMARY
# ================================================================

log "=== SESSION F EXPERIMENTS COMPLETE ==="
log "Data files:"
ls -la "$BENCH_DIR/neuron_data/" | grep -v "^total" | tee -a "$LOG_DIR/master.log"
log "Log files:"
ls -la "$LOG_DIR/" | grep -v "^total" | tee -a "$LOG_DIR/master.log"

echo ""
echo "========================================"
echo "All experiments complete."
echo "Data: $BENCH_DIR/neuron_data/"
echo "Logs: $LOG_DIR/"
echo "========================================"
