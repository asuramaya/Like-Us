#!/bin/bash
# Cross-architecture v2 — using OPEN models that don't need auth
# Mistral-7B-Instruct-v0.1 and Pythia-6.9B are both in TransformerLens and ungated

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

log "=== CROSS-ARCH V2: OPEN MODELS ==="

# Models to try — all in TransformerLens, all open
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.1"
    "EleutherAI/pythia-6.9b-deduped"
    "microsoft/Phi-3-mini-4k-instruct"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_TAG=$(echo "$MODEL" | tr '/' '_' | tr '.' '-')
    log "Trying: $MODEL"

    # Quick load test
    LOADED=$(PYTHONUNBUFFERED=1 python3 -c "
import os; os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from transformer_lens import HookedTransformer
try:
    model = HookedTransformer.from_pretrained('$MODEL', device='mps', dtype=torch.float16)
    print(f'LOADED:{model.cfg.n_layers}:{model.cfg.n_heads}:{model.cfg.d_model}')
    del model; torch.mps.empty_cache()
except Exception as e:
    print(f'FAIL:{e}')
" 2>&1 | grep -E "^(LOADED|FAIL):" | head -1)

    if echo "$LOADED" | grep -q "^LOADED:"; then
        log "$LOADED"

        # Full controls
        run_experiment "${MODEL_TAG}_controls" \
            "$BENCH_DIR/run_controls.py" --model "$MODEL"

        # Causal patching (THE question)
        run_experiment "${MODEL_TAG}_patching" \
            "$BENCH_DIR/patch_all_layers.py" --model "$MODEL" --scenarios 5

        # Exhaustion (cosine, attention flow, logit lens, clustering, heads)
        run_experiment "${MODEL_TAG}_exhaust" \
            "$BENCH_DIR/exhaust_small.py" --model "$MODEL"

        log "CROSS-ARCH COMPLETE: $MODEL"
        break
    else
        log "FAIL: $MODEL — $LOADED"
        continue
    fi
done

log "=== CROSS-ARCH V2 DONE ==="
