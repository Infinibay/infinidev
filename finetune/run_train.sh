#!/bin/bash
# Fine-tuning launcher for Qwen3-32B on 2x RTX A5000 + 256GB RAM
#
# Strategy A (recommended): DeepSpeed ZeRO-3 — uses BOTH GPUs + CPU offload
# Strategy B (faster):      Unsloth single-GPU — simpler but seq_len limited
#
# Usage:
#   ./finetune/run_train.sh deepspeed   # Strategy A (default)
#   ./finetune/run_train.sh unsloth     # Strategy B

set -euo pipefail
cd "$(dirname "$0")/.."

STRATEGY="${1:-deepspeed}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

echo "============================================"
echo "  Infinidev Fine-tuning"
echo "  Strategy: $STRATEGY"
echo "  Model:    $MODEL"
echo "  Seq len:  $MAX_SEQ_LEN"
echo "============================================"

if [ "$STRATEGY" = "deepspeed" ]; then
    # ── DeepSpeed ZeRO-3: 2 GPUs + CPU offload ──────────────────────────
    # Memory budget:
    #   32B bf16 = ~64GB params → sharded across 2 GPUs (32GB each) + CPU offload
    #   Optimizer states on CPU (256GB RAM handles it easily)
    #   Activations: gradient checkpointing keeps them manageable
    #
    # If still OOM: reduce MAX_SEQ_LEN to 2048 or increase grad-accum
    echo ""
    echo "Launching torchrun with 2 GPUs + DeepSpeed ZeRO-3..."
    echo "  Params offloaded to CPU, optimizer on CPU"
    echo "  If OOM: reduce MAX_SEQ_LEN or LORA_R"
    echo ""

    torchrun --nproc_per_node=2 \
        finetune/train.py \
        --model "$MODEL" \
        --deepspeed finetune/ds_config_zero3.json \
        --no-quantize \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --batch-size 1 \
        --grad-accum 8 \
        --loss-chunk-size 256

elif [ "$STRATEGY" = "unsloth" ]; then
    # ── Unsloth: single GPU, 4-bit QLoRA ─────────────────────────────────
    # Memory budget:
    #   32B 4-bit ≈ 17GB model + ~3GB LoRA/activations ≈ 20GB (fits 24GB)
    #   BUT only if seq_len ≤ 2048. For longer, use deepspeed strategy.
    #
    # Runs ~2x faster per step than DeepSpeed thanks to Unsloth kernels.
    UNSLOTH_MODEL="${UNSLOTH_MODEL:-unsloth/Qwen3-32B-bnb-4bit}"
    UNSLOTH_SEQ="${UNSLOTH_SEQ:-2048}"

    echo ""
    echo "Launching Unsloth on single GPU..."
    echo "  Model: $UNSLOTH_MODEL"
    echo "  WARNING: seq_len capped at $UNSLOTH_SEQ (longer examples truncated)"
    echo "  If OOM: reduce UNSLOTH_SEQ to 1024 or use --gpu 1 for the other GPU"
    echo ""

    python finetune/train_unsloth.py \
        --model "$UNSLOTH_MODEL" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --max-seq-len "$UNSLOTH_SEQ" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --batch-size 1 \
        --grad-accum 8

elif [ "$STRATEGY" = "senn" ]; then
    # ── SENN: QLoRA + Self-Explorable Neural Networks ────────────────────
    # Same QLoRA as above, but adds conceptor learning + causal loss
    # after maturity detection. ~5-10% overhead from interventions.
    echo ""
    echo "Launching SENN-enhanced QLoRA training..."
    echo "  Phase 1: Standard SFT (until loss plateau)"
    echo "  Phase 2: SFT + causal loss + diversity loss"
    echo ""

    python finetune/train_senn.py \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --max-seq-len "${SENN_SEQ:-2048}" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --batch-size 1 \
        --grad-accum 8 \
        --loss-chunk-size 256 \
        --causal-lambda 0.1 \
        --diversity-lambda 0.05 \
        --intervention-freq 0.15 \
        --concept-rank 16 \
        --concepts-per-layer 4

else
    echo "Unknown strategy: $STRATEGY"
    echo "Usage: $0 [deepspeed|unsloth|senn]"
    exit 1
fi
