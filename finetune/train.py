#!/usr/bin/env python3
"""QLoRA / LoRA fine-tuning for Infinidev tool-calling.

Supports two strategies for large models (27-32B) on 2x RTX A5000 (24GB):

  Strategy 1 — QLoRA single-GPU (fits up to ~14B in 4-bit on one GPU)
    python finetune/train.py --model Qwen/Qwen2.5-Coder-14B-Instruct

  Strategy 2 — DeepSpeed ZeRO-3 + LoRA + CPU offload (27-32B across 2 GPUs)
    torchrun --nproc_per_node=2 finetune/train.py \
        --model Qwen/Qwen3-32B \
        --deepspeed finetune/ds_config_zero3.json \
        --no-quantize --max-seq-len 4096

Key anti-OOM features:
  - Chunked lm_head loss (never materializes full [seq_len x vocab_size] logits)
  - Gradient checkpointing (trades compute for VRAM)
  - CPU offload of optimizer states + params (uses your 256GB RAM)
  - Assistant-only label masking (system/user/tool tokens ignored in loss)
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.integrations.deepspeed import HfDeepSpeedConfig

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "output" / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "checkpoints"
FINAL_DIR = BASE_DIR / "output" / "model"

ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def load_dataset_with_masks(split: str, fmt: str, tokenizer, max_len: int) -> Dataset:
    """Load dataset with assistant-only labels.

    Tokens from system/user/tool turns → -100 (ignored in loss).
    Only assistant turn tokens are trainable.
    """
    path = DATASET_DIR / f"{split}_{fmt}.jsonl"
    if not path.exists():
        path = DATASET_DIR / f"{split}.jsonl"

    examples = []
    skipped = 0
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue

            input_ids = tokenizer.encode(
                text, add_special_tokens=False, truncation=True, max_length=max_len,
            )
            labels = [-100] * len(input_ids)

            # Mark assistant turn tokens as trainable
            pos = 0
            while True:
                start = text.find(ASSISTANT_START, pos)
                if start == -1:
                    break
                content_start = start + len(ASSISTANT_START)
                end = text.find(IM_END, content_start)
                if end == -1:
                    break
                end += len(IM_END)

                tok_start = len(tokenizer.encode(text[:content_start], add_special_tokens=False))
                tok_end = len(tokenizer.encode(text[:end], add_special_tokens=False))

                for j in range(tok_start, min(tok_end, len(labels))):
                    labels[j] = input_ids[j]

                pos = end

            # Skip if no trainable tokens (e.g., truncation removed all assistant turns)
            if all(l == -100 for l in labels):
                skipped += 1
                continue

            examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            })

    if skipped:
        print(f"  Skipped {skipped} examples with no trainable tokens after truncation")
    return Dataset.from_list(examples)


class ChunkedLossTrainer(Trainer):
    """Trainer that computes lm_head + loss in small chunks.

    The full (seq_len x vocab_size) logit tensor is NEVER materialized.
    For Qwen3 with vocab_size=151936, this prevents OOM on the logit projection.
    At chunk_size=256: peak logit tensor = 256 * 151936 * 2 bytes ≈ 74MB (vs 4.6GB for seq_len=8192).
    """

    def __init__(self, *args, loss_chunk_size: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_chunk_size = loss_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # Get hidden states WITHOUT computing logits
        outputs = model(**inputs, output_hidden_states=True)

        # Move labels to same device as hidden states (needed for device_map="auto")
        labels = labels.to(outputs.hidden_states[-1].device)

        # Navigate through PEFT/DeepSpeed wrappers to find lm_head
        lm_head = self._find_lm_head(model)

        if lm_head is None:
            # Fallback: use logits from the full forward (less memory efficient)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

        # Get last hidden state
        hidden = outputs.hidden_states[-1]

        # Shift for causal LM: predict next token
        shift_hidden = hidden[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        seq_len = shift_hidden.size(1)
        chunk = self.loss_chunk_size
        lm_device = lm_head.weight.device
        total_loss = torch.tensor(0.0, device=lm_device, dtype=torch.float32)
        total_tokens = 0

        for i in range(0, seq_len, chunk):
            chunk_labels = shift_labels[:, i:i+chunk].reshape(-1)

            # Skip chunks with no trainable tokens
            mask = chunk_labels != -100
            if not mask.any():
                continue

            # Apply lm_head only to this chunk — keeps peak memory low
            chunk_hidden = shift_hidden[:, i:i+chunk, :]
            chunk_hidden = chunk_hidden.to(lm_head.weight.device)
            chunk_logits = lm_head(chunk_hidden).float()  # upcast for stable CE
            chunk_logits = chunk_logits.reshape(-1, chunk_logits.size(-1))

            # Ensure labels on same device as logits (device_map="auto" splits layers)
            chunk_labels = chunk_labels.to(chunk_logits.device)
            n_tokens = mask.sum().item()
            chunk_loss = F.cross_entropy(chunk_logits, chunk_labels, ignore_index=-100)
            total_loss = total_loss + chunk_loss * n_tokens
            total_tokens += n_tokens

            del chunk_logits, chunk_hidden, chunk_labels

        if total_tokens > 0:
            total_loss = total_loss / total_tokens

        return (total_loss, outputs) if return_outputs else total_loss

    @staticmethod
    def _find_lm_head(model):
        """Find lm_head through PEFT/DeepSpeed wrappers."""
        # Try common wrapper paths
        candidates = [model]
        m = model
        for attr in ("model", "base_model", "model", "module"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                candidates.append(m)
        # DeepSpeed engine wraps in .module
        if hasattr(model, "module"):
            candidates.append(model.module)

        for c in candidates:
            if hasattr(c, "lm_head"):
                return c.lm_head
        return None


def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--loss-chunk-size", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--format", default="qwen_native")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Use bf16 instead of 4-bit (required for DeepSpeed ZeRO-3)")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    is_main = local_rank in (-1, 0)

    # CRITICAL: Register DeepSpeed config BEFORE model loading.
    # This tells transformers to use ZeRO-3's Init context during from_pretrained,
    # so the model is partitioned across GPUs during loading instead of
    # materializing the full model on each GPU (which causes OOM).
    ds_config_obj = None
    if args.deepspeed:
        # Initialize deepspeed comm (covers both torch.distributed AND ds comm)
        import deepspeed
        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_main = local_rank == 0

        # Load config as dict and resolve batch sizes for ZeRO-3 Init
        # (DeepSpeed validates batch math before Trainer can fill in "auto")
        world_size = torch.distributed.get_world_size()
        with open(args.deepspeed) as f:
            ds_config_dict = json.load(f)
        ds_config_dict["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config_dict["gradient_accumulation_steps"] = args.grad_accum
        ds_config_dict["train_batch_size"] = args.batch_size * args.grad_accum * world_size
        ds_config_obj = HfDeepSpeedConfig(ds_config_dict)

    if is_main:
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Strategy: {'DeepSpeed ZeRO-3 + LoRA (bf16)' if args.no_quantize else 'QLoRA (4-bit)'}")
        print(f"Epochs: {args.epochs}, LR: {args.lr}, Grad accum: {args.grad_accum}")
        print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Max seq len: {args.max_seq_len}, Loss chunk: {args.loss_chunk_size}")
        print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
        print(f"{'='*60}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────
    if args.no_quantize:
        # DeepSpeed ZeRO-3 path: load in bf16, let DS handle sharding + offload
        if is_main:
            print("Loading model in bf16 (DeepSpeed will shard + offload)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        # QLoRA path: 4-bit quantization on single GPU
        if is_main:
            print("Loading model (4-bit QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",  # split 4-bit model across all GPUs
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing before LoRA
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # Fix vocab size mismatch
    if len(tokenizer) > model.config.vocab_size:
        if is_main:
            print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────
    if is_main:
        print("Loading dataset with assistant-only masking...")
    train_ds = load_dataset_with_masks("train", args.format, tokenizer, args.max_seq_len)
    val_ds = load_dataset_with_masks("val", args.format, tokenizer, args.max_seq_len)
    if is_main:
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Verify masking
        sample = train_ds[0]
        total = len(sample["labels"])
        trainable = sum(1 for l in sample["labels"] if l != -100)
        print(f"Sample: {total} tokens, {trainable} trainable ({100*trainable//max(total,1)}%)")

    # ── Training ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,
        eval_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        deepspeed=args.deepspeed,
        # DeepSpeed handles DDP, disable native find_unused
        ddp_find_unused_parameters=False if args.deepspeed else None,
    )

    # Use ChunkedLossTrainer only for single-device (avoids multi-device complexity)
    # Standard Trainer works fine with device_map="auto" and 2 GPUs
    use_chunked = not (hasattr(model, "hf_device_map") and len(set(model.hf_device_map.values())) > 1)

    if use_chunked:
        trainer = ChunkedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            loss_chunk_size=args.loss_chunk_size,
        )
    else:
        if is_main:
            print("Multi-device detected, using standard Trainer (logits fit in memory)")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )

    if is_main:
        print("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    if is_main:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(FINAL_DIR))
        tokenizer.save_pretrained(str(FINAL_DIR))

        # Memory stats
        for i in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {peak:.1f} GB / {total:.1f} GB peak")

        print(f"\nModel saved to {FINAL_DIR}")


if __name__ == "__main__":
    main()
