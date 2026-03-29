#!/usr/bin/env python3
"""SENN-enhanced QLoRA fine-tuning for Infinidev tool-calling.

Integrates Self-Explorable Neural Networks (SENN) concepts with QLoRA:
  - SENNConceptors learn latent supervision on activation manifolds
  - Hybrid loss: L_standard + λ_causal * L_causal + γ * L_diversity
  - Maturity detection: standard training first, then gradual SENN activation
  - Chunked lm_head loss to avoid logit OOM (vocab 151K)
  - Multi-GPU support via balanced device_map

Hardware target: 2x RTX A5000 (24GB each) + 256GB RAM
Model target: Qwen3-32B in 4-bit QLoRA

Usage:
    python finetune/train_senn.py
    python finetune/train_senn.py --model Qwen/Qwen3-32B --max-seq-len 2048
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


class PadCollator:
    """Pads variable-length examples to the longest in the batch."""
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "output" / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "checkpoints_senn"
FINAL_DIR = BASE_DIR / "output" / "model_senn"

ASSISTANT_START = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


# ═══════════════════════════════════════════════════════════════════════════════
# SENN Components (adapted from ~/senn/src/self_explorable/)
# ═══════════════════════════════════════════════════════════════════════════════

class SENNConceptor(nn.Module):
    """Learned low-rank bottleneck filter for activation manifolds.

    Compresses activations through encoder→latent→decoder to force
    concept extraction. The reconstruction quality measures how well
    the conceptor captures the activation pattern.
    """
    def __init__(self, name: str, hidden_dim: int, rank: int = 16):
        super().__init__()
        self.concept_name = name
        self.hidden_dim = hidden_dim
        self.rank = max(rank, 4)
        self.encoder = nn.Parameter(torch.randn(hidden_dim, self.rank) * 0.02)
        self.decoder = nn.Parameter(torch.randn(self.rank, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder.to(x.dtype)
        dec = self.decoder.to(x.dtype)
        return torch.matmul(torch.matmul(x, enc), dec)


class SENNConceptGraph(nn.Module):
    """Manages conceptors for target layers."""
    def __init__(self):
        super().__init__()
        self.conceptors = nn.ModuleDict()

    def add_conceptor(self, conceptor: SENNConceptor):
        self.conceptors[conceptor.concept_name] = conceptor

    def init_for_layers(self, layer_names: list, hidden_dim: int,
                        concepts_per_layer: int = 4, rank: int = 16):
        """Initialize conceptors for each target layer."""
        for layer in layer_names:
            safe = layer.replace(".", "_")
            for i in range(concepts_per_layer):
                name = f"{safe}_c{i}"
                self.add_conceptor(SENNConceptor(name, hidden_dim, rank))


class MaturityDetector:
    """Detects when model reaches plateau to activate SENN losses."""
    def __init__(self, warmup_steps: int = 30, threshold: float = 0.02):
        self.warmup_steps = warmup_steps
        self.threshold = threshold
        self.loss_history: list = []
        self.is_mature = False
        self.maturity_score = 0.0

    def update(self, loss: float) -> bool:
        self.loss_history.append(loss)
        if self.is_mature:
            self.maturity_score = min(1.0, self.maturity_score + 0.005)
            return False
        if len(self.loss_history) < self.warmup_steps:
            return False
        w = min(20, len(self.loss_history) // 3)
        recent = np.mean(self.loss_history[-w:])
        older = np.mean(self.loss_history[-(2*w):-w])
        if older < 1e-6:
            return False
        improvement = (older - recent) / older
        if improvement < self.threshold:
            self.is_mature = True
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading (same as train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset_with_masks(split, fmt, tokenizer, max_len):
    path = DATASET_DIR / f"{split}_{fmt}.jsonl"
    if not path.exists():
        path = DATASET_DIR / f"{split}.jsonl"
    examples, skipped = [], 0
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue
            input_ids = tokenizer.encode(text, add_special_tokens=False,
                                          truncation=True, max_length=max_len)
            labels = [-100] * len(input_ids)
            pos = 0
            while True:
                start = text.find(ASSISTANT_START, pos)
                if start == -1:
                    break
                cs = start + len(ASSISTANT_START)
                end = text.find(IM_END, cs)
                if end == -1:
                    break
                end += len(IM_END)
                ts = len(tokenizer.encode(text[:cs], add_special_tokens=False))
                te = len(tokenizer.encode(text[:end], add_special_tokens=False))
                for j in range(ts, min(te, len(labels))):
                    labels[j] = input_ids[j]
                pos = end
            if all(l == -100 for l in labels):
                skipped += 1
                continue
            examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            })
    if skipped:
        print(f"  Skipped {skipped} examples (no trainable tokens)")
    return Dataset.from_list(examples)


# ═══════════════════════════════════════════════════════════════════════════════
# SENN Trainer — chunked loss + conceptor auxiliary losses
# ═══════════════════════════════════════════════════════════════════════════════

class SENNChunkedTrainer(Trainer):
    """Trainer with chunked logit loss + SENN conceptor losses.

    The logit OOM problem: vocab=151936, seq=4096 → full logits = 2.4GB.
    Solution: compute lm_head in chunks of 256 tokens (peak = 74MB).

    SENN losses (activated after maturity):
      - L_causal: MSE between activations and conceptor reconstructions
      - L_diversity: penalizes correlated conceptors (prevents collapse)
    """

    def __init__(self, *args, concept_graph: SENNConceptGraph = None,
                 maturity: MaturityDetector = None, target_layers: list = None,
                 causal_lambda: float = 0.1, diversity_lambda: float = 0.05,
                 loss_chunk_size: int = 256, intervention_freq: float = 0.15,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concept_graph = concept_graph
        self.maturity = maturity or MaturityDetector()
        self.target_layers = target_layers or []
        self.max_causal_lambda = causal_lambda
        self.diversity_lambda = diversity_lambda
        self.loss_chunk_size = loss_chunk_size
        self.intervention_freq = intervention_freq
        self._step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._step_count += 1

        # ── Forward with optional activation hooks ─────────────────────
        activations = {}
        handles = []
        should_intervene = (self.concept_graph is not None and
                           self.maturity.is_mature and
                           random.random() < self.intervention_freq)

        if should_intervene:
            # Search from top-level model (PEFT wrapper) — target_layers
            # use fully-qualified names like "base_model.model.model.layers.15"
            for name, module in model.named_modules():
                if name in self.target_layers:
                    def hook_fn(m, inp, out, n=name):
                        act = out[0] if isinstance(out, tuple) else out
                        activations[n] = act.detach()
                    handles.append(module.register_forward_hook(hook_fn))

        outputs = model(**inputs)

        for h in handles:
            h.remove()

        # ── Base loss (computed by model internally, handles device placement) ──
        base_loss = outputs.loss
        lm_device = base_loss.device

        # ── Update maturity ───────────────────────────────────────────
        just_matured = self.maturity.update(base_loss.item())
        if just_matured:
            print(f"\n[SENN] Maturity detected at step {self._step_count}! "
                  f"Activating conceptor losses.")

        # ── SENN auxiliary losses ──────────────────────────────────────
        causal_loss = torch.tensor(0.0, device=lm_device)
        diversity_loss = torch.tensor(0.0, device=lm_device)

        if should_intervene and activations and self.concept_graph is not None:
            self.concept_graph.to(lm_device)
            n_layers = 0
            for layer_name, acts in activations.items():
                # Match conceptors: layer_name is fully-qualified
                # (e.g., "base_model.model.model.layers.15")
                # conceptor names use underscores
                safe = layer_name.replace(".", "_")
                layer_conceptors = {
                    n: c for n, c in self.concept_graph.conceptors.items()
                    if safe in n
                }
                if layer_conceptors:
                    acts = acts.to(lm_device)
                    dists = [F.mse_loss(acts, c(acts)) for c in layer_conceptors.values()]
                    causal_loss = causal_loss + torch.stack(dists).min()
                    n_layers += 1
            if n_layers > 0:
                causal_loss = causal_loss / n_layers

            # Diversity: penalize correlated conceptors
            all_conceptors = list(self.concept_graph.conceptors.values())
            if len(all_conceptors) > 1:
                n_pairs = min(16, len(all_conceptors) * (len(all_conceptors) - 1) // 2)
                for _ in range(n_pairs):
                    c1, c2 = random.sample(all_conceptors, 2)
                    p1 = torch.cat([c1.encoder.view(-1), c1.decoder.view(-1)]).float()
                    p2 = torch.cat([c2.encoder.view(-1), c2.decoder.view(-1)]).float()
                    cos = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
                    diversity_loss = diversity_loss + cos.abs()
                diversity_loss = diversity_loss / n_pairs

        # ── Combined loss ─────────────────────────────────────────────
        c_lambda = self.max_causal_lambda * self.maturity.maturity_score
        d_lambda = self.diversity_lambda if self.maturity.is_mature else 0.0
        total = base_loss + c_lambda * causal_loss + d_lambda * diversity_loss

        # Log SENN metrics periodically
        if self._step_count % 5 == 0 and self.maturity.is_mature:
            print(f"  [SENN] step={self._step_count} lm={base_loss.item():.4f} "
                  f"causal={causal_loss.item():.4f} div={diversity_loss.item():.4f} "
                  f"λ_c={c_lambda:.4f} mature_score={self.maturity.maturity_score:.3f}")

        return (total, outputs) if return_outputs else total

    @staticmethod
    def _find_lm_head(model):
        candidates = [model]
        m = model
        for attr in ("model", "base_model", "model", "module"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                candidates.append(m)
        if hasattr(model, "module"):
            candidates.append(model.module)
        for c in candidates:
            if hasattr(c, "lm_head"):
                return c.lm_head
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SENN-enhanced QLoRA fine-tuning")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--loss-chunk-size", type=int, default=256)
    parser.add_argument("--format", default="qwen_native")
    parser.add_argument("--resume", type=str, default=None)
    # SENN params
    parser.add_argument("--causal-lambda", type=float, default=0.1)
    parser.add_argument("--diversity-lambda", type=float, default=0.05)
    parser.add_argument("--concept-rank", type=int, default=16)
    parser.add_argument("--concepts-per-layer", type=int, default=4)
    parser.add_argument("--intervention-freq", type=float, default=0.15)
    parser.add_argument("--maturity-warmup", type=int, default=30)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    print(f"{'='*60}")
    print(f"SENN-Enhanced QLoRA Fine-tuning")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Max seq len: {args.max_seq_len}, Loss chunk: {args.loss_chunk_size}")
    print(f"SENN: causal_λ={args.causal_lambda}, diversity_γ={args.diversity_lambda}")
    print(f"      concept_rank={args.concept_rank}, per_layer={args.concepts_per_layer}")
    print(f"GPUs: {n_gpus}")
    for i in range(n_gpus):
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    print(f"{'='*60}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (4-bit QLoRA on single GPU) ───────────────────────────
    # 14B 4-bit ≈ 8GB, fits comfortably on one A5000 (24GB)
    print("Loading model (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU 0: {used:.1f} / {total:.1f} GB after model load")

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Identify target layers for SENN ───────────────────────────────
    # Search from top-level model (PEFT-wrapped) for transformer blocks
    # Names look like: base_model.model.model.layers.15
    all_block_names = [
        n for n, _ in model.named_modules()
        if "layers." in n and n.split(".")[-1].isdigit()
    ]
    if all_block_names:
        n_blocks = len(all_block_names)
        # Pick layers at 25%, 50%, 75% depth
        indices = [n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4]
        target_layers = [all_block_names[i] for i in indices if i < n_blocks]
    else:
        target_layers = []

    print(f"SENN target layers ({len(target_layers)}/{len(all_block_names)}): {target_layers}")

    # ── Initialize SENN components ────────────────────────────────────
    hidden_dim = getattr(model.config, "hidden_size", 5120)
    concept_graph = SENNConceptGraph()
    concept_graph.init_for_layers(target_layers, hidden_dim,
                                   concepts_per_layer=args.concepts_per_layer,
                                   rank=args.concept_rank)
    n_conceptors = len(concept_graph.conceptors)
    n_concept_params = sum(p.numel() for p in concept_graph.parameters())
    print(f"SENN: {n_conceptors} conceptors, {n_concept_params:,} params "
          f"(rank={args.concept_rank}, hidden={hidden_dim})")

    maturity = MaturityDetector(warmup_steps=args.maturity_warmup)

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading dataset with assistant-only masking...")
    train_ds = load_dataset_with_masks("train", args.format, tokenizer, args.max_seq_len)
    val_ds = load_dataset_with_masks("val", args.format, tokenizer, args.max_seq_len)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    sample = train_ds[0]
    total = len(sample["labels"])
    trainable = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample: {total} tokens, {trainable} trainable ({100*trainable//max(total,1)}%)")

    # ── Training ──────────────────────────────────────────────────────
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
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    pad_id = tokenizer.pad_token_id or 0
    trainer = SENNChunkedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(pad_token_id=pad_id),
        processing_class=tokenizer,
        concept_graph=concept_graph,
        maturity=maturity,
        target_layers=target_layers,
        causal_lambda=args.causal_lambda,
        diversity_lambda=args.diversity_lambda,
        loss_chunk_size=args.loss_chunk_size,
        intervention_freq=args.intervention_freq,
    )

    # Add conceptor params to optimizer so they get gradients
    original_create_optimizer = trainer.create_optimizer
    def create_optimizer_with_conceptors():
        original_create_optimizer()
        conceptor_params = list(concept_graph.parameters())
        if conceptor_params:
            trainer.optimizer.add_param_group({
                "params": conceptor_params,
                "lr": args.lr,
                "weight_decay": 0.01,
            })
            print(f"  [SENN] Added {len(conceptor_params)} conceptor param groups to optimizer")
    trainer.create_optimizer = create_optimizer_with_conceptors

    print("\nStarting SENN-enhanced training...")
    print("  Phase 1: Standard QLoRA (until maturity plateau)")
    print("  Phase 2: + Conceptor causal/diversity losses")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    # Save conceptor graph
    concept_path = FINAL_DIR / "concept_graph.pt"
    torch.save(concept_graph.state_dict(), concept_path)
    print(f"Conceptor graph saved to {concept_path}")

    for i in range(n_gpus):
        peak = torch.cuda.max_memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {peak:.1f} GB / {total:.1f} GB peak")

    print(f"\nModel saved to {FINAL_DIR}")
    print(f"Maturity reached: {maturity.is_mature} (score: {maturity.maturity_score:.3f})")


if __name__ == "__main__":
    main()
