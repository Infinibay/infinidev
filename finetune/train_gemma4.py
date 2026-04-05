#!/usr/bin/env python3
"""Fine-tune Gemma 4 26B-A4B (MoE) with Unsloth — single GPU, QLoRA.

Optimized for Gemma 4's native tool-calling format. Uses train_on_responses_only
to mask user/system/tool_response turns so the model only learns from its own
tool_call and text outputs.

The 26B-A4B is a Mixture-of-Experts model (~4B active per token). Conservative
LoRA settings (r=8) are used by default to avoid destabilizing the expert router.

Usage:
    # Default: 26B-A4B on GPU 0
    python finetune/train_gemma4.py

    # Bare format (no system prompt at all)
    python finetune/train_gemma4.py --format gemma4_bare

    # E4B variant (smaller, faster iteration)
    python finetune/train_gemma4.py --model unsloth/gemma-4-E4B-it --lora-r 16 --max-seq-len 8192
"""

import argparse
import json
import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "output" / "dataset"
OUTPUT_DIR = BASE_DIR / "output" / "model_gemma4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/gemma-4-26B-A4B-it")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--format", default="gemma4",
                        choices=["gemma4", "gemma4_bare"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    # ── 1. Load model ───────────────────────────────────────────────────
    print(f"Loading {args.model} on GPU {args.gpu}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        device_map={"": args.gpu},
    )

    mem_used = torch.cuda.memory_allocated(args.gpu) / 1024**3
    mem_total = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
    print(f"  Model loaded: {mem_used:.1f} GB / {mem_total:.1f} GB")

    # ── 2. Apply LoRA ────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_len,
    )

    # ── 3. Load dataset ──────────────────────────────────────────────────
    train_path = DATASET_DIR / f"train_{args.format}.jsonl"
    val_path = DATASET_DIR / f"val_{args.format}.jsonl"

    print(f"Loading dataset from {train_path}...")
    train_data = []
    with open(train_path) as f:
        for line in f:
            item = json.loads(line)
            train_data.append({"text": item["text"]})

    val_data = []
    with open(val_path) as f:
        for line in f:
            item = json.loads(line)
            val_data.append({"text": item["text"]})

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── 4. Trainer ───────────────────────────────────────────────────────
    from trl import SFTTrainer, SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            save_steps=25,
            save_total_limit=3,
            eval_strategy="no",
            report_to="none",
            seed=42,
        ),
    )

    # ── 5. Train only on model responses ─────────────────────────────────
    # Gemma 4 uses <|turn>model\n as the response marker
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    # Verify masking
    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    total = len(labels)
    trainable = sum(1 for l in labels if l != -100)
    print(f"Sample: {total} tokens, {trainable} trainable ({100*trainable//max(total,1)}%)")

    # ── 6. Train ─────────────────────────────────────────────────────────
    print("Starting training...")
    stats = trainer.train()

    for i in range(torch.cuda.device_count()):
        peak = round(torch.cuda.max_memory_reserved(i) / 1024**3, 2)
        total_mem = round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
        print(f"GPU {i}: {peak} GB / {total_mem} GB peak")

    # ── 7. Save LoRA adapter ─────────────────────────────────────────────
    model.save_pretrained(str(OUTPUT_DIR / "lora"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora"))
    print(f"LoRA adapter saved to {OUTPUT_DIR / 'lora'}")

    # ── 8. Export to GGUF for Ollama ─────────────────────────────────────
    print("Saving GGUF (q4_k_m) for Ollama...")
    gguf_dir = OUTPUT_DIR / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"GGUF saved to {gguf_dir}")
    print(f"\nTo import to Ollama:")
    print(f"  ollama create infinidev-gemma4 -f {gguf_dir}/Modelfile")


if __name__ == "__main__":
    main()
