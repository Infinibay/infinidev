"""Run a >100B dense coding LLM via AirLLM (layer-by-layer inference).

AirLLM loads transformer layers one at a time from disk, so even a 400B
dense model fits in a few GB of VRAM at the cost of throughput (<1 tok/s).

Only DENSE transformers are listed here — AirLLM has no expert-level
sharding for MoE, so giants like Qwen3-480B, GLM-4.5, Kimi-K2, and
DeepSeek-V3 end up paying the full-dense I/O cost with no benefit.

Install:
    uv pip install airllm bitsandbytes accelerate

Example:
    uv run python scripts/airllm_run.py \\
        --model mistral-large \\
        --prompt "Write a Python LRU cache with O(1) get/put." \\
        --compression 4bit
"""
from __future__ import annotations

import argparse
import sys
import time

# Dense (non-MoE) models >100B that AirLLM can actually stream layer-by-layer.
# Rough numbers assume 4-bit compression on an NVMe consumer GPU box.
PRESETS: dict[str, str] = {
    # ~123B dense. Top-tier at code. ~62 GB on disk. ~0.3-1 tok/s. SWEET SPOT.
    "mistral-large": "mistralai/Mistral-Large-Instruct-2411",
    # ~104B dense. Strong generalist, decent at code. ~52 GB on disk.
    "command-r-plus": "CohereForAI/c4ai-command-r-plus",
    # ~180B dense. Older, weaker at code, but runs. ~90 GB on disk.
    "falcon-180b":   "tiiuae/falcon-180B-chat",
    # ~405B dense. Maximum realistic size. ~200 GB on disk. ~0.1-0.3 tok/s.
    "llama-405b":    "meta-llama/Llama-3.1-405B-Instruct",
}

DEFAULT_SYSTEM = (
    "You are a senior software engineer. Reply with concise, correct code. "
    "Prefer standard library solutions. Include a one-line complexity note."
)


def build_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="AirLLM runner for big coding models.")
    ap.add_argument("--model", default="qwen",
                    help=f"HF repo id, or one of {list(PRESETS)} (default: qwen)")
    ap.add_argument("--prompt", required=True, help="User prompt / coding task.")
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--compression", choices=["none", "4bit", "8bit"], default="4bit",
                    help="Weight compression. 4bit fits more but slightly less accurate.")
    ap.add_argument("--layer-shards", default=None,
                    help="Optional directory to cache per-layer shards (speeds up reruns).")
    args = ap.parse_args()

    model_id = PRESETS.get(args.model, args.model)
    print(f"[airllm] loading {model_id} (compression={args.compression}) ...", flush=True)

    try:
        from airllm import AutoModel
    except ImportError:
        print("ERROR: airllm not installed. Run: uv pip install airllm bitsandbytes accelerate",
              file=sys.stderr)
        return 1

    kwargs: dict = {}
    if args.compression != "none":
        kwargs["compression"] = args.compression
    if args.layer_shards:
        kwargs["layer_shards_saving_path"] = args.layer_shards

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, **kwargs)
    print(f"[airllm] ready in {time.perf_counter() - t0:.1f}s", flush=True)

    tok = model.tokenizer
    messages = build_messages(args.system, args.prompt)
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"{args.system}\n\nUser: {args.prompt}\nAssistant:"

    inputs = tok(text, return_tensors="pt", truncation=True, padding=False)
    input_ids = inputs["input_ids"].to("cuda") if hasattr(inputs["input_ids"], "to") else inputs["input_ids"]

    print("[airllm] generating ...", flush=True)
    t0 = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        return_dict_in_generate=False,
    )
    elapsed = time.perf_counter() - t0

    generated = output_ids[0][input_ids.shape[-1]:] if output_ids.ndim == 2 else output_ids
    reply = tok.decode(generated, skip_special_tokens=True)

    n = len(generated)
    print(f"\n--- reply ({n} tok, {elapsed:.1f}s, {n / max(elapsed, 1e-6):.2f} tok/s) ---\n")
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
