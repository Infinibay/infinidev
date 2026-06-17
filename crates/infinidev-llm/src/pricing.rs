//! Model pricing + per-call cost.
//!
//! Prices are USD per 1,000,000 tokens. This is *maintained data* — public
//! rates change and new models appear; update the table below. Local providers
//! (Ollama, llama.cpp, vLLM) have no entry and cost nothing.
//!
//! Matching is by longest substring of the (lowercased) bare model id, so
//! `gpt-4o-mini` wins over `gpt-4o`, and dated/versioned ids
//! (`claude-3-5-sonnet-20241022`) match their family prefix.

use crate::provider;
use crate::types::Usage;

#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    /// USD per 1M input (prompt) tokens.
    pub input: f64,
    /// USD per 1M output (completion) tokens.
    pub output: f64,
    /// USD per 1M cached-read input tokens, if the provider prices them.
    pub cache_read: Option<f64>,
    /// USD per 1M cache-write input tokens, if priced.
    pub cache_write: Option<f64>,
}

impl Pricing {
    const fn io(input: f64, output: f64) -> Self {
        Self { input, output, cache_read: None, cache_write: None }
    }
    const fn cached(input: f64, output: f64, cache_read: f64, cache_write: f64) -> Self {
        Self { input, output, cache_read: Some(cache_read), cache_write: Some(cache_write) }
    }
}

/// (model-id substring, pricing). Order doesn't matter; lookup picks the
/// longest matching pattern.
static TABLE: &[(&str, Pricing)] = &[
    // ── OpenAI ───────────────────────────────────────────────
    ("gpt-4o-mini", Pricing::io(0.15, 0.60)),
    ("gpt-4o", Pricing::io(2.50, 10.00)),
    ("gpt-4.1-mini", Pricing::io(0.40, 1.60)),
    ("gpt-4.1-nano", Pricing::io(0.10, 0.40)),
    ("gpt-4.1", Pricing::io(2.00, 8.00)),
    ("o3-mini", Pricing::io(1.10, 4.40)),
    ("o4-mini", Pricing::io(1.10, 4.40)),
    ("o3", Pricing::io(2.00, 8.00)),
    // ── Anthropic ────────────────────────────────────────────
    ("claude-3-5-haiku", Pricing::cached(0.80, 4.00, 0.08, 1.00)),
    ("claude-3-5-sonnet", Pricing::cached(3.00, 15.00, 0.30, 3.75)),
    ("claude-3-opus", Pricing::cached(15.00, 75.00, 1.50, 18.75)),
    ("claude-haiku", Pricing::cached(0.80, 4.00, 0.08, 1.00)),
    ("claude-sonnet", Pricing::cached(3.00, 15.00, 0.30, 3.75)),
    ("claude-opus", Pricing::cached(15.00, 75.00, 1.50, 18.75)),
    // ── Google Gemini ────────────────────────────────────────
    ("gemini-1.5-flash", Pricing::io(0.075, 0.30)),
    ("gemini-1.5-pro", Pricing::io(1.25, 5.00)),
    ("gemini-2.5-flash-lite", Pricing::io(0.10, 0.40)),
    ("gemini-2.5-flash", Pricing::io(0.30, 2.50)),
    ("gemini-2.5-pro", Pricing::io(1.25, 10.00)),
    // ── DeepSeek ─────────────────────────────────────────────
    ("deepseek-reasoner", Pricing::cached(0.55, 2.19, 0.14, 0.0)),
    ("deepseek-chat", Pricing::cached(0.27, 1.10, 0.07, 0.0)),
    // ── MiniMax (exact figures from infinidev's llm.py) ──────
    ("minimax-m2.7", Pricing::cached(0.30, 1.20, 0.03, 0.375)),
    ("minimax-m2", Pricing::cached(0.30, 1.20, 0.03, 0.375)),
    // ── Mistral ──────────────────────────────────────────────
    ("mistral-large", Pricing::io(2.00, 6.00)),
    ("mistral-small", Pricing::io(0.20, 0.60)),
    ("codestral", Pricing::io(0.30, 0.90)),
    // ── Qwen (DashScope, approximate) ────────────────────────
    ("qwen-max", Pricing::io(1.60, 6.40)),
    ("qwen-plus", Pricing::io(0.40, 1.20)),
    ("qwen-turbo", Pricing::io(0.05, 0.20)),
];

/// Look up pricing for a model id (prefix-stripped, case-insensitive),
/// choosing the most specific (longest) matching pattern.
pub fn pricing_for(model: &str) -> Option<Pricing> {
    let bare = provider::bare_model(model).to_lowercase();
    TABLE
        .iter()
        .filter(|(pat, _)| bare.contains(pat))
        .max_by_key(|(pat, _)| pat.len())
        .map(|(_, p)| *p)
}

/// Computed cost of one call, in USD, broken down by token class.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cost {
    pub input_usd: f64,
    pub output_usd: f64,
    pub cache_read_usd: f64,
    pub total_usd: f64,
}

/// Compute the USD cost of a call. Returns `None` for models with no known
/// pricing (e.g. local Ollama models), which the caller should treat as free.
pub fn cost(model: &str, usage: &Usage) -> Option<Cost> {
    let p = pricing_for(model)?;
    const M: f64 = 1_000_000.0;

    let cache_read = usage.cache_read_tokens.unwrap_or(0);
    let billable_input = usage.prompt_tokens.saturating_sub(cache_read);

    let input_usd = billable_input as f64 / M * p.input;
    let output_usd = usage.completion_tokens as f64 / M * p.output;
    let cache_read_usd = match p.cache_read {
        Some(rate) => cache_read as f64 / M * rate,
        None => 0.0,
    };
    let total_usd = input_usd + output_usd + cache_read_usd;
    Some(Cost { input_usd, output_usd, cache_read_usd, total_usd })
}
