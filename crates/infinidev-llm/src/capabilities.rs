//! Per-provider / per-model capability matrix — ported from
//! `src/infinidev/config/model_capabilities.py`. These flags drive how the
//! engine builds requests (native vs manual tool-calling, schema sanitization,
//! whether to ask for `tool_choice: required`, reasoning handling).

#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub function_calling: bool,
    pub tool_choice_required: bool,
    pub json_mode: bool,
    pub vision: bool,
    /// Emits reasoning as a separate `reasoning_content` field / `<think>` block.
    pub thinking_sections: bool,
    /// Rejects `anyOf` / complex JSON schemas (e.g. Qwen) — schemas must be flattened.
    pub needs_schema_sanitization: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        // Optimistic defaults, matching the Python dataclass.
        Self {
            function_calling: true,
            tool_choice_required: true,
            json_mode: true,
            vision: false,
            thinking_sections: false,
            needs_schema_sanitization: false,
        }
    }
}

impl Capabilities {
    const fn new(fc: bool, tcr: bool, json: bool) -> Self {
        Self {
            function_calling: fc,
            tool_choice_required: tcr,
            json_mode: json,
            vision: false,
            thinking_sections: false,
            needs_schema_sanitization: false,
        }
    }
}

/// Conservative preset for an unknown provider.
fn conservative() -> Capabilities {
    Capabilities {
        function_calling: true,
        tool_choice_required: false,
        json_mode: false,
        ..Default::default()
    }
}

/// Provider-level preset before model-name refinement.
pub fn preset(provider_id: &str) -> Capabilities {
    match provider_id {
        "openai" | "anthropic" | "gemini" | "mistral" | "zai" | "zai_coding" | "kimi"
        | "openrouter" => Capabilities::new(true, true, true),
        "minimax" => Capabilities {
            thinking_sections: true, // MiniMax M2.x emits reasoning_content
            ..Capabilities::new(true, false, true)
        },
        "qwen" => Capabilities {
            needs_schema_sanitization: true, // Qwen rejects anyOf / complex schemas
            ..Capabilities::new(true, true, true)
        },
        // Local / self-hosted: be conservative — tool_choice:required and
        // json_mode vary by the loaded model.
        "ollama" | "llama_cpp" | "vllm" | "openai_compatible" => conservative(),
        _ => conservative(),
    }
}

/// Full capabilities for a (provider, model) pair: the provider preset refined
/// with model-name heuristics for vision and reasoning.
pub fn for_model(provider_id: &str, model: &str) -> Capabilities {
    let mut caps = preset(provider_id);
    let m = model.to_lowercase();

    // Vision-capable families.
    if ["gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4", "claude", "gemini", "pixtral", "llava", "qwen-vl", "qwen3-vl"]
        .iter()
        .any(|k| m.contains(k))
    {
        caps.vision = true;
    }

    // Reasoning / thinking families.
    if ["deepseek-r", "-thinking", "o1", "o3", "o4", "magistral", "qwq", "glm-4.7", "minimax"]
        .iter()
        .any(|k| m.contains(k))
    {
        caps.thinking_sections = true;
    }

    caps
}
