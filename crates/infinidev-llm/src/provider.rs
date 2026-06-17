//! Provider registry — ported from `src/infinidev/config/providers.py`.
//!
//! The chat wire protocol (`Wire`) is the key abstraction: 15 providers
//! collapse to a few real dialects. Most are OpenAI-compatible; Anthropic and
//! Gemini have their own request/response shapes.

/// The chat-completions wire protocol a provider speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    /// `POST {base}/chat/completions` — OpenAI schema. Covers OpenAI, Ollama
    /// (via `/v1`), vLLM, llama.cpp, Mistral, Z.AI, Kimi, MiniMax, OpenRouter,
    /// Qwen (DashScope compat), and any OpenAI-compatible endpoint.
    OpenAiCompat,
    /// `POST {base}/v1/messages` — Anthropic Messages API.
    Anthropic,
    /// Google Gemini `generateContent`.
    Gemini,
}

/// How a provider's available models are discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListFormat {
    Ollama,
    OpenAi,
    Anthropic,
    Gemini,
    Static,
    FreeText,
}

/// Configuration for a single LLM provider.
#[derive(Debug, Clone)]
pub struct Provider {
    pub id: &'static str,
    pub display_name: &'static str,
    /// LiteLLM-style model prefix (kept for parity / model-id round-tripping).
    pub prefix: &'static str,
    pub default_base_url: &'static str,
    pub api_key_required: bool,
    pub base_url_editable: bool,
    pub wire: Wire,
    pub list_format: ListFormat,
    /// True when LiteLLM handled the endpoint natively (informational here).
    pub is_native: bool,
    pub static_models: &'static [&'static str],
}

impl Provider {
    /// The base URL to POST chat requests against, given a (possibly
    /// user-overridden) configured base. Ollama needs `/v1` appended for its
    /// OpenAI-compatible endpoint.
    pub fn chat_base_url(&self, configured: &str) -> String {
        let base = if configured.is_empty() {
            self.default_base_url
        } else {
            configured
        };
        let base = base.trim_end_matches('/');
        if self.id == "ollama" && !base.contains("/v1") {
            format!("{base}/v1")
        } else {
            base.to_string()
        }
    }
}

/// The provider registry. 15 providers, faithful to the Python config.
pub static PROVIDERS: &[Provider] = &[
    Provider {
        id: "ollama",
        display_name: "Ollama (Local)",
        prefix: "ollama_chat/",
        default_base_url: "http://localhost:11434",
        api_key_required: false,
        base_url_editable: true,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::Ollama,
        is_native: false,
        static_models: &[],
    },
    Provider {
        id: "openai",
        display_name: "OpenAI",
        prefix: "openai/",
        default_base_url: "https://api.openai.com/v1",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: true,
        static_models: &["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "o3", "o3-pro", "o3-mini", "o4-mini"],
    },
    Provider {
        id: "anthropic",
        display_name: "Claude (Anthropic)",
        prefix: "anthropic/",
        default_base_url: "https://api.anthropic.com",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::Anthropic,
        list_format: ListFormat::Anthropic,
        is_native: true,
        static_models: &[
            "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101",
            "claude-sonnet-4-0", "claude-opus-4-0",
        ],
    },
    Provider {
        id: "gemini",
        display_name: "Gemini (Google)",
        prefix: "gemini/",
        default_base_url: "https://generativelanguage.googleapis.com",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::Gemini,
        list_format: ListFormat::Gemini,
        is_native: true,
        static_models: &[
            "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ],
    },
    Provider {
        id: "zai",
        display_name: "Z.AI (Zhipu/GLM)",
        prefix: "zai/",
        default_base_url: "https://api.z.ai/api/paas/v4",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: true,
        static_models: &["glm-5", "glm-5-turbo", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-flash", "glm-4.5-air"],
    },
    Provider {
        id: "zai_coding",
        display_name: "Z.AI Coding Plan",
        prefix: "zai/",
        default_base_url: "https://api.z.ai/api/coding/paas/v4",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: false,
        static_models: &["glm-5", "glm-5-turbo", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-flash", "glm-4.5-air"],
    },
    Provider {
        id: "kimi",
        display_name: "Kimi (Moonshot)",
        prefix: "moonshot/",
        default_base_url: "https://api.moonshot.ai/v1",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: false,
        static_models: &["kimi-k2.5", "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2-0905-preview", "kimi-k2-turbo-preview"],
    },
    Provider {
        id: "minimax",
        display_name: "Minimax",
        prefix: "minimax/",
        default_base_url: "https://api.minimax.io/v1",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        // OpenAI-compatible `/v1/models`; the static list is only a fallback if
        // the live call fails (no key / offline).
        list_format: ListFormat::OpenAi,
        is_native: false,
        static_models: &["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5", "MiniMax-M2.1"],
    },
    Provider {
        id: "mistral",
        display_name: "Mistral (La Plateforme)",
        prefix: "mistral/",
        default_base_url: "https://api.mistral.ai/v1",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: true,
        static_models: &[
            "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
            "ministral-3b-latest", "ministral-8b-latest", "magistral-medium-latest",
            "magistral-small-latest", "codestral-latest", "devstral-medium-latest",
            "devstral-small-latest", "pixtral-large-latest",
        ],
    },
    Provider {
        id: "openrouter",
        display_name: "OpenRouter",
        prefix: "openrouter/",
        default_base_url: "https://openrouter.ai/api/v1",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: true,
        static_models: &[],
    },
    Provider {
        id: "llama_cpp",
        display_name: "llama.cpp Server",
        prefix: "custom_openai/",
        default_base_url: "http://localhost:8080/v1",
        api_key_required: false,
        base_url_editable: true,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::FreeText,
        is_native: false,
        static_models: &[],
    },
    Provider {
        id: "vllm",
        display_name: "vLLM Server",
        prefix: "custom_openai/",
        default_base_url: "http://localhost:8000/v1",
        api_key_required: false,
        base_url_editable: true,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: false,
        static_models: &[],
    },
    Provider {
        id: "openai_compatible",
        display_name: "OpenAI Compatible",
        prefix: "custom_openai/",
        default_base_url: "",
        api_key_required: true,
        base_url_editable: true,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::FreeText,
        is_native: false,
        static_models: &[],
    },
    Provider {
        id: "qwen",
        display_name: "Qwen (Alibaba)",
        prefix: "custom_openai/",
        default_base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_required: true,
        base_url_editable: false,
        wire: Wire::OpenAiCompat,
        list_format: ListFormat::OpenAi,
        is_native: false,
        static_models: &[
            "qwen3.6-plus", "qwen3.5-plus", "qwen3.5-flash", "qwen3.5-397b-a17b",
            "qwen3.5-122b-a10b", "qwen3-max", "qwen3-coder-plus", "qwen3-coder-flash",
            "qwen3-235b-a22b", "qwen3-32b", "qwen3-30b-a3b", "qwen-max", "qwen-plus",
            "qwen-turbo", "qwen-flash", "qwq-plus",
        ],
    },
];

/// All known LiteLLM-style prefixes, longest first so `custom_openai/` wins
/// over a hypothetical `openai/` substring match.
const PREFIXES: &[&str] = &[
    "ollama_chat/", "custom_openai/", "openrouter/", "anthropic/", "moonshot/",
    "minimax/", "mistral/", "gemini/", "openai/", "ollama/", "zai/",
];

/// Look up a provider by id.
pub fn get(id: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// Strip a known provider prefix from a model id, returning the bare model
/// name the provider's API expects (`ollama_chat/gemma:7b` → `gemma:7b`).
pub fn bare_model(model: &str) -> &str {
    // `ollama/` → treat like `ollama_chat/` for stripping purposes.
    for p in PREFIXES {
        if let Some(rest) = model.strip_prefix(p) {
            return rest;
        }
    }
    model
}

/// Normalize a model id the way the Python layer does: `ollama/` → `ollama_chat/`
/// so the chat (not generate) endpoint is used.
pub fn normalize_model(model: &str) -> String {
    if let Some(rest) = model.strip_prefix("ollama/") {
        format!("ollama_chat/{rest}")
    } else {
        model.to_string()
    }
}
