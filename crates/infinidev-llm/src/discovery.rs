//! Live model discovery per provider — ported from the Python `fetch_models`.
//! Ollama (`/api/tags`) and OpenAI-style (`/models`) are queried live; other
//! providers fall back to their static model list.

use serde::Deserialize;

use crate::error::{LlmError, Result};
use crate::provider::{self, ListFormat};

/// List available models for a provider. Returns bare model ids (what the
/// chat API expects), not LiteLLM-prefixed names.
pub async fn list_models(
    provider_id: &str,
    api_key: Option<String>,
    base_url: &str,
) -> Result<Vec<String>> {
    let p = provider::get(provider_id)
        .ok_or_else(|| LlmError::Config(format!("unknown provider: {provider_id}")))?;
    let base = if base_url.is_empty() { p.default_base_url } else { base_url };
    let base = base.trim_end_matches('/');
    match p.list_format {
        ListFormat::Ollama => fetch_ollama(base).await,
        ListFormat::OpenAi => fetch_openai(base, api_key).await,
        _ => Ok(p.static_models.iter().map(|s| s.to_string()).collect()),
    }
}

async fn fetch_ollama(base: &str) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Tags {
        #[serde(default)]
        models: Vec<Model>,
    }
    #[derive(Deserialize)]
    struct Model {
        name: String,
    }
    let url = format!("{base}/api/tags");
    let resp = reqwest::Client::new().get(url).send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(LlmError::Api { status: status.as_u16(), message: "could not list Ollama models".into() });
    }
    let tags: Tags = resp.json().await?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

async fn fetch_openai(base: &str, api_key: Option<String>) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Models {
        #[serde(default)]
        data: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        id: String,
    }
    let url = format!("{base}/models");
    let mut rb = reqwest::Client::new().get(url);
    if let Some(k) = api_key {
        rb = rb.bearer_auth(k);
    }
    let resp = rb.send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(LlmError::Api { status: status.as_u16(), message: "could not list models".into() });
    }
    let m: Models = resp.json().await?;
    Ok(m.data.into_iter().map(|e| e.id).collect())
}
