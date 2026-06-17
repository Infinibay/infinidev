//! Live model discovery per provider — ported from the Python `fetch_models`.
//! Ollama (`/api/tags`), OpenAI-style (`/models`), Anthropic (`/v1/models`) and
//! Gemini (`/v1/models?key=`) are all queried live. Anthropic/Gemini fall back
//! to their curated static list only if the live call fails (no key, offline),
//! so the picker is never empty. `Static`/`FreeText` providers have no list API.

use serde::Deserialize;

use crate::error::{LlmError, Result};
use crate::provider::{self, ListFormat, Provider};

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
    // Every provider with a real model-list endpoint is queried live; the
    // curated static list is only a fallback when the call fails (no key,
    // offline, unreachable server) so the picker is never empty. This mirrors
    // the Python `fetch_models` (live first, static on error). `Static` /
    // `FreeText` providers have no list API.
    match p.list_format {
        // Local servers: surface the connection error rather than masking a
        // down server with a (nonexistent) static list.
        ListFormat::Ollama => fetch_ollama(base).await,
        ListFormat::OpenAi => Ok(or_static(fetch_openai(base, api_key).await, p)),
        ListFormat::Anthropic => Ok(or_static(fetch_anthropic(base, api_key).await, p)),
        ListFormat::Gemini => Ok(or_static(fetch_gemini(base, api_key).await, p)),
        _ => Ok(static_list(p)),
    }
}

fn static_list(p: &Provider) -> Vec<String> {
    p.static_models.iter().map(|s| s.to_string()).collect()
}

/// Use the live result, or fall back to the provider's static list on failure.
fn or_static(live: Result<Vec<String>>, p: &Provider) -> Vec<String> {
    match live {
        Ok(models) if !models.is_empty() => models,
        _ => static_list(p),
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

/// Anthropic Messages API: `GET {base}/v1/models` with `x-api-key` +
/// `anthropic-version` headers → `data[].id`.
async fn fetch_anthropic(base: &str, api_key: Option<String>) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Models {
        #[serde(default)]
        data: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        id: String,
    }
    let url = format!("{base}/v1/models");
    let mut rb = reqwest::Client::new()
        .get(url)
        .header("anthropic-version", "2023-06-01");
    if let Some(k) = api_key {
        rb = rb.header("x-api-key", k);
    }
    let resp = rb.send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(LlmError::Api { status: status.as_u16(), message: "could not list Anthropic models".into() });
    }
    let m: Models = resp.json().await?;
    Ok(m.data.into_iter().map(|e| e.id).collect())
}

/// Gemini: `GET {base}/v1/models?key=<api_key>` → `models[].name`, stripping the
/// `models/` prefix the API returns.
async fn fetch_gemini(base: &str, api_key: Option<String>) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct Models {
        #[serde(default)]
        models: Vec<Entry>,
    }
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default)]
        name: String,
    }
    let mut rb = reqwest::Client::new().get(format!("{base}/v1/models"));
    if let Some(k) = api_key {
        rb = rb.query(&[("key", k)]);
    }
    let resp = rb.send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(LlmError::Api { status: status.as_u16(), message: "could not list Gemini models".into() });
    }
    let m: Models = resp.json().await?;
    Ok(m
        .models
        .into_iter()
        .filter_map(|e| {
            let name = e.name.strip_prefix("models/").unwrap_or(&e.name);
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failed (or empty) live fetch falls back to the provider's curated
    /// static list — so the picker is never empty offline / without a key.
    #[test]
    fn or_static_falls_back_on_error_and_empty() {
        let p = provider::get("anthropic").unwrap();
        let err: Result<Vec<String>> = Err(LlmError::Api { status: 401, message: "no key".into() });
        assert_eq!(or_static(err, p), static_list(p));
        assert!(!static_list(p).is_empty());
        // An empty live result also triggers the fallback.
        assert_eq!(or_static(Ok(vec![]), p), static_list(p));
        // A non-empty live result is used as-is (live wins over static).
        assert_eq!(or_static(Ok(vec!["live-model".into()]), p), vec!["live-model".to_string()]);
    }

    /// Every provider either discovers live or is intentionally free-text;
    /// none serves a static list as its *primary* source. (Minimax moved to
    /// live `/v1/models` discovery; only `Static`/`FreeText` remain non-live,
    /// and `Static` is reserved for providers without a list endpoint.)
    #[test]
    fn no_provider_is_static_primary() {
        for p in provider::PROVIDERS {
            let live = matches!(
                p.list_format,
                ListFormat::Ollama | ListFormat::OpenAi | ListFormat::Anthropic | ListFormat::Gemini
            );
            let free_text = matches!(p.list_format, ListFormat::FreeText);
            assert!(
                live || free_text,
                "provider {} still uses a hardcoded static model list as its primary source",
                p.id
            );
        }
    }
}
