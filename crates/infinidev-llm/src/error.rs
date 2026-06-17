use thiserror::Error;

/// Errors surfaced by the LLM layer.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("provider API error (HTTP {status}): {message}")]
    Api { status: u16, message: String },

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("streaming error: {0}")]
    Stream(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("{0} is not implemented yet")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, LlmError>;
