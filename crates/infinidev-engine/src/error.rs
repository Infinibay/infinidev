use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Llm(#[from] infinidev_llm::LlmError),

    #[error(transparent)]
    Tool(#[from] infinidev_tools::ToolError),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
