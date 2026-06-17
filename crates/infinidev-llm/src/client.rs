use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

/// A chat LLM endpoint. Implementations translate the provider-agnostic
/// [`ChatRequest`] to/from their wire dialect.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// A short label for diagnostics (e.g. "openai-compat").
    fn wire_name(&self) -> &'static str;

    /// One-shot completion.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse>;

    /// Streaming completion — a sequence of incremental deltas.
    async fn chat_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk>>>;
}
