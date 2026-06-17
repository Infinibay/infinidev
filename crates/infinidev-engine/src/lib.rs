//! Infinidev agent engine — the tool-calling loop that ties the LLM layer and
//! the tools into an agent, emitting a stream of [`EngineEvent`]s a UI consumes.

pub mod config;
pub mod error;
pub mod event;
pub mod prompt;

mod engine;
mod orchestrator;

pub use config::EngineConfig;
pub use engine::Engine;
pub use error::{EngineError, Result};
pub use event::{EngineEvent, EngineHost};
pub use orchestrator::Orchestrator;

// Re-export so callers (e.g. the Tauri shell) don't need to depend on the
// lower crates directly.
pub use infinidev_llm::{list_models, Message, Provider, Role, PROVIDERS};
pub use infinidev_tools::{
    background_manager, default_tools, search, tools_for, SearchHit, TaskView, ToolContext,
};

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use infinidev_llm::{ChatRequest, ChatResponse, LlmClient, StreamChunk, ToolCallDelta};
    use infinidev_tools::{default_tools, ToolContext};
    use std::sync::Mutex;

    /// A scripted LLM client: each `chat_stream` returns the next queued turn.
    struct MockClient {
        turns: Mutex<std::collections::VecDeque<Vec<StreamChunk>>>,
    }

    #[async_trait]
    impl LlmClient for MockClient {
        fn wire_name(&self) -> &'static str {
            "mock"
        }
        async fn chat(&self, _req: &ChatRequest) -> infinidev_llm::Result<ChatResponse> {
            Ok(ChatResponse::default())
        }
        async fn chat_stream(
            &self,
            _req: &ChatRequest,
        ) -> infinidev_llm::Result<BoxStream<'static, infinidev_llm::Result<StreamChunk>>> {
            let chunks = self.turns.lock().unwrap().pop_front().unwrap_or_default();
            let s = futures::stream::iter(
                chunks.into_iter().map(Ok::<_, infinidev_llm::LlmError>),
            );
            Ok(Box::pin(s))
        }
    }

    #[derive(Default)]
    struct CollectingHost {
        events: Mutex<Vec<EngineEvent>>,
    }
    #[async_trait]
    impl EngineHost for CollectingHost {
        fn emit(&self, event: EngineEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

    fn tool_call_chunk(index: u32, id: &str, name: &str, args: &str) -> StreamChunk {
        StreamChunk {
            tool_call: Some(ToolCallDelta {
                index,
                id: Some(id.to_string()),
                name: Some(name.to_string()),
                arguments_fragment: Some(args.to_string()),
            }),
            ..Default::default()
        }
    }
    fn content_chunk(text: &str) -> StreamChunk {
        StreamChunk { content: Some(text.to_string()), ..Default::default() }
    }

    #[tokio::test]
    async fn loop_executes_tool_then_answers() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());

        // Turn 1: call list_directory. Turn 2: final answer.
        let mut turns = std::collections::VecDeque::new();
        turns.push_back(vec![tool_call_chunk(0, "c1", "list_directory", "{}")]);
        turns.push_back(vec![content_chunk("All done — the directory is empty.")]);

        let client = MockClient { turns: Mutex::new(turns) };
        let engine = Engine::with_client(
            Box::new(client),
            default_tools(),
            ctx,
            EngineConfig::ollama("mock-model"),
        );
        let host = CollectingHost::default();

        let result = engine.run_turn("list the files", &[], &host).await.unwrap();
        assert!(result.contains("All done"));

        let events = host.events.lock().unwrap();
        let has_tool_call = events.iter().any(|e| matches!(e, EngineEvent::ToolCall { name, .. } if name == "list_directory"));
        let has_tool_ok = events.iter().any(|e| matches!(e, EngineEvent::ToolResult { ok: true, name, .. } if name == "list_directory"));
        let has_turn_end = events.iter().any(|e| matches!(e, EngineEvent::TurnEnd { .. }));
        let streamed = events.iter().any(|e| matches!(e, EngineEvent::StreamChunk { chunk } if chunk.contains("All done")));
        assert!(has_tool_call, "expected a ToolCall event");
        assert!(has_tool_ok, "expected a successful ToolResult event");
        assert!(has_turn_end, "expected a TurnEnd event");
        assert!(streamed, "expected the answer to be streamed");
    }

    #[tokio::test]
    async fn loop_creates_a_file_and_emits_file_change() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());

        let mut turns = std::collections::VecDeque::new();
        turns.push_back(vec![tool_call_chunk(
            0,
            "c1",
            "create_file",
            r#"{"path":"hello.txt","content":"hi"}"#,
        )]);
        turns.push_back(vec![content_chunk("Created hello.txt.")]);

        let client = MockClient { turns: Mutex::new(turns) };
        let engine = Engine::with_client(
            Box::new(client),
            default_tools(),
            ctx,
            EngineConfig::ollama("mock-model"),
        );
        let host = CollectingHost::default();

        engine.run_turn("create hello.txt", &[], &host).await.unwrap();

        // The file really exists, and a FileChange event fired.
        assert!(dir.path().join("hello.txt").exists());
        let events = host.events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, EngineEvent::FileChange { action, .. } if action == "create")));
    }

    fn mock_with(turns: Vec<Vec<StreamChunk>>) -> MockClient {
        MockClient { turns: Mutex::new(turns.into_iter().collect()) }
    }

    #[tokio::test]
    async fn orchestrator_chat_agent_responds_without_developing() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());

        // Chat tier: one turn calling `respond`. Developer must never run.
        let chat = mock_with(vec![vec![tool_call_chunk(
            0,
            "c1",
            "respond",
            r#"{"message":"It computes a checksum."}"#,
        )]]);
        // Developer client returns a sentinel that would be obvious if used.
        let dev = mock_with(vec![vec![content_chunk("DEVELOPER RAN (should not happen)")]]);
        let engine = Engine::with_client(
            Box::new(dev),
            default_tools(),
            ctx.clone(),
            EngineConfig::ollama("mock"),
        );
        let orch = Orchestrator::with_parts(
            Box::new(chat),
            infinidev_tools::tools_for(true),
            engine,
            ctx,
            EngineConfig::ollama("mock"),
        );
        let host = CollectingHost::default();

        let result = orch.run_turn("what does foo() do?", &[], &host).await.unwrap();
        assert_eq!(result, "It computes a checksum.");

        let events = host.events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, EngineEvent::Decision { kind, .. } if kind == "respond")));
        assert!(events.iter().any(|e| matches!(e, EngineEvent::TurnEnd { result } if result == "It computes a checksum.")));
        // The developer loop never emitted a StepStart.
        assert!(!events.iter().any(|e| matches!(e, EngineEvent::StepStart { .. })));
    }

    #[tokio::test]
    async fn manual_tool_calling_loop_works() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());

        // Turn 1: a fenced tool_call block (no native tool_calls). Turn 2: final.
        let mut turns = std::collections::VecDeque::new();
        turns.push_back(vec![content_chunk(
            "Listing.\n```tool_call\n{\"name\":\"list_directory\",\"arguments\":{}}\n```",
        )]);
        turns.push_back(vec![content_chunk("All done — empty directory.")]);

        let mut cfg = EngineConfig::ollama("mock");
        cfg.planning = false;
        cfg.force_manual_tools = true;
        let engine = Engine::with_client(
            Box::new(MockClient { turns: Mutex::new(turns) }),
            default_tools(),
            ctx,
            cfg,
        );
        let host = CollectingHost::default();

        let result = engine.run_turn("list the files", &[], &host).await.unwrap();
        assert!(result.contains("All done"));
        let events = host.events.lock().unwrap();
        // The tool ran even though it was requested via text, not the native slot.
        assert!(events.iter().any(|e| matches!(e, EngineEvent::ToolCall { name, .. } if name == "list_directory")));
        assert!(events.iter().any(|e| matches!(e, EngineEvent::ToolResult { ok: true, name, .. } if name == "list_directory")));
    }

    #[tokio::test]
    async fn orchestrator_escalates_then_runs_developer() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());

        // Chat tier escalates with a preview; developer answers in one turn.
        let chat = mock_with(vec![vec![tool_call_chunk(
            0,
            "c1",
            "escalate",
            r#"{"understanding":"add a greeting","user_visible_preview":"Voy a agregarlo."}"#,
        )]]);
        let dev = mock_with(vec![vec![content_chunk("Added the greeting.")]]);
        let engine = Engine::with_client(
            Box::new(dev),
            default_tools(),
            ctx.clone(),
            EngineConfig::ollama("mock"),
        );
        let orch = Orchestrator::with_parts(
            Box::new(chat),
            infinidev_tools::tools_for(true),
            engine,
            ctx,
            EngineConfig::ollama("mock"),
        );
        let host = CollectingHost::default();

        let result = orch.run_turn("add a greeting", &[], &host).await.unwrap();
        assert_eq!(result, "Added the greeting.");

        let events = host.events.lock().unwrap();
        assert!(events.iter().any(|e| matches!(e, EngineEvent::Decision { kind, detail } if kind == "escalate" && detail == "add a greeting")));
        assert!(events.iter().any(|e| matches!(e, EngineEvent::Notify { text, .. } if text == "Voy a agregarlo.")));
        // Developer actually ran (a StepStart fired) and the turn ended once.
        assert!(events.iter().any(|e| matches!(e, EngineEvent::StepStart { .. })));
        let turn_ends = events.iter().filter(|e| matches!(e, EngineEvent::TurnEnd { .. })).count();
        assert_eq!(turn_ends, 1, "exactly one terminal TurnEnd (developer's was swallowed)");
    }
}
