//! The turn orchestrator — the three-tier pipeline that wraps the developer
//! [`Engine`], ported from the Python `engine/orchestration/`.
//!
//! Tier 1 — **chat agent** (read-only): every turn enters here. It either
//! answers conversationally (`respond`, the turn ends) or decides real work is
//! needed (`escalate`, hand off to the developer). This keeps "what does this
//! function do?" from spinning up the full plan-execute loop, and stops the
//! model from refusing write tasks it *thinks* it lacks tools for.
//!
//! Tier 2 — **developer loop** ([`Engine::run_turn`]): runs only on escalation.
//!
//! Tier 3 — **critic review**: runs only when the developer changed files.
//! A reviewer judges the work; if it flags concrete problems, the developer
//! gets exactly one fix iteration.
//!
//! The whole pipeline talks to the UI through the same [`EngineHost`] /
//! [`EngineEvent`] channel as the bare engine, so callers can swap
//! `Engine::run_turn` for `Orchestrator::run_turn` transparently.

use std::sync::Mutex;

use async_trait::async_trait;
use infinidev_llm::{
    apply_quirks, build_client, cost, ChatRequest, LlmClient, Message, Role, Tool as LlmTool,
    ToolCall,
};
use infinidev_tools::{tools_for, Tool, ToolContext, ToolError};

use crate::config::EngineConfig;
use crate::engine::{stream_completion, truncate, Engine};
use crate::error::Result;
use crate::event::{EngineEvent, EngineHost};
use crate::prompt::{CHAT_AGENT_PROMPT, REVIEW_PROMPT};

/// Chat-agent investigation budget. Read-only grounding is usually 0–3 calls;
/// the extra headroom lets a deeper look finish before auto-escalating.
const CHAT_MAX_ITERATIONS: u32 = 5;
const TOOL_OUTPUT_LIMIT: usize = 8000;

/// The top-level turn orchestrator. Holds its own read-only client + tools for
/// the chat-agent tier and an embedded developer [`Engine`].
pub struct Orchestrator {
    chat_client: Box<dyn LlmClient>,
    chat_tools: Vec<Box<dyn Tool>>,
    engine: Engine,
    ctx: ToolContext,
    cfg: EngineConfig,
}

/// The chat-agent tier's verdict for a turn.
enum Decision {
    Respond { reply: String },
    Escalate { understanding: String, preview: String },
}

impl Orchestrator {
    /// Build an orchestrator for the configured provider.
    pub fn new(cfg: EngineConfig, ctx: ToolContext) -> Result<Self> {
        let chat_client = build_client(&cfg.provider, cfg.api_key.clone(), &cfg.base_url)?;
        let chat_tools = tools_for(true); // read-only boundary for the chat tier
        let engine = Engine::new(cfg.clone(), ctx.clone())?;
        Ok(Self { chat_client, chat_tools, engine, ctx, cfg })
    }

    /// Build an orchestrator from injected parts (tests, custom hosts). The
    /// chat tier uses `chat_client` + `chat_tools`; escalations run `engine`.
    pub fn with_parts(
        chat_client: Box<dyn LlmClient>,
        chat_tools: Vec<Box<dyn Tool>>,
        engine: Engine,
        ctx: ToolContext,
        cfg: EngineConfig,
    ) -> Self {
        Self { chat_client, chat_tools, engine, ctx, cfg }
    }

    /// Run one user turn through the full pipeline.
    pub async fn run_turn(
        &self,
        user_input: &str,
        history: &[Message],
        host: &dyn EngineHost,
    ) -> Result<String> {
        self.run_turn_images(user_input, &[], history, host).await
    }

    /// Like [`run_turn`](Self::run_turn) but the user message carries attached
    /// images (`data:` URLs), threaded to both the chat tier and the developer.
    pub async fn run_turn_images(
        &self,
        user_input: &str,
        images: &[String],
        history: &[Message],
        host: &dyn EngineHost,
    ) -> Result<String> {
        // Escape hatch: skip triage entirely and run the developer directly.
        if !self.cfg.orchestrate {
            return self.engine.run_turn_images(user_input, images, history, host).await;
        }

        host.emit(EngineEvent::Phase { phase: "working".into() });
        match self.chat_agent(user_input, images, history, host).await? {
            Decision::Respond { reply } => {
                host.emit(EngineEvent::Decision { kind: "respond".into(), detail: String::new() });
                host.emit(EngineEvent::Phase { phase: "idle".into() });
                host.emit(EngineEvent::TurnEnd { result: reply.clone() });
                Ok(reply)
            }
            Decision::Escalate { understanding, preview } => {
                host.emit(EngineEvent::Decision {
                    kind: "escalate".into(),
                    detail: understanding.clone(),
                });
                if !preview.is_empty() {
                    host.emit(EngineEvent::Notify {
                        speaker: "infinidev".into(),
                        text: preview,
                        kind: "preview".into(),
                    });
                }
                self.develop_and_review(user_input, images, history, &understanding, host).await
            }
        }
    }

    fn find_chat_tool(&self, name: &str) -> Option<&dyn Tool> {
        self.chat_tools.iter().find(|t| t.name() == name).map(|b| b.as_ref())
    }

    /// Tier 1: the read-only chat-agent loop. Streams completions with the
    /// read toolbox + the two terminators (`respond` / `escalate`); executes
    /// read tools and re-prompts until it terminates or exhausts its budget.
    async fn chat_agent(
        &self,
        user_input: &str,
        images: &[String],
        history: &[Message],
        host: &dyn EngineHost,
    ) -> Result<Decision> {
        let mut messages: Vec<Message> = Vec::with_capacity(history.len() + 2);
        messages.push(Message::system(CHAT_AGENT_PROMPT));
        messages.extend_from_slice(history);
        messages.push(if images.is_empty() {
            Message::user(user_input)
        } else {
            Message::user_with_images(user_input, images.to_vec())
        });

        let mut schemas: Vec<LlmTool> = self.chat_tools.iter().map(|t| t.schema()).collect();
        schemas.push(respond_schema());
        schemas.push(escalate_schema());

        for _ in 0..CHAT_MAX_ITERATIONS {
            let mut req = ChatRequest::new(&self.cfg.model, messages.clone())
                .with_tools(schemas.clone())
                .with_temperature(0.1)
                .with_max_tokens(2000);
            apply_quirks(&self.cfg.provider, &self.cfg.model, &mut req);

            let (content, tool_calls, usage) =
                stream_completion(self.chat_client.as_ref(), &req, host).await?;
            emit_usage(&self.cfg.model, usage.as_ref(), host);

            if tool_calls.is_empty() {
                // Plain text instead of a terminator — treat as a respond; the
                // text was already streamed to the UI.
                let reply = if content.trim().is_empty() { "(no reply)".into() } else { content };
                return Ok(Decision::Respond { reply });
            }

            messages.push(Message {
                role: Role::Assistant,
                content: if content.is_empty() { None } else { Some(content) },
                name: None,
                tool_calls: Some(tool_calls.clone()),
                tool_call_id: None,
                reasoning_content: None,
                images: Vec::new(),
            });

            // A terminator ends the turn immediately, even if it shares the
            // batch with read calls.
            for tc in &tool_calls {
                match tc.function.name.as_str() {
                    "respond" => {
                        let args = parse_args(tc);
                        let message =
                            args.get("message").and_then(|v| v.as_str()).unwrap_or("").trim();
                        if message.is_empty() {
                            continue; // useless terminator; keep looking / re-prompt
                        }
                        // The args weren't streamed as visible content, so emit
                        // the reply now unless the model also chatted in plain text.
                        if !messages
                            .last()
                            .and_then(|m| m.content.as_deref())
                            .map(|c| !c.trim().is_empty())
                            .unwrap_or(false)
                        {
                            host.emit(EngineEvent::StreamChunk { chunk: message.to_string() });
                        }
                        return Ok(Decision::Respond { reply: message.to_string() });
                    }
                    "escalate" => {
                        let args = parse_args(tc);
                        let understanding =
                            args.get("understanding").and_then(|v| v.as_str()).unwrap_or("").trim();
                        if understanding.is_empty() {
                            continue;
                        }
                        let preview = args
                            .get("user_visible_preview")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        return Ok(Decision::Escalate {
                            understanding: understanding.to_string(),
                            preview,
                        });
                    }
                    _ => {}
                }
            }

            // No terminator — run the read-only tools and feed results back.
            for tc in &tool_calls {
                let name = tc.function.name.clone();
                let id = tc.id.clone().unwrap_or_default();
                let args = parse_args(tc);
                host.emit(EngineEvent::ToolCall { id: id.clone(), name: name.clone(), args: args.clone() });
                let result = match self.find_chat_tool(&name) {
                    Some(tool) => tool.execute(args, &self.ctx).await,
                    None => Err(ToolError::NotFound(format!("unknown tool: {name}"))),
                };
                let (ok, output) = match result {
                    Ok(o) => (true, o),
                    Err(e) => (false, format!("ERROR: {e}")),
                };
                let shown = truncate(&output, TOOL_OUTPUT_LIMIT);
                host.emit(EngineEvent::ToolResult { id: id.clone(), name: name.clone(), ok, output: shown.clone() });
                messages.push(Message::tool_named(id, name, shown));
            }
        }

        // Budget exhausted without a terminator. The chat agent burning its
        // whole budget is itself a signal the request is non-trivial, so we
        // escalate rather than strand the user (mirrors the Python auto-escalate).
        Ok(Decision::Escalate {
            understanding: format!(
                "[Auto-escalation: the chat agent used its full {CHAT_MAX_ITERATIONS}-step \
                 read budget without concluding.] Original request: {user_input}"
            ),
            preview: String::new(),
        })
    }

    /// Tiers 2 + 3: run the developer loop on the escalated request, then (if
    /// it changed files) a critic review with up to one fix iteration.
    async fn develop_and_review(
        &self,
        user_input: &str,
        images: &[String],
        history: &[Message],
        understanding: &str,
        host: &dyn EngineHost,
    ) -> Result<String> {
        // Forward the chat agent's framing so the developer doesn't re-derive it.
        let dev_input = if understanding.is_empty() {
            user_input.to_string()
        } else {
            format!("[Triage context: {understanding}]\n\n{user_input}")
        };

        let dev_host = CaptureHost::new(host);
        let result = self.engine.run_turn_images(&dev_input, images, history, &dev_host).await?;
        let changed = dev_host.changed_files();

        if self.cfg.review && !changed.is_empty() {
            if let Some((status, notes)) = self.review(user_input, &result, &changed, host).await {
                host.emit(EngineEvent::Review { status: status.clone(), notes: notes.clone() });
                if status == "changes" && !notes.is_empty() {
                    let bullets = notes
                        .iter()
                        .map(|n| format!("- {n}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    let fix_input = format!(
                        "A reviewer flagged these problems with your changes. Fix them, then give \
                         a concise final summary:\n{bullets}"
                    );
                    let mut fix_history = history.to_vec();
                    fix_history.push(Message::user(&dev_input));
                    fix_history.push(Message::assistant(&result));
                    let fix_host = CaptureHost::new(host);
                    let fixed = self.engine.run_turn(&fix_input, &fix_history, &fix_host).await?;
                    host.emit(EngineEvent::Phase { phase: "idle".into() });
                    host.emit(EngineEvent::TurnEnd { result: fixed.clone() });
                    return Ok(fixed);
                }
            }
        }

        host.emit(EngineEvent::Phase { phase: "idle".into() });
        host.emit(EngineEvent::TurnEnd { result: result.clone() });
        Ok(result)
    }

    /// Tier 3: a single non-streaming critic call. Returns (status, notes) with
    /// status "approve" | "changes", or `None` if the review call failed (we
    /// then silently skip review rather than block the turn).
    async fn review(
        &self,
        user_input: &str,
        result: &str,
        changed: &[(String, String)],
        host: &dyn EngineHost,
    ) -> Option<(String, Vec<String>)> {
        let files = changed
            .iter()
            .map(|(p, a)| format!("- {a}: {p}"))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!(
            "User request:\n{user_input}\n\nThe agent's final summary:\n{result}\n\n\
             Files the agent changed:\n{files}\n\nReview the work."
        );
        let messages = vec![Message::system(REVIEW_PROMPT), Message::user(user)];
        let mut req = ChatRequest::new(&self.cfg.model, messages)
            .with_temperature(0.0)
            .with_max_tokens(600);
        apply_quirks(&self.cfg.provider, &self.cfg.model, &mut req);

        let resp = self.chat_client.chat(&req).await.ok()?;
        emit_usage(&self.cfg.model, Some(&resp.usage), host);
        Some(parse_review(resp.content.as_deref().unwrap_or("")))
    }
}

/// Parse a tool call's JSON arguments, defaulting to `{}` on malformed input.
fn parse_args(tc: &ToolCall) -> serde_json::Value {
    serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| serde_json::json!({}))
}

/// Emit a `Usage` event with computed cost, if usage is present.
fn emit_usage(model: &str, usage: Option<&infinidev_llm::Usage>, host: &dyn EngineHost) {
    if let Some(u) = usage {
        host.emit(EngineEvent::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cost_usd: cost(model, u).map(|c| c.total_usd),
        });
    }
}

/// Parse the critic's verdict. First line `APPROVE`/`CHANGES`; on changes, the
/// `- ` / `* ` bullets that follow become the notes (capped at 5).
fn parse_review(text: &str) -> (String, Vec<String>) {
    let first = text.lines().next().unwrap_or("").trim().to_uppercase();
    if first.contains("CHANGES") && !first.contains("APPROVE") {
        let notes: Vec<String> = text
            .lines()
            .filter_map(|l| {
                let t = l.trim();
                t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")).map(|s| s.trim().to_string())
            })
            .filter(|s| !s.is_empty())
            .take(5)
            .collect();
        ("changes".to_string(), notes)
    } else {
        ("approve".to_string(), Vec::new())
    }
}

fn respond_schema() -> LlmTool {
    LlmTool::function(
        "respond",
        "End the turn with a conversational reply to the user. Use for greetings, \
         thanks, opinions, or any question you can answer from reading the code.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The reply, in the user's language. 1–3 sentences."
                }
            },
            "required": ["message"]
        }),
    )
}

fn escalate_schema() -> LlmTool {
    LlmTool::function(
        "escalate",
        "Hand the turn to the developer for real work (write files, run commands, \
         install, git, edit symbols). Use when the user clearly asked for execution \
         or approved a proposal you made.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "understanding": {
                    "type": "string",
                    "description": "1–2 sentences, in your own words, describing what the user wants."
                },
                "user_visible_preview": {
                    "type": "string",
                    "description": "Optional 1-sentence message shown to the user before the \
                                    developer runs, in the user's language."
                }
            },
            "required": ["understanding"]
        }),
    )
}

/// Wraps the real host while the developer loop runs: records `FileChange`
/// events (so the orchestrator knows whether to review) and swallows the
/// developer's terminal `TurnEnd` + `Phase{idle}` so the orchestrator can emit
/// the real terminal events after the review tier.
struct CaptureHost<'a> {
    inner: &'a dyn EngineHost,
    changed: Mutex<Vec<(String, String)>>,
}

impl<'a> CaptureHost<'a> {
    fn new(inner: &'a dyn EngineHost) -> Self {
        Self { inner, changed: Mutex::new(Vec::new()) }
    }
    fn changed_files(&self) -> Vec<(String, String)> {
        self.changed.lock().unwrap().clone()
    }
}

#[async_trait]
impl EngineHost for CaptureHost<'_> {
    fn emit(&self, event: EngineEvent) {
        match &event {
            EngineEvent::FileChange { path, action } => {
                self.changed.lock().unwrap().push((path.clone(), action.clone()));
                self.inner.emit(event);
            }
            // Swallow the developer's terminal events; the orchestrator owns
            // the real ones (emitted after review).
            EngineEvent::TurnEnd { .. } => {}
            EngineEvent::Phase { phase } if phase == "idle" => {}
            _ => self.inner.emit(event),
        }
    }

    async fn ask_user(&self, prompt: String, kind: String) -> Option<String> {
        self.inner.ask_user(prompt, kind).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_parses_approve() {
        let (status, notes) = parse_review("APPROVE\nlooks good");
        assert_eq!(status, "approve");
        assert!(notes.is_empty());
    }

    #[test]
    fn review_parses_changes_with_notes() {
        let (status, notes) = parse_review(
            "CHANGES\n- missing null check in foo()\n- test not updated\nsome trailing prose",
        );
        assert_eq!(status, "changes");
        assert_eq!(notes, vec!["missing null check in foo()", "test not updated"]);
    }

    #[test]
    fn review_defaults_to_approve_on_garbage() {
        assert_eq!(parse_review("").0, "approve");
        assert_eq!(parse_review("the work seems fine to me").0, "approve");
    }

    #[test]
    fn terminator_schemas_are_well_formed() {
        let r = respond_schema();
        assert_eq!(r.function.name, "respond");
        assert_eq!(r.function.parameters["required"][0], "message");
        let e = escalate_schema();
        assert_eq!(e.function.name, "escalate");
        assert_eq!(e.function.parameters["required"][0], "understanding");
    }
}
