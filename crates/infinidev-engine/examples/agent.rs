//! Live end-to-end smoke of the embedded engine (the same code path the Tauri
//! desktop runs). Runs one agentic turn — with the real tool set — against a
//! local model, printing the event stream.
//!
//!   cargo run -p infinidev-engine --example agent -- [model] [prompt]
//!
//! Operates on the current working directory as the project.

use async_trait::async_trait;
use infinidev_engine::{Engine, EngineConfig, EngineEvent, EngineHost, ToolContext};

struct PrintHost;

#[async_trait]
impl EngineHost for PrintHost {
    fn emit(&self, ev: EngineEvent) {
        match ev {
            EngineEvent::StreamChunk { chunk } => print!("{chunk}"),
            EngineEvent::StepStart { step, max } => eprintln!("\n── step {step}/{max}"),
            EngineEvent::ToolCall { name, args, .. } => eprintln!("[tool] {name} {args}"),
            EngineEvent::ToolResult { name, ok, output, .. } => {
                let preview: String = output.chars().take(160).collect();
                eprintln!("[tool done] {name} ok={ok} :: {preview}");
            }
            EngineEvent::FileChange { path, action } => eprintln!("[file {action}] {path}"),
            EngineEvent::Usage { total_tokens, cost_usd, .. } => {
                eprintln!("[usage] {total_tokens} tokens cost={cost_usd:?}");
            }
            EngineEvent::TurnEnd { .. } => eprintln!("\n[turn end]"),
            EngineEvent::Error { message } => eprintln!("[error] {message}"),
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "gemma4:e4b".to_string());
    let prompt = args.next().unwrap_or_else(|| {
        "Use the list_directory tool to look at the project root, then tell me in one sentence what kind of project this is.".to_string()
    });

    let cfg = EngineConfig::ollama(model);
    let ctx = ToolContext::new(std::env::current_dir().unwrap());
    let engine = Engine::new(cfg, ctx).expect("build engine");

    eprintln!("→ running a turn (tools: file/search/shell/git/knowledge/symbols)\n");
    let result = engine.run_turn(&prompt, &[], &PrintHost).await.expect("run turn");
    eprintln!("\n=== final answer ===\n{result}");
}
