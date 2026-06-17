//! Live smoke test against any OpenAI-compatible provider (default: Ollama).
//!
//! Usage:
//!   cargo run -p infinidev-llm --example chat -- [model] [prompt]
//!   cargo run -p infinidev-llm --example chat -- gemma4:e4b "Say hi in one line."
//!
//! Env overrides: INFINIDEV_PROVIDER, INFINIDEV_BASE_URL, INFINIDEV_API_KEY.

use std::io::Write;

use futures::StreamExt;
use infinidev_llm::{build_client, cost, ChatRequest, Message};

#[tokio::main]
async fn main() -> infinidev_llm::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "gemma4:e4b".to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| "In one short sentence, what are you?".to_string());

    let provider = std::env::var("INFINIDEV_PROVIDER").unwrap_or_else(|_| "ollama".to_string());
    let base = std::env::var("INFINIDEV_BASE_URL").unwrap_or_default();
    let api_key = std::env::var("INFINIDEV_API_KEY").ok();

    let client = build_client(&provider, api_key, &base)?;
    let req = ChatRequest::new(&model, vec![Message::user(&prompt)]).with_temperature(0.2);

    eprintln!("→ {provider} / {model} (wire: {})\n", client.wire_name());

    // Streaming
    print!("stream: ");
    std::io::stdout().flush().ok();
    let mut stream = client.chat_stream(&req).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(text) = chunk.content {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
    }
    println!("\n");

    // Non-streaming + cost
    let resp = client.chat(&req).await?;
    println!("full:   {}", resp.content.unwrap_or_default());
    println!(
        "usage:  prompt={} completion={} total={}",
        resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens
    );
    match cost(&model, &resp.usage) {
        Some(c) => println!("cost:   ${:.6}", c.total_usd),
        None => println!("cost:   (local / unpriced — free)"),
    }
    Ok(())
}
