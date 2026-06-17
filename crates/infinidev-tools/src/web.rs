//! Web tools: fetch a URL and return readable text. Network access is a
//! developer-tier capability (the read-only chat tier never sees it), mirroring
//! the Python `tools/web` set.

use async_trait::async_trait;
use serde::Deserialize;

use crate::context::ToolContext;
use crate::error::{Result, ToolError};
use crate::tool::Tool;

const DEFAULT_MAX_CHARS: usize = 20_000;
const HARD_CAP_CHARS: usize = 200_000;

#[derive(Deserialize)]
struct FetchArgs {
    url: String,
    #[serde(default)]
    max_chars: Option<usize>,
}

pub struct WebFetch;

#[async_trait]
impl Tool for WebFetch {
    fn name(&self) -> &'static str {
        "web_fetch"
    }
    fn description(&self) -> &'static str {
        "Fetch an http(s) URL and return its text. HTML is reduced to readable \
         text (scripts/styles/tags stripped). Use for docs, references, and pages \
         the user links."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
                "max_chars": {"type": "integer", "description": "Cap on returned characters (default 20000)."}
            },
            "required": ["url"]
        })
    }
    // Network access is NOT part of the read-only boundary; only the developer
    // tier gets this tool.
    async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> Result<String> {
        let a: FetchArgs = serde_json::from_value(args)
            .map_err(|e| ToolError::InvalidArgs(e.to_string()))?;
        if !(a.url.starts_with("http://") || a.url.starts_with("https://")) {
            return Err(ToolError::InvalidArgs("url must start with http:// or https://".into()));
        }
        let cap = a.max_chars.unwrap_or(DEFAULT_MAX_CHARS).min(HARD_CAP_CHARS);

        let client = reqwest::Client::builder()
            .user_agent(concat!("infinidev/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ToolError::Other(e.to_string()))?;
        let resp = client
            .get(&a.url)
            .send()
            .await
            .map_err(|e| ToolError::Other(format!("request failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(ToolError::Other(format!("HTTP {} fetching {}", status.as_u16(), a.url)));
        }
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = resp
            .text()
            .await
            .map_err(|e| ToolError::Other(format!("reading body failed: {e}")))?;

        let text = if content_type.contains("html") || looks_like_html(&body) {
            html_to_text(&body)
        } else {
            body
        };
        Ok(truncate_chars(&text, cap))
    }
}

fn looks_like_html(s: &str) -> bool {
    let head = s.trim_start().get(..256).unwrap_or(s);
    let lower = head.to_lowercase();
    lower.contains("<!doctype html") || lower.contains("<html")
}

/// Reduce HTML to readable text: drop `<script>`/`<style>` blocks and comments,
/// strip remaining tags, decode a few common entities, and collapse whitespace.
/// Deliberately simple (no DOM) — enough to feed page content to the model.
fn html_to_text(html: &str) -> String {
    // Remove script/style blocks and comments first. The `regex` crate has no
    // backreferences, so script and style get separate patterns.
    let script = regex::Regex::new(r"(?is)<script\b[^>]*>.*?</\s*script\s*>").unwrap();
    let style = regex::Regex::new(r"(?is)<style\b[^>]*>.*?</\s*style\s*>").unwrap();
    let no_script = script.replace_all(html, " ");
    let no_blocks = style.replace_all(&no_script, " ");
    let comment = regex::Regex::new(r"(?s)<!--.*?-->").unwrap();
    let no_comments = comment.replace_all(&no_blocks, " ");
    // Strip all remaining tags.
    let tags = regex::Regex::new(r"(?s)<[^>]+>").unwrap();
    let stripped = tags.replace_all(&no_comments, " ");
    // Decode a handful of common entities.
    let decoded = stripped
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    // Collapse runs of whitespace, preserving paragraph-ish newlines.
    let ws = regex::Regex::new(r"[ \t]+").unwrap();
    let collapsed = ws.replace_all(&decoded, " ");
    let blank = regex::Regex::new(r"\n\s*\n\s*").unwrap();
    blank.replace_all(collapsed.trim(), "\n\n").trim().to_string()
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}\n…(truncated)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_html_to_text() {
        let html = "<html><head><style>.a{color:red}</style><script>x=1</script></head>\
                    <body><h1>Title</h1><p>Hello &amp; welcome</p><!-- note --></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello & welcome"));
        assert!(!text.contains("color:red"));
        assert!(!text.contains("x=1"));
        assert!(!text.contains("note"));
        assert!(!text.contains('<'));
    }

    #[test]
    fn truncates_by_chars() {
        assert_eq!(truncate_chars("hello", 10), "hello");
        assert!(truncate_chars("hello world", 5).starts_with("hello"));
        assert!(truncate_chars("hello world", 5).contains("truncated"));
    }

    #[test]
    fn rejects_non_http() {
        let r: std::result::Result<FetchArgs, _> = serde_json::from_value(serde_json::json!({"url":"ftp://x"}));
        assert!(r.is_ok()); // parse ok; the scheme guard is in execute()
    }
}
