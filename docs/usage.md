# Provider usage

`/usage` adapts to the selected provider in both interfaces. In the TUI, provider queries
run in the background so chat remains responsive. The command does not generate a model
response to measure usage.

| Provider | Available information |
| --- | --- |
| ChatGPT Subscription (Codex) | Subscription quota windows, remaining percentages, reset times, and credits when reported by Codex |
| OpenAI API | Observed token totals and rate-limit headers; organization usage and costs with an optional admin key |
| Anthropic API | Observed token totals and rate-limit headers; organization usage and costs with an optional admin key |
| Ollama | Observed token totals; no hosted billing quota |
| Other providers, including GLM and Qwen | Observed token totals and recognized rate-limit headers; account quota lookup is not integrated |

## Codex subscription

Install the Codex CLI and sign in with `codex login`. Infinidev starts a short-lived
`codex app-server` client and calls the documented
[`account/rateLimits/read`](https://learn.chatgpt.com/docs/app-server) method. It shows
the window durations returned by the server instead of assuming all plans use the
same periods. The CLI inherits `CODEX_HOME` when configured. Missing credentials,
unavailable quotas, and timeouts produce an explicit unavailable message.

## API organization reports

An inference API key does not expose Codex or Claude subscription quota. Organizational
API reporting is a separate capability with separate credentials:

```bash
export INFINIDEV_USAGE_OPENAI_ADMIN_KEY='your-openai-admin-key'
export INFINIDEV_USAGE_ANTHROPIC_ADMIN_KEY='your-anthropic-admin-key'
```

These settings are optional and masked in the settings UI. Reporting credentials go
only to the provider's official reporting endpoints, never to model inference calls.
Without a reporting key, `/usage` still shows the calls observed by Infinidev.

Reports cover the current UTC day across the organization, including other models and
projects, and may lag. OpenAI provides separate
[usage and cost endpoints](https://platform.openai.com/docs/api-reference/usage).
Anthropic's [Usage and Cost API](https://platform.claude.com/docs/en/manage-claude/usage-cost-api)
requires organization reporting access; its cost amounts are converted from USD cents
and exclude Priority Tier costs. Neither report is a remaining account balance.

## Observed usage

Local totals cover completed calls in this Infinidev process, including agent work,
scoped to the selected endpoint, model, and credential or Codex account. They reset on
restart and do not reconstruct earlier sessions. Streamed usage is counted once when
the stream finishes or closes. If a call omits usage, the report marks the token totals
as incomplete. Costs appear only when supplied by the SDK and identify how many calls
were priced.

Rate-limit values are snapshots from the most recently observed response headers, with
their observation time. They are not a fresh quota query or a billing balance. See the
[OpenAI](https://developers.openai.com/api/docs/guides/rate-limits) and
[Anthropic](https://platform.claude.com/docs/en/api/rate-limits) header definitions.
