# Passive context recording (SDK, opt-in per invocation)

`run_task(..., record_context_dir=Path("/absolute/private/capture-new"))` records
metadata for that principal invocation and its hook continuation. Default None
keeps recording disabled; no session/global setting or model tool is added.
The directory must not exist, its parent must exist, and it must be outside the
workspace. No-follow descriptor support is required for this opt-in feature.
The current interface is SDK-only; typing a task does not silently enable it.

`metadata.jsonl` contains source nodes, producer-declared logical bundles,
canonical request snapshots and allowlisted projections of existing host events.
`manifest.json` identifies the recording invocation and observed child run IDs.
This is not session liveness or a task-completion certificate. Parent host IDs
and source checkpoints are not reconstructed when unavailable.

History source nodes identify actual selected database records and roles, loader
and renderer clipping, sizes and redacted selected-view hashes. The digest is
NOT the original content hash. Loader character limits and renderer byte ranges
name their stages explicitly. No history is rewritten or classified as authority.
Bundles describe logical provenance, not byte-for-byte span coverage. Requests
validate whether their message still matches that declared assembly source.
Changed/unknown mappings are marked; they do not inherit a false original mapping.

Request scope is principal canonical input BEFORE the LLM adapter. Provider
events contain adapted metadata from existing litellm_input traces, not HTTP
wire or rendered Strata chat-template content. A later hook/SDK transform can
change the actual provider input; this recording does not claim otherwise.

Records omit bodies, tool arguments, reasoning fields, assets, environment and
headers. Supplied private/unknown categories omit their digest. Public-text hashes
use existing configured-secret and credential-shape redaction first; redaction
is best effort, not proof that arbitrary unknown sensitive text was recognized.
No private/unknown-content fingerprint is persisted. Directory mode0700 and
files0600 restrict local access. Treat the resulting metadata as private.

Only already emitted public-message events are observed. Asking the model for
new decision notes would change prompts/tokens/behavior and is NOT implemented
or part of the baseline. Existing prose is not proof of model understanding.
A tool_result means a dispatcher attempt; its opaque body is not parsed as proof
of success. Typed pre-dispatch rejections retain executed=false.

Limits:64KiB per record and4MiB total. Records are whole or omitted; omitted/write
failures make the manifest incomplete. Preflight errors fail before model/tool
work. Mid-run recording failures log static diagnostics without retrying model
calls or changing their result. A missing/failed manifest means recording status
is unknown. No upload, DB migration, persistent recording, inference, installation
or live-process control is performed by this feature.
