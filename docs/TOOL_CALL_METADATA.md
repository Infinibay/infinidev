# Explicit per-call public declarations (SDK opt-in)

`run_task(..., tool_call_metadata=True)` or `engine.run_turn(..., tool_call_metadata=True)` enables a principal-only transport for that invocation and its host hook continuation. DefaultFalse keeps existing schemas and protocol unchanged. Classic CLI opt-in is `infinidev --no-tui --resume --tool-call-metadata` (or `--prompt ... --tool-call-metadata`). It applies to this launch only, never settings/session persistence. TUI/phase/autonomous and flow shortcuts reject the flag rather than ignore it. No TUI control or global setting is added. Unsupported gather/phase/autonomous SDK routes reject the opt-in before work.

Every offered tool, including finish_turn/update_plan, receives the same envelope:

```json
{"metadata":{"reason":"Public operational purpose","expected_result":"Observable evidence expected"},"input":{"original_parameter":"literal value"}}
```

Both metadata fields are required strict nonblank strings, at most500 Unicode characters. Extra envelope/metadata fields are rejected. Values are not trimmed or fabricated. This requests concise public declarations, never private reasoning. Schema descriptions do not certify truth or understanding. Native and manual catalogs derive from the same original tool schema; manual examples reflect the active envelope.

The entire batch validates before any call executes. Only input enters original tool validation, permissions, hooks and executor; original rationale fields remain untouched and may retain legacy defaults. No metadata becomes an executable parameter, alias or permission. Three invalid batches still stop the turn. Default mode does not unwrap a model-invented envelope.

Original envelope stays in action history. Valid declarations are published as model-declared metadata, explicitly not host evidence, and retained in the same assistant action, never as synthetic user instructions. Native tool-call IDs, arguments and result pairing stay unchanged. When pressure compacts a manual action, only its validated public declarations are restored to that same assistant message after all compactors; large executable inputs remain compacted. The private retention marker is excluded from provider messages and payload measurements. No extra assistant round is added, and existing thinking ages and tool-result retention policies are preserved. Deferred finish_turn does not publish its accompanying declaration. This opt-in adds public/context content and tokens; it is a treatment, unlike passive recording. No semantic intent classifier, additional reviewer model, retry budget or result-provenance extension is added.

No live rollout or autonomy improvement is implied by offline transport tests. Root owns reviewed deployment, fresh-code confirmation, caller forwarding and isolated new-task comparison against baseline, with functional gates, corrections/rescues, repetitions, tokens and latency.

Stored public declarations remain assistant session messages. Historical replay labels them as AGENT within conversation context. This change does not rewrite stored history or filter actual human input by matching declaration text. Runtime retention markers are ephemeral and are not a database migration. Native-to-manual fallback preserves existing native actions. Default-off, cancellation, deferred closure and host-hook forwarding keep their previous contracts.
