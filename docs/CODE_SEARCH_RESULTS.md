# Code search result observations (I1a)

`code_search` adds `search_metadata` to its existing JSON response. The input
schema, arguments, default regex behavior, backend selection and legacy values
remain the same. Successful responses retain `pattern`, `file_path`, `match_count`,
`truncated` and `matches`. Error responses retain their original `error` string.
There is no literal-mode argument in I1a, additional tool, prompt layer, retry,
indexing, model feedback, cancellation or execution limit.

| Field | Meaning |
| --- | --- |
| `schema_version` | Version of this observation format, currently 1. |
| `outcome` | `matches`, `empty`, `partial`, `error` or `unavailable`. |
| `mode` | Existing local `regex` mode, or `unknown` for the remote helper. |
| `backend` | `python_re`, `git_grep`, `grep`, `pod_helper` or `unknown`. |
| `dialect` | `python`, extended regex `ere`, or `unknown`. These are not interchangeable syntaxes. |
| `scope` | `file`, `directory`, `remote_unknown` or `unknown`. |
| `root` | Resolved search path passed to the backend; a remote path can be relative. Not a repository identity. |
| `coverage` | `completed_scope`, `limited` or `unknown`. |
| `filters_applied` | Known local case/context/result limits and applicable directory filters; null for remote unknown. |
| `reasons` | Mechanical observations qualifying the result, without classifying diagnostic prose. |
| `backend_exit_code` | Search subprocess exit code when observed; excludes Git-discovery probes. |
| `backend_stderr` | Optional first 2000 characters of backend stderr, including whitespace. |
| `backend_stderr_truncated` | Whether this new diagnostic preview was shortened. Legacy error bodies are not shortened. |
| `unparsed_output_lines` | Optional count of lines the existing directory parser could not interpret. Not a count of missing matches. |

`completed_scope` means normal completion within the actual backend's observed
scope and filters, with no detected uncertainty. It is not an atomic filesystem
snapshot, source fingerprint, universal absence proof or certification that a
programming question is answered. `empty` therefore means no returned matches
under those specific conditions, not that a symbol cannot exist elsewhere.

A cap, nonempty stderr or unparsed output produces a `partial` observation, even
when `match_count` is zero. Actual hits remain in `matches`. `limited` means an
observed restriction; `unknown` means completion cannot be determined from this
interface. Errors remain errors, with no new zero count. A backend invocation
that raises OSError is `unavailable`; this does not assert permanent lack of support.
Validation/permission failures outside this tool's execution need not have this
block; missing metadata must not be interpreted as empty.

`backend_stderr` is a reason code whenever stderr is nonempty, not a judgment that
its text describes a particular error. Other reasons include `result_limit`,
`unparsed_output`, `omitted_output`, `backend_timeout` and `remote_metadata_unknown`.
`aggregate_exit` applies only to exit 123 from the existing find/xargs path;
other backends returning 123 are qualified as `backend_exit_uncertain`.
The legacy aggregate path may discard stdout; I1a exposes this uncertainty without
restoring or reinterpreting the old payload. Restoring matches is a separate change.

Files use Python re; directories use git grep -E or grep -E. The existing parser,
case/locale behavior, exclusions and filters remain unchanged. Directory-only
filters are not reported as applied to direct files. `git_defaults` identifies the
existing tracked/nonignored-untracked Git search; `grep_defaults` identifies the
existing fallback exclusion list in CodeSearchTool. Remote helper metadata is
unknown rather than inferred from local filesystem checks or legacy truncated=false.

For Git with extension wildcard pathspecs and a depth bound, the effective depth
is null and coverage unknown (`git_depth_with_pathspec_uncertain`).
[Git's manual](https://git-scm.com/docs/git-grep#Documentation/git-grep.txt---max-depthltdepthgt)
documents a wildcard exception, although local Git 2.56.1 applied the depth in
tracked and untracked probes. This metadata does not assert the bound was ignored.
For grep's find/xargs depth branch, the effective directory exclusions are null
(`directory_exclusions_not_applied`): explicit file operands can include files
inside excluded directories. These observations do not change existing commands
or matches. A null effective filter does not certify the requested restriction.

Metadata contains no timestamps, per-call nonces or changing counters. Identical
searches/results can keep identical output hashes within a variant. Adding this
block changes the complete returned JSON, so cross-variant hash changes do not
measure semantic novelty. No source hash or freshness is invented from a result hash.

Offline tests verify the additive contract and uncertainty cases. They do not
prove improved autonomous investigation. A live pilot requires Root's frozen task,
checkpoint, source-evidence rubric, model and common budget. Primary measurements
are supported answers, current citations and pertinent next questions/tests; token
cost, latency, calls and finish status are secondary. No live pilot is implied here.
