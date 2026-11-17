# Code search result observations (I1a)

`code_search` adds `search_metadata` to its existing JSON response. The input
schema, arguments, default regex behavior and backend selection remain the same.
The directory-context correction below changes how backend records are decoded;
other existing response values retain their meaning. Successful responses retain `pattern`, `file_path`, `match_count`,
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

Files use Python re; directories use git grep -E or grep -E. Case/locale behavior,
exclusions and filters remain unchanged. The corrected directory parser is described below. Directory-only
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


## Directory context correction (2026-11-17)

Directory hits now retain requested nearby rows in each hit's `context`, using the
same line/content shape as direct-file search. Context rows are not hits and do
not consume the match cap. GNU grep uses filename-NUL output (`--null`); git grep
uses `-z --column` so the producer distinguishes hits from context. Paths with
hyphens/colons are not decoded by guessing path separators. Input schemas,
patterns, backend selection and search filters are unchanged; this framing change
is internal to the two directory subprocess backends. Single-file/pod paths are unchanged.

Recognized context is no longer counted as unparsed output. Unrecognized rows,
nonempty stderr, caps, depth/filter uncertainty and aggregate exits still qualify
coverage; the parser never clears those existing reasons. `completed_scope` is
limited to the observed search scope and is never a global absence proof.

Offline replay of C's disclosed effective query on its unchanged snapshot returns
the same pre-fix JSON digest as the historical tool result. The fixed replay keeps
three definition locators and five context lines per hit. Historical raw grep
stdout was not captured; this is a source/fixture replay, not reconstructed model
input. No inference, next-read recommendation, repeated-result gate or causal claim
about C's eight searches is part of this correction.

The long `--null` flag is intentional: Apple's grep uses `-Z` for decompression,
while `--null` selects filename framing ([Apple's option definitions](https://github.com/apple-oss-distributions/text_cmds/blob/main/grep/grep.c)).
Tests run real GNU grep/Git locally and assert the portable argv contract; they do
not claim a macOS runtime test. Output records split on LF only, preserving VT,
FF, NEL and Unicode line-separator characters inside source content. Existing
subprocess text-mode CR/LF normalization is outside this parser correction.
