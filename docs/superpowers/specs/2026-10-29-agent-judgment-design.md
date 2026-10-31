# Trustworthy execution and engineering judgment

The user has delegated implementation choices and requested working changes to
Infinidev, with real programming tasks and sustained work as the product target.
This iteration strengthens one concrete execution boundary and the instructions
that guide engineering decisions. Existing Python, LiteLLM, permission checks,
prompt profiles, and local-model compatibility remain the implementation base.

## Approaches considered

1. Rewrite orchestration: potentially broad gains, but changes too many contracts
   before a demonstrated failure establishes what needs replacing.
2. Add instructions alone: inexpensive, but cannot prevent a deterministic failed
   check from being accepted by the engine.
3. Fix the completion boundary, then align guidance and documentation: selected
   by the implementer under the user's delegated authority. This produces a
   checkable improvement without another workflow engine or configuration flag.

## Runtime contract

A gate must distinguish a correction turn, an accepted completion, and a bounded
failure. After the configured objective correction budget, a failing check closes
the Step as blocked, with the actual failure in its summary and evidence. It must
not reuse the model's success answer. Unexpected verifier failures count as failed
verification, never a pass. Passing evidence from an earlier invocation of the
same Step is invalidated before a new check and restored only by a new pass.

Keep the legacy boolean gate entry points for callers, but give the engine one
typed evaluation entry point returning `StepResult | None`. `None` requests a
correction turn; a result owns the accepted outcome. No persistence-format change,
new dependency, extra model call, or additional runtime setting is needed.

## Guidance and product direction

Default and compact prompts should describe recoverable memory honestly. Replace
imaginary scoring pressure with actionable recovery instructions. Encourage
ordinary reversible decisions, a conventional baseline for brainstorming, clear
module ownership, visual verification for UI changes, and concise explanations in
the user's language. Preserve authority and permission boundaries. Keep durable
notes about decisions, constraints, unresolved work, and next checks.

The README should lead with real tasks and hosted reasoning/coding models, explain
subscription configuration already supported by the project, and describe local
models as a supported choice. It must distinguish mechanical checks from judgment
and avoid promising that prompts guarantee correct decisions.

## Validation and limits

Use failing regression tests before runtime changes. Cover correction exhaustion,
verifier exceptions, recovery to a pass, a second Step, gate ordering, native and
manual tool calls, and the actual inner-loop consumer. Run focused suites and the
full repository suite, reporting pre-existing failures separately. Prompt contract
tests establish composition and consistency; they do not establish model quality.
Record further architectural ideas separately from implemented behavior.
