# Research: high-value prompt capabilities

## Purpose and selection rules

This note maps recurring workflows from the skill collections requested by the user to
small, opt-in prompt capabilities that fit Infinidev's existing developer loop. It is a
curation document, not an instruction to install or execute third-party skills.

A candidate is included only when it:

1. applies across languages, frameworks, and model providers;
2. addresses a distinct engineering failure mode rather than restating an existing
   capability;
3. can improve agent behavior through instructions and existing tools;
4. preserves the user's authority and requires evidence before claiming success; and
5. is concise enough to activate conditionally without burdening unrelated tasks.

Vendor-specific setup, paid services, offensive-security playbooks, publishing or payment
operations, and domain-specific content generation are excluded. A prompt also cannot repair
model context limits, tool/runtime defects, or missing product capabilities; those require
code or configuration changes.

## Sources inspected

- [awesome-skills organization](https://github.com/awesome-skills): its public catalog
  prominently includes code review, first-principles analysis, 5 Whys root-cause analysis,
  mobile UI/accessibility, diagram validation, and session-pattern analysis.
- [Awesome GitHub Copilot skills](https://awesome-copilot.github.com/skills/): the published
  catalog exposes task-oriented skills, including a mentoring workflow with progressive
  hints rather than immediate answers.
- [Agentic Awesome Skills](https://github.com/sickn33/agentic-awesome-skills): its catalog
  reports more than 2,000 skills and groups recurring engineering work into focused surfaces
  for web apps, secure development, QA, DevOps, accessibility, APIs, data, observability,
  incident response, privacy, mobile, and AI evaluation. Its own safety guidance recommends
  exact reviewed subsets instead of loading the entire catalog.
- [AAS source ledger](https://github.com/sickn33/agentic-awesome-skills#official-sources):
  examples used for cross-checking include pre-release review, browser testing, SQL cost
  audits, cron diagnosis, migration conflicts, context pruning, session audits, production
  readiness, algorithm invariants, and evidence-backed research.

These catalogs overlap heavily and include mirrors. Frequency therefore indicates a recurring
workflow, not independent empirical validation or permission to copy a skill's prose.

## Existing coverage and implementation status

The catalog initially covered eight broad boundaries: `debugging`, `testing`, `review`,
`security`, `documentation`, `performance`, interactive `accessibility`, and persisted-data
`data_migration`. The 30 candidates below were subsequently implemented, along with four
complaint-driven safeguards: `authority_boundaries`, `verification_integrity`,
`change_preservation`, and `progress_discipline`.

The runtime catalog therefore contains 42 opt-in capabilities. They are seeded disabled by
default and are composed only when explicitly enabled by a shared or project prompt profile.
For the authoritative current IDs and activation examples, see `docs/prompt-profiles.md`; this
research note records the rationale and source mapping that led to that catalog.

The additions complement rather than replace the initial capabilities. For example, release
readiness does not duplicate testing: it checks operational artifacts and the deployed revision
after tests have passed. Root-cause analysis does not duplicate debugging: it governs repeated
or systemic failures where repairing the nearest symptom is insufficient.

## Implemented capability matrix

| Proposed ID | Trigger / recurring problem | Infinidev-compatible guidance | Catalog signals | Decision |
| --- | --- | --- | --- | --- |
| `capability.requirements_clarity` | The requested outcome is materially ambiguous or has conflicting interpretations | Separate facts, assumptions, and user-owned choices; inspect local evidence first; ask only for an unresolved consequential choice | rich elicitation, idea-to-PRD, first-principles workflows | Include; prevents confident work on the wrong target without turning every task into an interview |
| `capability.root_cause_analysis` | A failure repeats, spans components, or has only a symptom-level explanation | Build a causal chain from observations, test the nearest discriminating hypothesis, and repair the earliest demonstrated controllable cause | 5 Whys, systematic debugging, performance RCA | Include; narrower trigger than ordinary debugging |
| `capability.architecture_impact` | A change crosses module, service, package, or public-contract boundaries | Trace callers, data flow, ownership, and compatibility before editing; minimize moved boundaries and verify affected entry points | software graphs, architecture review, change-impact analysis | Include; addresses local fixes that break distant consumers |
| `capability.api_contracts` | An API, protocol, schema, CLI contract, or integration boundary changes | Identify producers and consumers, preserve compatibility unless explicitly changed, validate error semantics, and exercise representative round trips | API platform, OpenAPI, auth, integration-testing collections | Include; distinct from database migration and generic testing |
| `capability.concurrency` | Work touches threads, async tasks, queues, locks, retries, or lifecycle cancellation | State ownership and ordering invariants; cover cancellation, failure, restart, and balanced accounting; use deterministic synchronization in tests | async networking, workflow runners, queue/incident patterns | Include; directly relevant to Infinidev's watcher/indexing lifecycle |
| `capability.resource_lifecycle` | Code acquires processes, files, sockets, watchers, tasks, or temporary resources | Pair every acquisition with idempotent cleanup; check partial startup, repeated stop, and retry after failure | browser sessions, remote jobs, service operations, lifecycle tooling | Include; catches leaks and unrecoverable partial initialization |
| `capability.error_recovery` | A workflow can partially fail or retry external/tool operations | Classify failures, preserve the original cause, retry only transient cases with bounds, and leave state resumable or clearly rolled back | tool-use guardian, apply/recovery plans, workflow diagnosis | Include; broader than data migration but narrower than generic robustness prose |
| `capability.observability` | Behavior depends on production/runtime diagnosis or a new operational path | Emit actionable structured diagnostics without secrets; preserve correlation and failure context; verify that operators can distinguish success, retry, and terminal failure | observability plugins, session audits, production health checks | Include; prompts evidence rather than indiscriminate logging |
| `capability.release_readiness` | The task prepares a release, deployment, or production handoff | Check migrations, configuration, generated artifacts, version/revision identity, rollback path, and live verification; never equate deploy output with success | pre-release review, production audit, pre-ship gate | Include; high-impact final boundary not covered by unit tests |
| `capability.dependency_change` | Adding, removing, upgrading, or replacing a dependency | Establish necessity and compatibility, use the repository's package manager and lockfile, inspect migration notes, and test affected runtime paths | framework upgrades, package-manager detection, dependency workflows | Include; recurring source of supply-chain and compatibility regressions |
| `capability.configuration_change` | Environment variables, config files, defaults, feature flags, or secrets handling changes | Trace precedence and defaults, preserve secret boundaries, validate absent/invalid/legacy values, and document only user-visible controls | environment management, deploy readiness, profile isolation | Include; avoids “works only in my configured shell” outcomes |
| `capability.cross_platform` | Behavior depends on OS, shell, filesystem, terminal, encoding, or architecture | Use portable APIs where required, isolate platform branches, preserve path/encoding semantics, and test supported variants available locally without claiming unrun coverage | Windows recovery, cross-host installers, terminal and mobile workflows | Include; provider-neutral and especially relevant to a terminal tool |
| `capability.cli_ux` | A command, option, terminal interaction, exit status, or machine-readable output changes | Preserve non-interactive use, stable exit semantics, actionable errors, cancellation, and script-safe output; exercise both success and failure paths | CLI installers, cron diagnosis, GitHub workflows | Include; complements visual accessibility for terminal behavior |
| `capability.ui_state_completeness` | Building or changing an interactive screen or component | Cover loading, empty, error, disabled, overflow, narrow-layout, and recovery states; verify behavior rather than judging only a screenshot | mobile design, anti-UI-slop, web UAT, design-system skills | Include; distinct from keyboard/focus accessibility |
| `capability.browser_verification` | A web behavior depends on rendering, navigation, storage, console, or network activity | Verify in a real browser when available; inspect console/network failures and responsive behavior; do not substitute static source inspection for runtime evidence | browser testing with DevTools, Playwright QA, web UAT | Include; tool-conditional and evidence-oriented |
| `capability.test_quality` | Tests are flaky, overly mocked, brittle, or being substantially redesigned | Assert observable contracts, control nondeterminism, avoid implementation coupling and fake pass paths, and prove the regression test fails for the original defect when practical | mock hunter, QA stabilization, TDD and guard skills | Include; strengthens test design beyond “add the smallest test” |
| `capability.algorithmic_correctness` | Work introduces nontrivial traversal, scheduling, parsing, optimization, or large-data logic | State invariants, termination, complexity, and edge conditions before implementation; test adversarial and boundary inputs | invariant guard, complexity cuts, algorithm-first skills | Include; prevents plausible-looking loops with hidden correctness/cost defects |
| `capability.data_integrity` | A change can duplicate, lose, reorder, or corrupt user/project data without changing a schema | Define atomicity, idempotency, ordering, and recovery invariants; test interruption and repeated application where relevant | data observability, ledger, durable knowledge, SQL validation | Include; complements migration guidance for same-schema writes |
| `capability.privacy` | Work handles personal, sensitive, retained, uploaded, or logged data | Minimize collection and retention, identify exposure paths, redact diagnostics, and require explicit authority for external transmission | privacy masking, private-first memory, compliance plugins | Include; distinct from exploit-oriented security guidance |
| `capability.research_evidence` | The task asks for current facts, comparisons, recommendations, or external research | Prefer primary/current sources, attach claims to sources, separate observations from inference, record conflicts and gaps, and avoid modifying code unless separately authorized | multi-source search, deep reading, papers, fact checking | Include; mitigates unsupported confident synthesis |
| `capability.incident_response` | The user reports an active outage, severe regression, or production incident | Prioritize containment and evidence preservation, establish timeline and impact, avoid speculative destructive changes, verify recovery, then record follow-up causes | incident-response roadmap, observability, production audit | Include; urgency changes safe sequencing |
| `capability.refactoring_discipline` | The task explicitly restructures code while preserving behavior | Establish observable behavior, move one boundary at a time, avoid opportunistic feature changes, and rerun the narrowest affected checks after each move | refactoring and codebase-design workflows | Include; deserves an explicit opt-in even though core policies may also govern refactors |
| `capability.dead_code_cleanup` | Removing obsolete code, flags, dependencies, or generated scaffolding | Prove reachability/ownership, inspect dynamic and documented entry points, remove tests only with the contract, and verify packaging/import boundaries | cleanup, scaffolding removal, repository maintenance | Include; guards against destructive “unused” assumptions |
| `capability.git_hygiene` | Work includes commits, rebases, conflict resolution, release branches, or PR preparation | Preserve unrelated working-tree changes, inspect the exact diff, avoid rewriting user history without explicit approval, and report generated/binary changes | GitHub workflows, handoff, change-tracking skills | Include; directly addresses capture artifacts and mixed worktrees |
| `capability.handoff` | Work will continue across sessions, agents, or maintainers | Record current state, decisions, evidence, unresolved risks, and exact next verification without presenting assumptions as facts | feature tracking, project lore, technical change tracking | Include; supports Infinidev's session model without adding storage machinery |
| `capability.context_management` | A long task risks losing constraints or accumulating irrelevant context | Preserve authoritative requirements and durable findings, summarize evidence not raw chatter, discard stale speculation, and re-read exact sources before consequential edits | context pruning, long-session compression, project memory | Include; prompt-level mitigation, not a claim to enlarge model context |
| `capability.cost_awareness` | The task can trigger paid APIs, large downloads, long compute, or broad external operations | Estimate or expose material cost first, choose a bounded representative check, and require explicit approval before paid or unusually expensive execution | budget-aware workflows, cost-gated media, SQL/warehouse cost audits | Include; preserves user authority over external spend |
| `capability.mentoring` | The user explicitly asks to learn rather than merely receive a finished change | Match their level, explain one concept at a time, use progressive hints or examples, and distinguish teaching from authorization to modify files | Awesome Copilot “Mentoring Juniors”, teaching workflows | Include; user-visible interaction mode with a clear trigger |
| `capability.localization` | UI, messages, dates, numbers, encoding, or translated content changes | Separate translatable text from logic, preserve placeholders and locale-sensitive formatting, test fallback and expansion, and avoid inferring cultural requirements | localization roadmap, mobile/web QA, international UX collections | Include; common product boundary absent from current catalog |
| `capability.ai_system_evaluation` | Building prompts, agents, retrieval, model routing, or tool-use behavior | Define representative cases and failure labels, separate model quality from harness failures, compare identical inputs, and retain adverse examples without tuning only to them | AI evaluation ops, agent/MCP builder, session audits | Include; directly relevant to improving Infinidev itself |

This matrix added 30 distinct capabilities. It intentionally consolidates hundreds of
framework- or product-specific skills into reusable failure-mode guidance rather than making
hundreds of always-visible fragments.

## Complaint-driven design

The user's follow-up asks for prompts based on what users complain about across different
models. The catalogs already encode several recurring complaint-shaped failure modes, but they
are not a substitute for direct complaint evidence. The implementation should use a separate,
sourced complaint pass and map only prompt-addressable problems.

The initial problem-to-capability hypotheses to validate are:

| Complaint pattern to verify across model communities | Prompt mitigation candidate | Not solvable by prompt alone |
| --- | --- | --- |
| Changes unrelated code or ignores scope | `requirements_clarity`, `architecture_impact`, `git_hygiene` | Faulty edit/apply tooling |
| Claims success without running or reading evidence | `test_quality`, `release_readiness`, `research_evidence` | Missing test environment or unavailable tools |
| Overengineers small requests | `requirements_clarity`, `refactoring_discipline` | Product defaults that force heavyweight workflows |
| Forgets earlier constraints in long sessions | `context_management`, `handoff` | Hard context-window truncation |
| Repeats failed tool calls or hides errors | `error_recovery`, `observability` | Runtime retry bugs or lost tool results |
| Produces plausible but incorrect APIs/facts | `api_contracts`, `research_evidence` | Stale model knowledge without source access |
| Writes brittle tests or mocks the behavior under test | `test_quality` | Broken test infrastructure |
| Leaves partial state after cancellation or failure | `concurrency`, `resource_lifecycle`, `data_integrity` | Non-transactional external systems |
| Produces UI that looks finished but omits states/accessibility | `ui_state_completeness`, existing `accessibility`, `browser_verification` | No renderer/browser available |
| Is excessively verbose or buries the answer | A future response-communication capability, if complaint evidence shows it is distinct from configured prompt style | Provider-enforced verbosity or UI rendering |

No model-specific blame should be encoded in capability text. If the same complaint appears for
multiple models, the prompt should name the observable failure mode and desired evidence, not a
provider. If a complaint is unique to one model/version, prefer a model profile override rather
than burdening the global catalog.

## Rejected or consolidated categories

- Framework best practices (React, SwiftUI, Rails, AWS, Supabase, and similar): useful only when
  that stack is present; repository instructions or on-demand references are better than global
  workflow capabilities.
- Content, marketing, finance, travel, health, media generation, and social publishing: outside
  Infinidev's core software-engineering loop and often carry external-state or domain-risk needs.
- Offensive security and fuzzing playbooks: require explicit authorization and specialized
  safeguards; the existing security capability remains defensive and boundary-focused.
- Brainstorming, first-principles thinking, and 5 Whys as separate stylistic modes: retain only
  the concrete engineering contracts captured by requirements clarity and root-cause analysis.
- Separate unit, integration, E2E, mobile, visual, load, and BDD capabilities: consolidate common
  test-quality behavior; activate browser verification only where runtime rendering matters.
- Separate logging, tracing, metrics, and session-audit capabilities: consolidate as
  observability, leaving tool-specific implementation to the repository.
- Multi-agent orchestration, skill installation, and catalog management: Infinidev already has
  role and tool orchestration; imported instructions could conflict with its authority model.
- Token-saving tricks and model routing claims: include only evidence-based context management
  and AI-system evaluation, not universal cost or quality promises.

## Implementation sequence used

The capabilities were implemented and reviewed in small tranches:

1. correctness boundaries: `requirements_clarity`, `architecture_impact`, `api_contracts`,
   `concurrency`, `resource_lifecycle`, `error_recovery`, `test_quality`, and
   `algorithmic_correctness`;
2. production boundaries: `observability`, `release_readiness`, `dependency_change`,
   `configuration_change`, `data_integrity`, `privacy`, and `incident_response`;
3. interaction and continuity: `cli_ux`, `ui_state_completeness`, `browser_verification`,
   `cross_platform`, `localization`, `handoff`, and `context_management`;
4. task modes: `research_evidence`, `refactoring_discipline`, `dead_code_cleanup`,
   `git_hygiene`, `cost_awareness`, `mentoring`, `root_cause_analysis`, and
   `ai_system_evaluation`.

A later complaint-source pass produced four additional safeguards: `authority_boundaries`,
`verification_integrity`, `change_preservation`, and `progress_discipline`; its evidence and
mapping are recorded in `docs/model-user-complaint-research.md`. Every capability remains opt-in
and disabled by default so catalog breadth does not increase unrelated tasks' context.
