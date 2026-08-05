# MiniMax M3 prompt-comprehension pilot review

## Scope and provenance

MiniMax-M3 reviewed the 16-family/48-case blind semantic pilot through
`https://api.minimax.io/v1/chat/completions`. Every family used a fresh stateless request containing
one user message, no system prompt, no conversation history, and no authored interpretation key.
Calls were sequential with a one-second inter-call delay. The API key was supplied interactively and
was not persisted in repository artifacts.

- Successful family reviews: 16/16
- API attempts: 17 (the first unparseable attempt was retained as a typed failure and then retried)
- Prompt tokens: 16,736
- Completion tokens, including provider reasoning: 51,544
- Total tokens: 68,280
- Successful-call latency: 544.53 seconds total; 30.46 seconds median; 74.48 seconds maximum
- Provider-reported model identity: `MiniMax-M3`
- Review verdicts: 10 `accept`, 6 `revise`, 0 `reject`
- Mechanically ready for adjudication: 9 families
- Families requiring review or rubric adjudication: 7

The full literal responses and parsed reviews are retained under
`bench/runs/minimax-m3-comprehension-review/`. No verdict in this report changes a case from `draft`.

## Likely instrument defects

These findings point to generator changes rather than one-off edits to materialized JSONL:

1. **Nested-scope code-review contrast:** “otherwise that particular prohibition does not apply” is
   unnatural and does not state whether fixing becomes affirmatively authorized or merely ceases to
   be prohibited. Rewrite the contrast with an explicit permission boundary.
2. **Planning priority family:** “A default says…” is benchmark-shaped language, and the adversarial
   prompt explicitly announces that its priorities cannot both be achieved. Replace it with realistic
   competing instructions without telegraphing the conflict.
3. **Testing precedence family:** the first/last pair changes tense and phrasing as well as position.
   Make the user and repository clauses byte-equivalent apart from order.
4. **Web-research vagueness family:** the contrast changes the primary task in addition to changing
   whether the ambiguity belongs to the user. Keep the same library-research task across variants.
5. **Quantifier family:** “some focused requirements” changes both quantifier and modifier. Use
   “some explicitly named requirements” so only quantifier strength changes. Either narrow the family
   name to the tested quantifiers or add separately controlled `any` and `most` families.

## Findings that require adjudication, not automatic revision

1. **Incremental-execution family:** MiniMax treated `big-bang` and `checkpointed` as an intended
   equivalent pair. The registry labels `big-bang` as the contrast and `checkpointed` as equivalent to
   the two-increment anchor. Its observation that checkpointed overlaps the anchor is therefore
   expected, not a defect. The review instruction or dossier should make pairwise relation targets
   harder to misread.
2. **Missing-referent family:** MiniMax accepted the family but failed the global
   `requests_are_self_contained` check because the contrast is intentionally unresolvable. This exposes
   a rubric defect: adversarial missing-context cases should be judged on whether the missing context
   is intentional and isolated, not required to be self-contained.
3. **Diversity concerns:** ten reviews mention repeated scaffolding. Most repetition is necessary for
   causal control and is not itself a rejection reason. The useful signal is narrower: findings should
   distinguish expected within-family control from excessive reuse across independent families and
   domains.

## What the pilot teaches us about the review protocol

Boolean checks are not universally applicable. `requests_are_self_contained` and
`authorization_is_unambiguous` may deliberately be false in controlled adversarial variants. The
review schema should add `pass`, `fail`, and `not_applicable_by_design`, with a required explanation
for the latter. It should also ask separately whether the intended defect is isolated from accidental
defects.

The packet currently exposes `intended_relation` per variant, but the model still misread at least one
relationship. A family-level relation map should explicitly identify anchor-equivalent and
anchor-contrast comparisons. That improves review accuracy without exposing the hidden interpretation
keys.

## Recommended next action

Revise the generator and review rubric for the five likely instrument defects and the two protocol
defects above. Regenerate the full draft and a new pilot with a new dataset hash. Then repeat only the
affected pilot families with MiniMax-M3 or another independent reviewer before spending calls on all
224 families. The nine mechanically ready families remain evidence for adjudication, not approved
dataset rows.

## Corrected-family rerun

The five likely instrument defects and two review-protocol defects were corrected in the generator
and rubric. The original reviewed dataset and packet were archived beside the first-run evidence
before regeneration. The new dataset SHA-256 is
`9f492dc5b5d234d67aae23f3890ba4ae56de9f7dc0d9308ad8c1c4f9a354adf4`.

MiniMax-M3 then re-reviewed only the seven disputed families using the new key-blind packet:

- Valid completed reviews: 7/7
- API attempts: 8; one response was rejected because the model replaced the supplied dataset hash
  with the well-known empty-file SHA-256, then the exact family was retried successfully
- Verdicts: 7 `accept`, 0 `revise`, 0 `reject`
- Check results: 40 `pass`, 2 `not_applicable_by_design`, 0 `fail`
- Prompt tokens: 9,102
- Completion tokens, including provider reasoning: 20,932
- Total tokens: 30,034
- Successful-call latency: 284.34 seconds total; 30.05 seconds median; 74.96 seconds maximum

The corrected rubric handled the intentionally unresolved referent case as intended: self-containment
and authorization were marked `not_applicable_by_design`, while the family itself was accepted because
the defect was deliberate and isolated. The explicit relation map also prevented the earlier
misclassification of the incremental-execution contrast.

This is strong evidence that the seven targeted corrections addressed the observed pilot failures.
It is not independent approval of the other nine families under the new dataset hash, and it does not
approve the full 224-family battery. A final adjudicator should inspect the seven new reviews and the
nine original reviews before deciding whether the unchanged families need hash-current re-review.
