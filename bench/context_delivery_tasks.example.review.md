# Context-delivery fixture review

The six validation tasks form two family-atomic groups. Each group keeps the
request, repository, bug, verifier, and required evidence constant while only
the full-corpus placement changes between front, middle, and end.

- `rollback-order`: the implementation violates dependency-safe compensation
  order. The focused verifier fails on forward rollback and passes only when
  completed steps unwind in reverse order without changing the API.
- `tenant-cache-key`: the implementation stores composite tenant/user keys but
  invalidates a scalar key. The verifier also retains a same-user entry in a
  second tenant, preventing a broad or cross-tenant deletion from passing.

Both requests are intentionally concise but actionable. The tests are visible
to every condition, so success is not a hidden-test guessing exercise. The
additional files are plausible lexical distractors and contain no alternate
solution. These fixtures are approved for harness validation, but two task
families remain a pilot rather than evidence of general repository-agent
quality.
