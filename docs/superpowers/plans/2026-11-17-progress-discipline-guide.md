# Progress discipline guide implementation plan

Use the existing opt-in `progress_discipline` capability to guide decisions after
tool results. Keep the default, tool catalog, permissions and runtime limits.
A true field or a polished final answer cannot establish that a task succeeded.

- Add a test of the principal system prompt with the capability off and on.
- Add the evaluated action guidance to the existing resource.
- Run the focused profile and command suites and the full suite.
- Run a separate paired smoke test through the production profile and report
  exact file outcomes, closure, calls and observed tokens.
