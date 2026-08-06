"""Semantic vocabulary for the Graph engine.

Mirrors ``prompts/analyst/planning_vocabulary.py``: a small set of
meaning-loaded definitions the model reads before working with the graph, so
every node, edge and status carries one shared meaning. The words are chosen
to name the *role* a thing plays in reasoning, not its storage shape.
"""

from __future__ import annotations


GRAPH_VOCABULARY = """\
## Graph vocabulary

The graph is a map of a piece of work as it is actually understood — intent,
questions, decisions and evidence in one place, each with a clear role.

A **Requirement** is something the user needs to be true. It is owned by the
user's words and does not stop being required until it is satisfied or the
user removes it.

A **Question** is an open uncertainty worth resolving. While it stays open it
is honest to say so; answering it turns it into knowledge.

A **Hypothesis** is a working belief acted on before it is confirmed. It is
useful precisely because it can be wrong — treat it as provisional, and mark
it confirmed, rejected or inconclusive once evidence speaks.

A **Decision** is a choice that closes a fork. It is recorded with its reason
so a later reader knows why this path, and not another, was taken.

A **Work** node is one unit of doing with a verifiable outcome. It is the part
of the graph that gets executed.

A **Verification** is a check that an outcome was really reached. It exists to
catch the difference between "done" and "done and shown to be done".

**Evidence** is what was observed — a tool result, a check that ran, state
inspected in the workspace. A remembered path, API or result is a hypothesis
until current evidence confirms it.

A **Blocker** names what is stopping progress and why. Naming it is itself
progress: a run that surfaces a blocker honestly is more useful than one that
pretends to be moving.

**Checkpoint** is what you leave behind before stepping away from a node: what
was being attempted, what was learned, what is still missing, and the next
safe step. A good checkpoint lets anyone — including a future you — resume
without re-deriving everything.

**Lifecycle** says where a node stands: proposed, ready, active, suspended,
resolved or abandoned. **Verdict** says what is believed about it: confirmed,
rejected, inconclusive, or still unknown. **Freshness** says whether its
grounding still holds: current, stale, or invalidated. A node can be resolved
yet stale once the goal or the code moves under it — the three ideas are kept
apart on purpose.

Edges carry meaning too. **Requires** is the one hard relation: it orders
execution and must never form a loop. **Supports**, **contradicts**,
**supersedes** and **invalidates** describe how ideas relate; they may cross
and re-cross, because understanding is not a straight line.

Keep three sources of authority distinct. ``USER_LITERAL`` records what the
user said or confirmed. ``DERIVED`` records the graph's own proposals —
nodes, checks and routes. ``OBSERVED_EVIDENCE`` records facts established
during the run. Derived material can guide the work but cannot expand the
goal or acquire user authority.
"""


__all__ = ["GRAPH_VOCABULARY"]
