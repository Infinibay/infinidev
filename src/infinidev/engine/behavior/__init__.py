"""Modular AI behavior-checker subsystem.

After each model message, registered checkers inspect the message + last
few history entries and emit punish/promote deltas. Scores per agent are
shown to the user as banners.
"""

from infinidev.engine.behavior.scorer import BehaviorScorer
from infinidev.engine.behavior.checker_base import BehaviorChecker, Verdict

__all__ = ["BehaviorScorer", "BehaviorChecker", "Verdict"]
