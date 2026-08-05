"""Apply migrations and compensate completed steps after a failure."""


class Migration:
    def __init__(self, steps):
        self.steps = list(steps)
        self._applied = []

    def apply(self) -> None:
        try:
            for step in self.steps:
                step.apply()
                self._applied.append(step)
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        for step in self._applied:
            step.rollback()
        self._applied.clear()
