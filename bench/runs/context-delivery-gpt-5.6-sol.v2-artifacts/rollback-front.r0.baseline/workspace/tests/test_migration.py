from src.migration import Migration


class Step:
    def __init__(self, name, events, *, fails=False):
        self.name = name
        self.events = events
        self.fails = fails

    def apply(self):
        self.events.append(f"apply:{self.name}")
        if self.fails:
            raise RuntimeError(self.name)

    def rollback(self):
        self.events.append(f"rollback:{self.name}")


def test_failed_step_rolls_back_completed_steps_in_reverse_order():
    events = []
    migration = Migration([
        Step("schema", events),
        Step("data", events),
        Step("index", events, fails=True),
    ])

    try:
        migration.apply()
    except RuntimeError:
        pass

    assert events == [
        "apply:schema",
        "apply:data",
        "apply:index",
        "rollback:data",
        "rollback:schema",
    ]
