"""Builder exceptions."""


class DependencyCycle(ValueError):
    """Raised when a build graph is not acyclic."""
