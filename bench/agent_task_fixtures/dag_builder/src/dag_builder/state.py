"""Successful build fingerprints."""

from dataclasses import dataclass, field


@dataclass
class BuildState:
    fingerprints: dict[str, str] = field(default_factory=dict)

    def mark_attempt(self, name: str, fingerprint: str) -> None:
        self.fingerprints[name] = fingerprint

    def changed(self, current: dict[str, str]) -> set[str]:
        return {
            name
            for name, fingerprint in current.items()
            if self.fingerprints.get(name) != fingerprint
        }
