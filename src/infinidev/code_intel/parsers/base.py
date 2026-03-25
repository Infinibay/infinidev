"""Base protocol for language parsers."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Tree
    from infinidev.code_intel.models import Symbol, Reference, Import


class LanguageParser(Protocol):
    """Interface that all language parsers must implement."""

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]: ...
    def extract_references(self, tree: Tree, source: bytes, file_path: str) -> list[Reference]: ...
    def extract_imports(self, tree: Tree, source: bytes, file_path: str) -> list[Import]: ...
