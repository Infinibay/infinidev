"""Tests for IndexQueue background indexing."""

import os
import threading

import pytest

from infinidev.cli.index_queue import IndexQueue


class TestIndexQueue:
    """Tests for IndexQueue."""

    def test_start_stop(self):
        """Queue starts and stops cleanly."""
        q = IndexQueue(project_id=1)
        q.start()
        assert q.is_running()
        q.stop()
        assert not q.is_running()

    def test_enqueue_processes(self, workspace_dir, temp_db):
        """Enqueued files get indexed."""
        # Create a Python file to index
        py_file = workspace_dir / "hello.py"
        py_file.write_text("def greet():\n    return 'hello'\n")

        indexed = threading.Event()
        q = IndexQueue(project_id=1, post_index_callback=lambda _: indexed.set())
        q.start()
        q.enqueue(str(py_file))

        assert indexed.wait(5.0), "index worker did not process the file"
        q.stop()

        # Check that the file was indexed
        from infinidev.code_intel.index import get_file_hash
        h = get_file_hash(1, str(py_file))
        assert h is not None  # file was indexed

    def test_post_index_callback(self, workspace_dir, temp_db):
        """Post-index callback fires after successful indexing."""
        py_file = workspace_dir / "callback_test.py"
        py_file.write_text("x = 1\n")

        called_with = []
        indexed = threading.Event()

        def record(path: str) -> None:
            called_with.append(path)
            indexed.set()

        q = IndexQueue(project_id=1, post_index_callback=record)
        q.start()
        q.enqueue(str(py_file))

        assert indexed.wait(5.0), "post-index callback did not fire"
        q.stop()

        assert len(called_with) == 1
        assert str(py_file) in called_with[0]

    def test_hash_skip(self, workspace_dir, temp_db):
        """Same file enqueued twice only indexes once (hash skip)."""
        py_file = workspace_dir / "skip.py"
        py_file.write_text("a = 1\n")

        call_count = []
        first_indexed = threading.Event()

        def record(_: str) -> None:
            call_count.append(1)
            first_indexed.set()

        q = IndexQueue(project_id=1, post_index_callback=record)
        q.start()
        q.enqueue(str(py_file))
        assert first_indexed.wait(5.0), "first index did not complete"
        q.enqueue(str(py_file))  # same content, should skip
        q.stop()

        assert len(call_count) == 1  # only indexed once

    def test_config_file_tracked(self, workspace_dir, temp_db):
        """Config files are tracked but not parsed for symbols."""
        toml_file = workspace_dir / "pyproject.toml"
        toml_file.write_text('[project]\nname = "test"\n')

        indexed = threading.Event()
        q = IndexQueue(project_id=1, post_index_callback=lambda _: indexed.set())
        q.start()
        q.enqueue(str(toml_file))
        assert indexed.wait(5.0), "config file was not indexed"
        q.stop()

        from infinidev.code_intel.index import get_file_hash
        h = get_file_hash(1, str(toml_file))
        assert h is not None  # tracked
