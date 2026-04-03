"""Unit tests for exploration tree models and propagation logic."""

import pytest

from infinidev.engine.tree.models import (
    Blocker,
    CONF_RANK,
    Fact,
    Question,
    STATE_RANK,
    TreeNode,
    TreeState,
    propagate,
    select_next_node,
)


# ── TreeNode basics ───────────────────────────────────────────────────────────


class TestTreeNode:
    def test_create_minimal(self):
        node = TreeNode(id="1", problem_statement="Test problem")
        assert node.id == "1"
        assert node.state == "pending"
        assert node.logic == "AND"
        assert node.depth == 0
        assert node.children == []
        assert node.facts == []

    def test_add_child(self):
        root = TreeNode(id="1", problem_statement="Root")
        child1 = root.add_child("Sub-problem 1")
        child2 = root.add_child("Sub-problem 2", logic="OR")

        assert len(root.children) == 2
        assert child1.id == "1.1"
        assert child2.id == "1.2"
        assert child1.depth == 1
        assert child2.logic == "OR"

    def test_add_child_nested(self):
        root = TreeNode(id="1", problem_statement="Root")
        child = root.add_child("Child")
        grandchild = child.add_child("Grandchild")

        assert grandchild.id == "1.1.1"
        assert grandchild.depth == 2

    def test_is_resolved(self):
        node = TreeNode(id="1", problem_statement="Test")

        for state in ("solvable", "unsolvable", "mitigable", "discarded"):
            node.state = state
            assert node.is_resolved(), f"{state} should be resolved"

        for state in ("pending", "exploring", "needs_decision", "needs_experiment"):
            node.state = state
            assert not node.is_resolved(), f"{state} should not be resolved"

    def test_collect_inheritable_facts(self):
        node = TreeNode(id="1", problem_statement="Test", facts=[
            Fact(content="Initial fact", source="initial"),
            Fact(content="Discovered fact", source="discovered", evidence="found it"),
            Fact(content="Inherited fact", source="inherited"),
        ])

        inheritable = node.collect_inheritable_facts()
        assert len(inheritable) == 1
        assert inheritable[0].content == "Discovered fact"


# ── Propagation — AND logic ──────────────────────────────────────────────────


class TestPropagationAND:
    def _make_and_tree(self, child_states: list[str]) -> TreeNode:
        """Helper: create a root with AND children in given states."""
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        for i, state in enumerate(child_states, 1):
            child = TreeNode(
                id=f"1.{i}",
                problem_statement=f"Child {i}",
                state=state,
                confidence="medium",
                depth=1,
            )
            root.children.append(child)
        return root

    def test_all_solvable(self):
        root = self._make_and_tree(["solvable", "solvable"])
        propagate(root)
        assert root.state == "solvable"

    def test_one_unsolvable(self):
        root = self._make_and_tree(["solvable", "unsolvable"])
        propagate(root)
        assert root.state == "unsolvable"

    def test_one_mitigable(self):
        root = self._make_and_tree(["solvable", "mitigable"])
        propagate(root)
        assert root.state == "mitigable"

    def test_one_pending_keeps_exploring(self):
        root = self._make_and_tree(["solvable", "pending"])
        propagate(root)
        assert root.state == "exploring"

    def test_worst_confidence(self):
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        root.children.append(TreeNode(
            id="1.1", problem_statement="A", state="solvable",
            confidence="high", depth=1,
        ))
        root.children.append(TreeNode(
            id="1.2", problem_statement="B", state="solvable",
            confidence="low", depth=1,
        ))
        propagate(root)
        assert root.confidence == "low"

    def test_constraints_propagate_up(self):
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        child = TreeNode(
            id="1.1", problem_statement="A", state="solvable",
            constraints=["Must use Python 3.12+"], depth=1,
        )
        root.children.append(child)
        propagate(root)
        assert "Must use Python 3.12+" in root.constraints

    def test_blockers_propagate_up(self):
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        child = TreeNode(
            id="1.1", problem_statement="A", state="mitigable",
            external_blockers=[Blocker(description="Need API key", blocker_type="api")],
            depth=1,
        )
        root.children.append(child)
        propagate(root)
        assert len(root.external_blockers) == 1
        assert root.external_blockers[0].description == "Need API key"

    def test_all_discarded_means_unsolvable(self):
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        root.children.append(TreeNode(
            id="1.1", problem_statement="A", state="discarded", depth=1,
        ))
        propagate(root)
        assert root.state == "unsolvable"

    def test_no_duplicate_constraints(self):
        root = TreeNode(id="1", problem_statement="Root", logic="AND",
                        constraints=["Existing"])
        child1 = TreeNode(
            id="1.1", problem_statement="A", state="solvable",
            constraints=["Existing", "New"], depth=1,
        )
        root.children.append(child1)
        propagate(root)
        assert root.constraints.count("Existing") == 1
        assert "New" in root.constraints


# ── Propagation — OR logic ───────────────────────────────────────────────────


class TestPropagationOR:
    def _make_or_tree(self, child_states: list[str]) -> TreeNode:
        root = TreeNode(id="1", problem_statement="Root", logic="OR")
        for i, state in enumerate(child_states, 1):
            child = TreeNode(
                id=f"1.{i}",
                problem_statement=f"Child {i}",
                state=state,
                confidence="medium",
                depth=1,
            )
            root.children.append(child)
        return root

    def test_one_solvable_enough(self):
        root = self._make_or_tree(["unsolvable", "solvable"])
        propagate(root)
        assert root.state == "solvable"

    def test_all_unsolvable(self):
        root = self._make_or_tree(["unsolvable", "unsolvable"])
        propagate(root)
        assert root.state == "unsolvable"

    def test_best_state_wins(self):
        root = self._make_or_tree(["unsolvable", "mitigable", "solvable"])
        propagate(root)
        assert root.state == "solvable"

    def test_all_pending_keeps_exploring(self):
        root = self._make_or_tree(["pending", "pending"])
        propagate(root)
        assert root.state == "exploring"

    def test_one_resolved_one_pending(self):
        root = self._make_or_tree(["solvable", "pending"])
        propagate(root)
        assert root.state == "solvable"

    def test_best_confidence(self):
        root = TreeNode(id="1", problem_statement="Root", logic="OR")
        root.children.append(TreeNode(
            id="1.1", problem_statement="A", state="solvable",
            confidence="low", depth=1,
        ))
        root.children.append(TreeNode(
            id="1.2", problem_statement="B", state="solvable",
            confidence="high", depth=1,
        ))
        propagate(root)
        assert root.confidence == "high"


# ── Deep propagation ─────────────────────────────────────────────────────────


class TestDeepPropagation:
    def test_three_level_and(self):
        """Root(AND) -> Child(AND) -> Grandchildren."""
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        child = TreeNode(id="1.1", problem_statement="Child", logic="AND", depth=1)
        gc1 = TreeNode(id="1.1.1", problem_statement="GC1", state="solvable",
                        confidence="high", depth=2)
        gc2 = TreeNode(id="1.1.2", problem_statement="GC2", state="solvable",
                        confidence="medium", depth=2)
        child.children = [gc1, gc2]
        root.children = [child]

        propagate(root)
        assert child.state == "solvable"
        assert child.confidence == "medium"
        assert root.state == "solvable"
        assert root.confidence == "medium"

    def test_mixed_and_or(self):
        """Root(AND) -> [Child1(OR), Child2(solvable)]."""
        root = TreeNode(id="1", problem_statement="Root", logic="AND")
        child1 = TreeNode(id="1.1", problem_statement="Alt paths", logic="OR", depth=1)
        child1.children = [
            TreeNode(id="1.1.1", problem_statement="Path A", state="unsolvable",
                     confidence="high", depth=2),
            TreeNode(id="1.1.2", problem_statement="Path B", state="solvable",
                     confidence="medium", depth=2),
        ]
        child2 = TreeNode(id="1.2", problem_statement="Must do", state="solvable",
                          confidence="high", depth=1)
        root.children = [child1, child2]

        propagate(root)
        # OR child: best is solvable
        assert child1.state == "solvable"
        # AND root: both solvable
        assert root.state == "solvable"
        # Confidence: min(medium, high) = medium
        assert root.confidence == "medium"


# ── TreeState ────────────────────────────────────────────────────────────────


class TestTreeState:
    @pytest.fixture
    def tree(self) -> TreeState:
        root = TreeNode(id="1", problem_statement="Root")
        c1 = root.add_child("Child 1")
        c2 = root.add_child("Child 2")
        c1.add_child("Grandchild 1.1")
        return TreeState(root=root)

    def test_get_node_root(self, tree: TreeState):
        assert tree.get_node("1") is not None
        assert tree.get_node("1").problem_statement == "Root"

    def test_get_node_deep(self, tree: TreeState):
        assert tree.get_node("1.1.1") is not None
        assert tree.get_node("1.1.1").problem_statement == "Grandchild 1.1"

    def test_get_node_missing(self, tree: TreeState):
        assert tree.get_node("9.9.9") is None

    def test_get_path_to_node(self, tree: TreeState):
        path = tree.get_path_to_node("1.1.1")
        assert len(path) == 3
        assert path[0].id == "1"
        assert path[1].id == "1.1"
        assert path[2].id == "1.1.1"

    def test_get_path_to_root(self, tree: TreeState):
        path = tree.get_path_to_node("1")
        assert len(path) == 1

    def test_get_path_missing(self, tree: TreeState):
        path = tree.get_path_to_node("9.9")
        assert path == []

    def test_get_siblings(self, tree: TreeState):
        siblings = tree.get_siblings("1.1")
        assert len(siblings) == 1
        assert siblings[0].id == "1.2"

    def test_get_siblings_root(self, tree: TreeState):
        siblings = tree.get_siblings("1")
        assert siblings == []

    def test_get_pending_nodes(self, tree: TreeState):
        pending = tree.get_pending_nodes()
        # Only leaf nodes: 1.1.1 (leaf under 1.1) and 1.2 (leaf)
        # Root and 1.1 have children, so they're excluded
        assert len(pending) == 2
        ids = {n.id for n in pending}
        assert "1.1.1" in ids
        assert "1.2" in ids

    def test_get_pending_filters_resolved(self, tree: TreeState):
        tree.get_node("1.1.1").state = "solvable"
        tree.get_node("1.2").state = "exploring"
        pending = tree.get_pending_nodes()
        # 1.1.1 resolved, 1.2 exploring — no pending leaves
        assert len(pending) == 0

    def test_count_nodes(self, tree: TreeState):
        assert tree.count_nodes() == 4

    def test_count_empty(self):
        tree = TreeState()
        assert tree.count_nodes() == 0


# ── Node selection ───────────────────────────────────────────────────────────


class TestSelectNextNode:
    def test_empty_tree(self):
        tree = TreeState()
        assert select_next_node(tree) is None

    def test_all_resolved(self):
        root = TreeNode(id="1", problem_statement="Root", state="solvable")
        tree = TreeState(root=root)
        assert select_next_node(tree) is None

    def test_single_pending(self):
        root = TreeNode(id="1", problem_statement="Root")
        tree = TreeState(root=root)
        assert select_next_node(tree).id == "1"

    def test_prefers_pivot_questions(self):
        root = TreeNode(id="1", problem_statement="Root", state="solvable")
        c1 = TreeNode(id="1.1", problem_statement="No pivots", depth=1)
        c2 = TreeNode(id="1.2", problem_statement="Has pivot", depth=1,
                       questions=[Question(content="Key question?", question_type="pivot")])
        root.children = [c1, c2]
        tree = TreeState(root=root)

        selected = select_next_node(tree)
        assert selected.id == "1.2"

    def test_prefers_shallower(self):
        root = TreeNode(id="1", problem_statement="Root", state="solvable")
        deep = TreeNode(id="1.1.1", problem_statement="Deep", depth=2)
        shallow = TreeNode(id="1.2", problem_statement="Shallow", depth=1)
        root.children = [
            TreeNode(id="1.1", problem_statement="Mid", state="solvable", depth=1,
                     children=[deep]),
            shallow,
        ]
        tree = TreeState(root=root)

        selected = select_next_node(tree)
        assert selected.id == "1.2"

    def test_prefers_more_facts(self):
        root = TreeNode(id="1", problem_statement="Root", state="solvable")
        c1 = TreeNode(id="1.1", problem_statement="No facts", depth=1)
        c2 = TreeNode(id="1.2", problem_statement="Has facts", depth=1,
                       facts=[Fact(content="fact1"), Fact(content="fact2")])
        root.children = [c1, c2]
        tree = TreeState(root=root)

        selected = select_next_node(tree)
        assert selected.id == "1.2"

    def test_includes_needs_experiment(self):
        root = TreeNode(id="1", problem_statement="Root", state="solvable")
        c1 = TreeNode(id="1.1", problem_statement="Experiment needed",
                       state="needs_experiment", depth=1)
        root.children = [c1]
        tree = TreeState(root=root)

        selected = select_next_node(tree)
        assert selected.id == "1.1"


# ── Fact model ───────────────────────────────────────────────────────────────


class TestFact:
    def test_defaults(self):
        f = Fact(content="Found X")
        assert f.source == "discovered"
        assert f.confidence == "medium"
        assert f.evidence == ""
        assert f.source_tool == ""

    def test_with_evidence(self):
        f = Fact(
            content="Function exists",
            evidence="def foo(): ...",
            source_tool="code_search",
            confidence="high",
        )
        assert f.source_tool == "code_search"


# ── Question model ───────────────────────────────────────────────────────────


class TestQuestion:
    def test_defaults(self):
        q = Question(content="Is X possible?")
        assert q.question_type == "informational"
        assert q.answer is None

    def test_pivot(self):
        q = Question(content="Does API support Y?", question_type="pivot")
        assert q.question_type == "pivot"


# ── Blocker model ────────────────────────────────────────────────────────────


class TestBlocker:
    def test_defaults(self):
        b = Blocker(description="Need sudo")
        assert b.blocker_type == "unknown"
        assert b.workaround is None

    def test_with_workaround(self):
        b = Blocker(
            description="Rate limited",
            blocker_type="api",
            workaround="Use batch endpoint",
        )
        assert b.workaround == "Use batch endpoint"


# ── State/Confidence ranking ─────────────────────────────────────────────────


class TestRanking:
    def test_state_order(self):
        assert STATE_RANK["unsolvable"] < STATE_RANK["mitigable"] < STATE_RANK["solvable"]

    def test_conf_order(self):
        assert CONF_RANK["low"] < CONF_RANK["medium"] < CONF_RANK["high"]
