from dag_builder.state import BuildState


def test_only_successful_builds_update_fingerprints() -> None:
    state = BuildState()

    state.mark_success("core", "v1")

    assert state.fingerprints == {"core": "v1"}
    assert state.changed({"core": "v1", "app": "v1"}) == {"app"}
