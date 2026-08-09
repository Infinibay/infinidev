"""Regression tests for command-level test runner detection."""

from infinidev.engine.guidance.test_runners import is_test_command


def test_detects_pytest_invoked_through_absolute_virtualenv_path():
    arguments = (
        '{"command":"PYTHONPATH=src '
        '/home/example/project/.venv/bin/pytest tests/test_widget.py -q"}'
    )

    assert is_test_command(arguments)


def test_does_not_treat_a_pytest_named_target_as_a_runner():
    assert not is_test_command('{"command":"cat docs/pytest"}')


def test_detects_django_runtests_script_invoked_by_python():
    assert is_test_command(
        '{"command":"python tests/runtests.py backends.base.test_creation -v 2"}'
    )


def test_does_not_authorize_mutating_a_django_runner_file():
    assert not is_test_command('{"command":"rm tests/runtests.py"}')
