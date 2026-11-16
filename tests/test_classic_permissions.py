"""Classic approval prompts are serviced while work runs and fail closed on cancel."""
from __future__ import annotations

import threading

import pytest

from infinidev.cli import classic_renderer as cr


def test_cancel_releases_claimed_permission_and_discards_stale_request():
    cancelled = threading.Event()
    pq = cr.PermissionQueue(is_cancelled=cancelled.is_set)
    answers = []
    worker = threading.Thread(target=lambda: answers.append(pq.request('fake', 'Deny', '')))
    worker.start()
    request = pq._q.get(timeout=1)
    try:
        cancelled.set()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert answers == [False]
        cancelled.clear()
        request.resolve(True)
        assert request.result is False
        assert request.done.is_set()
        cancelled.set()
        assert pq.request('late', 'Already cancelled', '') is False
        assert pq.pending() is None
    finally:
        request.result = False
        request.done.set()
        worker.join(timeout=1)


def test_permission_resolution_is_single_use_and_denied_requests_are_skipped():
    pq = cr.PermissionQueue()
    request = cr._PermissionRequest('fake', 'Request', '')
    pq._q.put(request)
    request.resolve(False)
    request.resolve(True)
    assert request.result is False
    assert pq.pending() is None


def test_cancel_before_enqueue_remains_denied_after_flag_reset(monkeypatch):
    cancelled = threading.Event()
    reached_put = threading.Event()
    release_put = threading.Event()
    pq = cr.PermissionQueue(is_cancelled=cancelled.is_set)
    original_put = pq._q.put
    requests = []
    answers = []
    def held_put(request):
        requests.append(request)
        reached_put.set()
        assert release_put.wait(timeout=2)
        original_put(request)
    monkeypatch.setattr(pq._q, 'put', held_put)
    worker = threading.Thread(target=lambda: answers.append(pq.request('old', 'Old request', '')))
    worker.start()
    try:
        assert reached_put.wait(timeout=1)
        cancelled.set()
        pq.deny_pending()
        cancelled.clear()
        release_put.set()
        worker.join(timeout=.2)
        requests[0].resolve(True)
        assert not worker.is_alive()
        assert answers == [False]
        assert requests[0].result is False
        assert pq.pending() is None
    finally:
        release_put.set()
        for request in requests:
            request.resolve(False)
        worker.join(timeout=1)


def test_ui_constructor_failure_denies_pending_without_starting_task(monkeypatch):
    from prompt_toolkit import application
    cancelled = threading.Event()
    pq = cr.PermissionQueue(is_cancelled=cancelled.is_set)
    queued = threading.Event()
    request = cr._PermissionRequest('fake', 'Preexisting request', '')
    pq._q.put(request)
    class Engine:
        def cancel(self):
            cancelled.set()
    def broken_app(**kwargs):
        raise RuntimeError('synthetic constructor failure')
    monkeypatch.setattr(application, 'Application', broken_app)
    status = cr.SessionStatus()
    with pytest.raises(RuntimeError, match='synthetic constructor failure'):
        cr.run_with_live_status(Engine(), lambda: queued.set(), status,
                                permission_queue=pq)
    assert not queued.is_set()
    assert request.done.is_set()
    assert request.result is False
    assert status.run_started_at is None


@pytest.mark.parametrize('answer,allowed', [('y\r', True), ('yes\r', True),
                                           ('n\r', False), ('\r', False),
                                           ('anything\r', False), ('\x04', False),
                                           ('\x03', False)])
def test_active_ui_resolves_permission_before_worker_finishes(monkeypatch, answer, allowed):
    from prompt_toolkit import application
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    cancelled = threading.Event()
    visible = threading.Event()
    finished = threading.Event()
    pq = cr.PermissionQueue(is_cancelled=cancelled.is_set)

    class Engine:
        is_cancelled = False
        def cancel(self):
            self.is_cancelled = True
            cancelled.set()

    original = application.Application
    def make_app(**kwargs):
        callback = kwargs.get('before_render')
        def before_render(app):
            if callback:
                callback(app)
            for window in app.layout.find_all_windows():
                control = window.content
                if hasattr(control, 'text'):
                    text = control.text() if callable(control.text) else control.text
                    if 'OFFLINE approval witness' in str(text):
                        visible.set()
        kwargs['before_render'] = before_render
        return original(**kwargs)
    monkeypatch.setattr(application, 'Application', make_app)

    def task():
        result = pq.request('fake', 'OFFLINE approval witness', 'No filesystem action')
        finished.set()
        return result

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        def respond():
            if visible.wait(timeout=2):
                assert not finished.is_set()
                pipe.send_text(answer)
        sender = threading.Thread(target=respond)
        sender.start()
        watchdog = threading.Timer(3, cancelled.set)
        watchdog.start()
        try:
            result = cr.run_with_live_status(Engine(), task, cr.SessionStatus(),
                                             permission_queue=pq)
        finally:
            cancelled.set()
            watchdog.cancel()
            sender.join(timeout=3)
    assert visible.is_set()
    assert finished.is_set()
    assert result is allowed


def test_ui_failure_denies_and_releases_permission(monkeypatch):
    from prompt_toolkit import application
    cancelled = threading.Event()
    pq = cr.PermissionQueue(is_cancelled=cancelled.is_set)
    answers = []
    class Engine:
        def cancel(self):
            cancelled.set()
    class BrokenApp:
        loop = None
        def __init__(self, **kwargs):
            pass
        def run(self, **kwargs):
            raise EOFError('synthetic input closed')
    monkeypatch.setattr(application, 'Application', BrokenApp)
    with pytest.raises(EOFError, match='synthetic input closed'):
        cr.run_with_live_status(Engine(), lambda: answers.append(
            pq.request('fake', 'Deny on input failure', '')), cr.SessionStatus(),
            permission_queue=pq)
    for _ in range(100):
        if answers:
            break
        threading.Event().wait(.01)
    assert answers == [False]
    assert pq.pending() is None


def test_ui_failure_after_claim_releases_request_outside_queue(monkeypatch):
    from prompt_toolkit import application
    cancelled = threading.Event()
    pq = cr.PermissionQueue(is_cancelled=cancelled.is_set)
    answers = []
    claimed = []
    original_pending = pq.pending
    def pending():
        request = original_pending()
        if request is not None:
            claimed.append(request)
        return request
    monkeypatch.setattr(pq, 'pending', pending)
    class Engine:
        def cancel(self):
            cancelled.set()
    class BrokenApp:
        loop = None
        def __init__(self, **kwargs):
            self.layout = kwargs['layout']
            self.before_render = kwargs['before_render']
        def run(self, **kwargs):
            for _ in range(100):
                self.before_render(self)
                if claimed:
                    break
                threading.Event().wait(.01)
            assert len(claimed) == 1
            assert pq._q.empty()
            raise OSError('synthetic failure after claim')
    monkeypatch.setattr(application, 'Application', BrokenApp)
    with pytest.raises(OSError, match='synthetic failure after claim'):
        cr.run_with_live_status(Engine(), lambda: answers.append(
            pq.request('fake', 'Claimed permission', '')), cr.SessionStatus(),
            permission_queue=pq)
    assert answers == [False]
    assert claimed[0].done.is_set()
