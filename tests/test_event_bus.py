"""Tests for the centralized EventBus."""

import threading

from infinidev.flows.event_listeners import EventBus


def test_subscribe_and_emit():
    bus = EventBus()
    received = []
    bus.subscribe(lambda et, pid, aid, d: received.append((et, pid, aid, d)))
    bus.emit("test_event", 1, "agent-1", {"key": "value"})
    assert len(received) == 1
    assert received[0] == ("test_event", 1, "agent-1", {"key": "value"})


def test_unsubscribe():
    bus = EventBus()
    received = []
    cb = lambda et, pid, aid, d: received.append(et)
    bus.subscribe(cb)
    bus.emit("first", 0, "", {})
    bus.unsubscribe(cb)
    bus.emit("second", 0, "", {})
    assert received == ["first"]


def test_has_subscribers():
    bus = EventBus()
    assert not bus.has_subscribers
    cb = lambda *a: None
    bus.subscribe(cb)
    assert bus.has_subscribers
    bus.unsubscribe(cb)
    assert not bus.has_subscribers


def test_multiple_subscribers():
    bus = EventBus()
    results_a, results_b = [], []
    bus.subscribe(lambda et, *_: results_a.append(et))
    bus.subscribe(lambda et, *_: results_b.append(et))
    bus.emit("evt", 0, "", {})
    assert results_a == ["evt"]
    assert results_b == ["evt"]


def test_subscriber_error_does_not_crash():
    bus = EventBus()
    received = []

    def bad_cb(*_):
        raise RuntimeError("boom")

    bus.subscribe(bad_cb)
    bus.subscribe(lambda et, *_: received.append(et))
    bus.emit("evt", 0, "", {})
    # Second subscriber should still receive the event
    assert received == ["evt"]


def test_thread_safety():
    bus = EventBus()
    received = []
    lock = threading.Lock()

    def cb(et, pid, aid, data):
        with lock:
            received.append(et)

    bus.subscribe(cb)

    threads = []
    for i in range(20):
        t = threading.Thread(target=bus.emit, args=(f"evt_{i}", 0, "", {}))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == 20


def test_singleton_import():
    """The module-level event_bus should be the same object across imports."""
    from infinidev.flows.event_listeners import event_bus as bus1
    from infinidev.flows.event_listeners import event_bus as bus2
    assert bus1 is bus2
