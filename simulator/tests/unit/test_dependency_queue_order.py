"""
A dependency serves its calls in the order they arrived.

When a dependency finishes a call, the worker it frees goes to the call that has
waited longest in its queue. A caller that reacts to the answer by sending another
call at the same instant does not get that worker: its new call joins the queue
behind the calls already there.

Example, used by the first test. `db` has one worker, takes 10 ms per call and fails
every call. `front` has one worker and calls `db`. `other` calls `db` too.

     0 ms   request A reaches `front`, which calls `db`: `db` is busy until 10 ms
     1 ms   request B reaches `other`, which calls `db`: the call waits at `db`
     2 ms   request C reaches `front` and waits: `front`'s worker is busy with A
    10 ms   `db` answers A's call. A fails, `front` starts C, and C calls `db`.
            B's call has waited since 1 ms, so `db` serves it first.
    20 ms   `db` answers B's call and starts C's
    30 ms   `db` answers C's call
"""

import pytest

from simulator.core.engine import Simulator
from simulator.core.models import TimeInterval
from simulator.faults.injection import PartialFailure
from simulator.runtime.service import ServiceConfig, ServiceRuntime
from simulator.utils.time import ms_to_ns, s_to_ns


def make_service(name, workers, deps=(), optional=(), call_pattern="sequential",
                 median_ms=10, always_fails=False):
    """A service with a fixed service time. `optional` lists the optional dependencies."""
    failures = []
    if always_fails:
        failures.append(PartialFailure(duration=TimeInterval(begin=0, end=s_to_ns(1)), p_fail=1.0))
    cfg = ServiceConfig(
        name=name,
        latency_median=ms_to_ns(median_ms),
        latency_lognorm_sigma=0.0,
        workers=workers,
        partial_failures=failures,
    )
    return ServiceRuntime(
        cfg=cfg,
        dependencies=list(deps),
        dependency_optionality=[dep in optional for dep in deps],
        dependency_call_pattern=call_pattern,
    ).bind(seed=1)


def send(sim, service, name, at_ms, answered_at):
    """Send request `name` to `service` at `at_ms`. Its answer time is stored in `answered_at`."""
    def answered():
        answered_at[name] = sim.timestep

    def arrive():
        service.submit_request(
            sim, on_attempt_done=lambda *attempt: None, on_root_done=answered,
        )

    sim.schedule(ms_to_ns(at_ms), arrive)


def is_ms(timestamp_ns, expected_ms):
    """Whether a timestamp is `expected_ms` (service times carry a few ns of rounding)."""
    return abs(timestamp_ns - ms_to_ns(expected_ms)) < ms_to_ns(0.01)


@pytest.mark.parametrize("call_pattern", ["sequential", "parallel"])
def test_a_waiting_call_is_served_before_a_call_sent_when_the_worker_frees(call_pattern):
    """The example at the top of this file: B is answered at 20 ms and C at 30 ms."""
    sim = Simulator(seed=1)
    db = make_service("db", workers=1, always_fails=True)
    front = make_service("front", workers=1, deps=[db], call_pattern=call_pattern)
    other = make_service("other", workers=10, deps=[db], call_pattern=call_pattern)

    answered_at = {}
    send(sim, front, "A", 0, answered_at)
    send(sim, other, "B", 1, answered_at)
    send(sim, front, "C", 2, answered_at)
    sim.run()

    assert is_ms(answered_at["A"], 10)
    assert is_ms(answered_at["B"], 20)
    assert is_ms(answered_at["C"], 30)


def test_the_next_sequential_call_also_joins_the_queue():
    """`front` calls `cache` (optional) and then `db`; `cache` calls `db` itself.

     0 ms   request A reaches `front`, which calls `cache`, which calls `db`
     1 ms   request B reaches `other`, which calls `db`: the call waits at `db`
    10 ms   `db` answers `cache`'s call. `cache` fails, and `front` moves on to its
            own call to `db`. B's call has waited since 1 ms, so `db` serves it first.
    20 ms   `db` answers B's call and starts `front`'s
    30 ms   `db` answers `front`'s call
    """
    sim = Simulator(seed=1)
    db = make_service("db", workers=1, always_fails=True)
    cache = make_service("cache", workers=10, deps=[db])
    front = make_service("front", workers=10, deps=[cache, db], optional=[cache])
    other = make_service("other", workers=10, deps=[db])

    answered_at = {}
    send(sim, front, "A", 0, answered_at)
    send(sim, other, "B", 1, answered_at)
    sim.run()

    assert is_ms(answered_at["B"], 20)
    assert is_ms(answered_at["A"], 30)


@pytest.mark.parametrize("call_pattern", ["sequential", "parallel"])
def test_a_call_reaches_an_idle_dependency_at_the_instant_it_is_sent(call_pattern):
    """`front` calls `db` (10 ms) and then works for 5 ms: the answer comes at 15 ms."""
    sim = Simulator(seed=1)
    db = make_service("db", workers=1)
    front = make_service("front", workers=1, deps=[db], call_pattern=call_pattern, median_ms=5)

    answered_at = {}
    send(sim, front, "A", 0, answered_at)
    sim.run()

    assert is_ms(answered_at["A"], 15)
