"""
A failed optional call that answers after the caller's deadline ends the attempt as a timeout.

A call that waits in a queue is only reported when a worker reaches it, which can be
after the deadline. The caller ignores the failure of an optional call, but its own
deadline has passed, so its attempt has timed out.

Example used below. `slow` has one worker and takes 100 ms per call. `front` takes 1 ms
for its own work, gives up on an attempt after 50 ms, and calls `slow` (optional).

     0 ms   a direct call reaches `slow`, which is busy with it until 100 ms
    10 ms   a request reaches `front`, which calls `slow`: the call waits in the queue.
            `front`'s deadline is 60 ms.
   100 ms   `slow` reaches the call, sees that its deadline has passed and reports a
            timeout. The call was optional, but `front`'s deadline has passed too, so
            `front` reports a timeout and frees its worker.
"""

import pytest

from simulator.core.engine import Simulator
from simulator.core.types import DropReason
from simulator.policies.timeout import StaticTimeout
from simulator.runtime.service import ServiceConfig, ServiceRuntime
from simulator.utils.time import ms_to_ns


def make_service(name, median_ms, workers, deps=(), call_pattern="sequential", timeout_ms=None):
    """A service with a fixed service time whose dependencies are all optional."""
    cfg = ServiceConfig(
        name=name,
        latency_median=ms_to_ns(median_ms),
        latency_lognorm_sigma=0.0,
        workers=workers,
        timeout=StaticTimeout(attempt_timeout=ms_to_ns(timeout_ms)) if timeout_ms else None,
    )
    return ServiceRuntime(
        cfg=cfg,
        dependencies=list(deps),
        dependency_optionality=[True] * len(deps),
        dependency_call_pattern=call_pattern,
    ).bind(seed=1)


def is_ms(timestamp_ns, expected_ms):
    """Whether a timestamp is `expected_ms` (service times carry a few ns of rounding)."""
    return abs(timestamp_ns - ms_to_ns(expected_ms)) < ms_to_ns(0.01)


@pytest.mark.parametrize("call_pattern", ["sequential", "parallel"])
def test_the_attempt_times_out_when_the_late_answer_arrives(call_pattern):
    """The example at the top of this file: `front` reports a timeout at 100 ms."""
    sim = Simulator(seed=1)
    slow = make_service("slow", median_ms=100, workers=1)
    front = make_service("front", median_ms=1, workers=10, deps=[slow],
                         call_pattern=call_pattern, timeout_ms=50)

    attempts = []

    def on_attempt_done(success, svc_time, drop_reason, queue_size, begin, deadline):
        attempts.append((sim.timestep, success, drop_reason))

    def request():
        front.submit_request(sim, on_attempt_done=on_attempt_done, on_root_done=lambda: None)

    sim.schedule(0, lambda: slow.submit_call(sim, lambda result: None))
    sim.schedule(ms_to_ns(10), request)
    sim.run()

    assert len(attempts) == 1
    answered_at, success, reason = attempts[0]
    assert is_ms(answered_at, 100)
    assert not success
    assert reason == DropReason.DEADLINE
    assert front.in_flight == 0
