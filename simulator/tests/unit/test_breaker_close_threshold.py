"""
The server-side time-based breaker closes when its success rate reaches the threshold.

The close test was written on the failure side: `failure_rate < 1 - success_threshold_rate`.
At a threshold of 1.0 that is `failure_rate < 0`, which nothing satisfies, so a
breaker told to close only when every probe succeeds never closed again. And
since `1 - 0.8` is not `0.2` in binary floating point, 4 successes out of 5 at
0.8 were refused as well.
Without a success threshold, it closes once the failure rate is back below the
rate that caused it to open.
"""

import pytest

from simulator.policies.retry_controls import CBState, LimiterTimeBasedCircuitBreakerPolicy
from simulator.utils.time import ms_to_ns, s_to_ns


def opened(success_threshold, min_requests=5):
    """A breaker tripped at t = 0 by ten failures."""
    breaker = LimiterTimeBasedCircuitBreakerPolicy(
        failure_threshold_rate=0.5,
        success_threshold_rate=success_threshold,
        min_requests=min_requests,
        window_duration=s_to_ns(1),
        half_open_delay=ms_to_ns(500),
    )
    for i in range(10):
        breaker.add_result(False, i * ms_to_ns(1))
    assert breaker.get_state() == CBState.OPEN
    return breaker


def after(breaker, results):
    """Feed results long after the open delay, when the failures have left the window."""
    start = s_to_ns(5)
    for i, ok in enumerate(results):
        breaker.add_result(ok, start + i * ms_to_ns(1))
    return breaker.get_state()


def test_closes_when_every_probe_must_succeed_and_does():
    assert after(opened(1.0), [True] * 5) == CBState.CLOSED     # it stayed OPEN


def test_one_failure_keeps_it_open_when_every_probe_must_succeed():
    assert after(opened(1.0), [True, True, False, True, True]) == CBState.OPEN


def test_closes_at_exactly_the_threshold():
    assert after(opened(0.8), [True, True, True, True, False]) == CBState.CLOSED   # it stayed OPEN


def test_stays_open_one_success_short_of_the_threshold():
    assert after(opened(0.8), [True, True, True, False, False]) == CBState.OPEN


@pytest.mark.parametrize("percent", range(1, 100))
def test_with_no_success_threshold_it_closes_below_the_failure_threshold(percent):
    """With no `success_threshold`, the breaker closes once the failure rate drops
    below `failure_threshold`.

    Example with a threshold of 10 %: the open breaker receives 100 results, 10 of
    them failures. That is exactly 10 %, enough to open a breaker, so it remains open.
    With one more success, it becomes 10 out of 101 (9.9 %), and it breaker closes.
    """
    failure_threshold = percent / 100
    breaker = LimiterTimeBasedCircuitBreakerPolicy(
        failure_threshold_rate=failure_threshold,
        success_threshold_rate=None,
        min_requests=20,
        window_duration=s_to_ns(1),
        half_open_delay=ms_to_ns(500),
    )
    for i in range(20):
        breaker.add_result(False, i * ms_to_ns(1))
    assert breaker.get_state() == CBState.OPEN

    results = [False] * percent + [True] * (100 - percent)
    assert after(breaker, results) == CBState.OPEN
    breaker.add_result(True, s_to_ns(5) + ms_to_ns(100))
    assert breaker.get_state() == CBState.CLOSED


def test_too_few_results_do_not_close_it():
    assert after(opened(0.8, min_requests=5), [True] * 4) == CBState.OPEN


def test_it_does_not_close_before_the_open_delay():
    breaker = opened(0.8)
    # 100 ms after it opened: the ten failures are still in the window anyway.
    for i in range(5):
        breaker.add_result(True, ms_to_ns(100) + i * ms_to_ns(1))
    assert breaker.get_state() == CBState.OPEN
