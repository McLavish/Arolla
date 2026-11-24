"""
The bursty (token bucket) limiter refills however often it is asked.

A refill was truncated to whole tokens on every call while the refill clock
still moved forward. A limiter asked more often than once per
`period / refill_rate` therefore added 0 tokens each time and never refilled:
after its initial burst it denied everything for the rest of the run.
"""

import pytest

from simulator.policies.load_limiter import BurstyRateLimiterPolicy
from simulator.policies.retry import RetryContext
from simulator.utils.time import ms_to_ns, s_to_ns


def drained(max_requests=5, refill_rate=10):
    """A bucket emptied at t = 0."""
    limiter = BurstyRateLimiterPolicy(
        max_requests=max_requests, refill_rate=refill_rate, period=s_to_ns(1),
    )
    for _ in range(max_requests):
        assert limiter.next_delay(RetryContext(attempt=2, now=0))[0]
    assert not limiter.next_delay(RetryContext(attempt=2, now=0))[0]
    return limiter


def granted(limiter, every_ms, for_s):
    """Grants when asked once every `every_ms` for `for_s` seconds."""
    step = ms_to_ns(every_ms)
    return sum(
        limiter.next_delay(RetryContext(attempt=2, now=t))[0]
        for t in range(step, s_to_ns(for_s) + 1, step)
    )


def test_it_refills_when_asked_more_often_than_a_token_takes():
    # 10 tokens/s, asked every 50 ms for 10 s: one grant every second ask.
    assert granted(drained(), every_ms=50, for_s=10) == 100   # it was 0


@pytest.mark.parametrize("every_ms", [1, 10, 25, 100])
def test_the_refill_rate_does_not_depend_on_how_often_it_is_asked(every_ms):
    # 100 tokens accrue in 10 s; allow one for floating-point accumulation.
    assert 99 <= granted(drained(), every_ms=every_ms, for_s=10) <= 100


def test_the_bucket_caps_what_accrues_while_idle():
    limiter = drained()
    # 10 idle seconds are 100 tokens; the bucket holds 5.
    now = s_to_ns(10)
    assert sum(limiter.next_delay(RetryContext(attempt=2, now=now))[0] for _ in range(20)) == 5


def test_less_than_a_token_is_not_a_grant():
    limiter = drained()
    assert not limiter.next_delay(RetryContext(attempt=2, now=ms_to_ns(90)))[0]    # 0.9 token
    assert limiter.next_delay(RetryContext(attempt=2, now=ms_to_ns(110)))[0]       # 1.1
    assert not limiter.next_delay(RetryContext(attempt=2, now=ms_to_ns(111)))[0]   # 0.11 left
