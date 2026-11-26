"""
Test client-side retry rate limiting.

A client sends 50 requests per second for 4 seconds, with up to three
retries per failed request. If every call fails, the client would send
approximately 600 retries without rate limiting.

A token bucket with an initial capacity of 5 tokens and a refill rate
of 2 tokens per second allows between 5 and 13 retries.
"""

import random

import pytest

from simulator.config.builders.policies import build_client_retry_policy
from simulator.config.loader import ConfigLoader
from simulator.config.schema import ClientConfigYAML
from simulator.policies.load_limiter import BurstyRateLimiterPolicy, LeakyRateLimiterPolicy
from simulator.policies.retry import FixedBackoffRetryPolicy, RetryContext
from simulator.policies.retry_controls import RateLimitedRetryPolicy, RetryBudgetPolicy
from simulator.utils.time import ms_to_ns, s_to_ns

SCENARIO = """
name: client_rate_limiter
seed: 5
services:
  - name: svc
    latency: {{median_ms: 5, lognorm_sigma: 0.2}}
    workers: 8
    partial_failures:
      - {{type: partial_failure, start_s: 0, end_s: 4, p_fail: 1.0}}
clients:
  - name: client
    workload: {{base_rps: 50, duration_s: 4}}
    retry: {{type: fixed, max_attempts: 4, delay_ms: 0}}
    {extra}
"""


def run(extra):
    cfg = ConfigLoader.load_from_string(SCENARIO.format(extra=extra))
    sim, clients, workloads, _, _ = ConfigLoader.build_simulation(cfg)
    for client, workload in zip(clients, workloads):
        workload.drive(sim, lambda s, c=client: c.start_request(s))
    sim.run()
    return clients


def retry_times(client):
    """The start times of all retries of a client, in order."""
    return sorted(a.interval.begin for root in client.roots for a in root.attempts[1:])


@pytest.mark.parametrize("limiter, retries_allowed", [
    # 5 tokens to start with, 2 more each second for 4 seconds.
    ("{type: bursty, max_requests: 5, period_ms: 1000, refill_rate: 2}", (5, 13)),
    # 6 retries in each of the four one-second windows.
    ("{type: fixed_window, max_requests: 6, period_ms: 1000}", (6, 24)),
])
def test_the_limiter_holds_back_the_retries_of_the_client(limiter, retries_allowed):
    """Test that the limiter holds back the retries of the client.

    The example at the top of this file.
    """
    client = run(f"rate_limiter: {limiter}")[0]

    requests = len(client.roots)
    assert requests > 150
    retries = client.attempts_total - requests
    low, high = retries_allowed
    assert low <= retries <= high


def test_without_a_limiter_every_failed_call_is_retried():
    """Test that without a limiter every failed call is retried up to `max_attempts`"""
    client = run("")[0]
    assert client.attempts_total == 4 * len(client.roots)


def test_a_leaky_limiter_spaces_the_retries():
    """Test that a leaky limiter spaces the retries.

    10 retries per second: two retries are never less than 100 ms apart.
    """
    client = run("rate_limiter: {type: leaky, max_requests: 10, period_ms: 1000, max_attempts: 4}")[0]

    times = retry_times(client)
    assert len(times) > 30
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    assert min(gaps) >= ms_to_ns(100)


def test_a_retry_waits_for_the_longer_of_the_two_delays():
    """Test that a retry waits for the longer of the policy's and the limiter's delay.

    The retry policy asks for 10 ms. The leaky limiter has a free slot for the first
    retry, so it waits 10 ms. The next slot is 100 ms later, so the second waits 100 ms.
    """
    policy = RateLimitedRetryPolicy(
        inner=FixedBackoffRetryPolicy(max_attempts=5, delay=ms_to_ns(10)),
        limiter=LeakyRateLimiterPolicy(max_requests=10, max_attempts=5, period=s_to_ns(1)),
    )
    assert policy.next_delay(RetryContext(attempt=1, now=0)) == (True, ms_to_ns(10))
    assert policy.next_delay(RetryContext(attempt=1, now=0)) == (True, ms_to_ns(100))


def test_a_retry_that_the_policy_refuses_does_not_use_the_limiter():
    """Test that a retry the policy refuses does not use the limiter.

    The policy allows 2 attempts. The retry after the second attempt is refused by
    the policy, so it takes no token from the limiter.
    """
    limiter = BurstyRateLimiterPolicy(max_requests=5, refill_rate=1, period=s_to_ns(1))
    policy = RateLimitedRetryPolicy(
        inner=FixedBackoffRetryPolicy(max_attempts=2, delay=0), limiter=limiter,
    )
    assert policy.next_delay(RetryContext(attempt=1, now=0)) == (True, 0)
    assert policy.next_delay(RetryContext(attempt=2, now=0)) == (False, 0)
    assert limiter._tokens == 4


def test_a_retry_that_the_budget_refuses_does_not_use_the_limiter():
    """Test that a retry the budget refuses does not use the limiter.

    The limiter sits inside the retry budget. The budget pays for 1 retry and
    refuses the second, so the limiter gives out only 1 of its 5 tokens.
    """
    cfg = ClientConfigYAML.model_validate({
        "name": "client",
        "workload": {"base_rps": 1, "duration_s": 1},
        "retry": {"type": "fixed", "max_attempts": 5, "delay_ms": 0},
        "rate_limiter": {"type": "bursty", "max_requests": 5, "period_ms": 1000},
        "retry_budget": {"budget_ratio": 0.1, "max_retries": 1, "min_retries_per_sec": 0},
    })
    budget = build_client_retry_policy(cfg, random.Random(1))

    assert isinstance(budget, RetryBudgetPolicy)
    assert isinstance(budget.inner, RateLimitedRetryPolicy)
    assert budget.next_delay(RetryContext(attempt=1, now=0)) == (True, 0)
    assert budget.next_delay(RetryContext(attempt=1, now=0)) == (False, 0)
    assert budget.inner.limiter._tokens == 4


def test_each_replica_of_a_client_has_its_own_limiter():
    """Test that each replica of a client has its own limiter"""
    clients = run("replicas: 2\n    rate_limiter: {type: bursty, max_requests: 5, period_ms: 1000}")

    first, second = (client.cfg.retry.limiter for client in clients)
    assert isinstance(first, BurstyRateLimiterPolicy)
    assert first is not second


def test_a_limiter_on_a_client_without_retries_is_rejected():
    """Test that a rate limiter on a client without retries is rejected"""
    scenario = SCENARIO.format(extra="rate_limiter: {type: bursty, max_requests: 5, period_ms: 1000}")
    scenario = scenario.replace("    retry: {type: fixed, max_attempts: 4, delay_ms: 0}\n", "")
    with pytest.raises(ValueError, match="rate_limiter"):
        ConfigLoader.load_from_string(scenario)
