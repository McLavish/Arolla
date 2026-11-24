"""
An adaptive timeout follows the latency of the recent successful attempts.

Example used below, with the default settings. The last successful attempts all
took 40 ms, so their 95th percentile is 40 ms and the next attempt gets
2 * 40 = 80 ms. The timeout starts at `attempt_ms` and keeps that value until 10
attempts have succeeded.
"""

import pytest

from simulator.config.loader import ConfigLoader
from simulator.policies.timeout import AdaptiveTimeout, StaticTimeout
from simulator.utils.time import ms_to_ns


def timeout_ms(timeout):
    return timeout.get_attempt_timeout() / ms_to_ns(1)


def record(timeout, successes=0, failures=0, latency_ms=40):
    for _ in range(successes):
        timeout.record_result(True, ms_to_ns(latency_ms))
    for _ in range(failures):
        timeout.record_result(False, 0)


def test_the_timeout_follows_the_latency_of_successful_attempts():
    """The example at the top of this file."""
    timeout = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(100))

    record(timeout, successes=9)
    assert timeout_ms(timeout) == 100  # still the starting value
    record(timeout, successes=1)
    assert timeout_ms(timeout) == 80


def test_the_global_timeout_does_not_change():
    timeout = AdaptiveTimeout(initial_global_timeout=ms_to_ns(500), initial_attempt_timeout=ms_to_ns(100))
    record(timeout, successes=20)
    assert timeout.get_global_timeout() == ms_to_ns(500)


def test_failed_attempts_do_not_shorten_the_timeout():
    """A failed attempt can be very short, for example when the queue is full. Here the
    timeout follows the median. After 10 successes of 40 ms and 10 failures of 0 ms,
    the median is still 40 ms and the timeout is still 80 ms."""
    timeout = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(100), target_percentile=0.5)
    record(timeout, successes=10, failures=10)
    assert timeout_ms(timeout) == 80


def test_a_low_success_rate_raises_the_timeout():
    """10 of the last 40 attempts succeeded: a success rate of 0.25, which is below 0.5.
    The timeout of 80 ms is divided by 0.25, which gives 320 ms."""
    timeout = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(100))
    record(timeout, successes=10, failures=30)
    assert timeout_ms(timeout) == 320


def test_a_starting_value_that_is_too_short_is_raised():
    """With a starting value of 10 ms every attempt times out, so there is never a
    success to learn from. The timeout is raised to 10 times its starting value,
    and no further."""
    timeout = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(10))
    record(timeout, failures=1)
    assert timeout_ms(timeout) == 100
    record(timeout, failures=200)
    assert timeout_ms(timeout) == 100


def test_the_timeout_stays_between_its_bounds():
    low = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(100), min_timeout=ms_to_ns(90))
    record(low, successes=10)  # 80 ms is below the lowest value allowed
    assert timeout_ms(low) == 90

    high = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(100), max_timeout=ms_to_ns(200))
    record(high, successes=10, failures=30)  # 320 ms is above the highest value allowed
    assert timeout_ms(high) == 200

    short = AdaptiveTimeout(initial_attempt_timeout=ms_to_ns(100), max_timeout=ms_to_ns(200))
    record(short, failures=5)  # 10 * 100 ms is above the highest value allowed
    assert timeout_ms(short) == 200


SCENARIO = """
name: adaptive_timeout
seed: 3
services:
{front}
  - name: svc
    latency: {{median_ms: 20, lognorm_sigma: 0.0}}
    workers: 8
    {service_extra}
clients:
  - name: client
    workload: {{base_rps: 20, duration_s: 2}}
    {client_extra}
"""


# A service that takes 1 ms for its own work and calls `svc`.
FRONT = """
  - name: front
    latency: {median_ms: 1, lognorm_sigma: 0.0}
    workers: 8
    dependencies: [{service: svc}]
"""


def run(service_extra="", client_extra="", front=""):
    """Run the scenario. The client calls `svc`, or `front` if it is given."""
    cfg = ConfigLoader.load_from_string(
        SCENARIO.format(service_extra=service_extra, client_extra=client_extra, front=front)
    )
    sim, clients, workloads, _, services = ConfigLoader.build_simulation(cfg)
    for client, workload in zip(clients, workloads):
        workload.drive(sim, lambda s, c=client: c.start_request(s))
    sim.run()
    return clients[0], services["svc"]


def test_the_settings_in_the_scenario_reach_the_timeout():
    client, _ = run(client_extra=(
        "timeout: {type: adaptive, global_ms: 900, attempt_ms: 100, target_percentile: 0.9,"
        " safety_margin: 3, window_size: 60, min_samples: 5, min_ms: 2, max_ms: 400,"
        " success_rate_window: 30, min_success_rate: 0.4}"
    ))
    timeout = client.cfg.timeout

    assert isinstance(timeout, AdaptiveTimeout)
    assert timeout.initial_global_timeout == ms_to_ns(900)
    assert timeout.initial_attempt_timeout == ms_to_ns(100)
    assert (timeout.target_percentile, timeout.safety_margin) == (0.9, 3)
    assert (timeout.window_size, timeout.min_samples) == (60, 5)
    assert (timeout.min_timeout, timeout.max_timeout) == (ms_to_ns(2), ms_to_ns(400))
    assert (timeout.success_rate_window, timeout.min_success_rate) == (30, 0.4)


def test_a_timeout_without_a_type_stays_fixed():
    client, _ = run(client_extra="timeout: {attempt_ms: 100}")
    assert isinstance(client.cfg.timeout, StaticTimeout)
    assert timeout_ms(client.cfg.timeout) == 100


def test_the_timeout_of_a_client_follows_the_calls_of_the_client():
    """Every call takes 20 ms, so the timeout of the client goes from 100 ms to 40 ms."""
    client, _ = run(client_extra="timeout: {type: adaptive, attempt_ms: 100}")

    assert len(client.roots) > 20
    assert all(root.attempts[-1].success for root in client.roots)
    assert timeout_ms(client.cfg.timeout) == pytest.approx(40, abs=0.01)


def test_the_timeout_of_a_service_follows_the_attempts_of_the_service():
    """The client calls `front`, which calls `svc`. Every attempt of `svc` takes 20 ms,
    so the timeout of `svc` goes from 100 ms to 40 ms. Each attempt is counted once."""
    client, service = run(
        service_extra="timeout: {type: adaptive, attempt_ms: 100, success_rate_window: 1000}",
        front=FRONT,
    )
    timeout = service.cfg.timeout

    assert timeout_ms(timeout) == pytest.approx(40, abs=0.01)
    assert len(timeout._successes) == len(client.roots)


def test_a_shorter_timeout_is_used_for_the_next_attempts():
    """The timeout of the service goes down to 40 ms. Then the service becomes slow
    (200 ms per attempt), so its attempts are cut off at about 40 ms, not at the
    starting value of 500 ms."""
    client, _ = run(service_extra=(
        "timeout: {type: adaptive, attempt_ms: 500}\n"
        "    latency_injections:\n"
        "      - {type: latency_injection, start_s: 1, end_s: 2, multiplier: 10}"
    ))

    timed_out = [a for root in client.roots for a in root.attempts if not a.success]
    assert len(timed_out) > 10
    for attempt in timed_out[:5]:
        assert (attempt.interval.end - attempt.interval.begin) / ms_to_ns(1) == pytest.approx(40, abs=0.01)


@pytest.mark.parametrize("settings, message", [
    ("attempt_ms: 100, max_ms: 50", "min_ms <= attempt_ms <= max_ms"),
    ("attempt_ms: 100, min_ms: 200", "min_ms <= attempt_ms <= max_ms"),
    ("min_ms: 200, max_ms: 100", "min_ms <= attempt_ms <= max_ms"),
    ("attempt_ms: 100, min_samples: 20, window_size: 10", "min_samples <= window_size"),
])
def test_settings_that_cannot_work_are_rejected(settings, message):
    scenario = SCENARIO.format(
        service_extra="", front="", client_extra=f"timeout: {{type: adaptive, {settings}}}"
    )
    with pytest.raises(ValueError, match=message):
        ConfigLoader.load_from_string(scenario)
