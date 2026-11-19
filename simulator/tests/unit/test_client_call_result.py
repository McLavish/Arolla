"""
A client with its own retry policy sees one result per call.

A call is one `submit_request`. When the service behind it retries on its own,
it reports several attempts for that one call. The client's policy must decide
on the CALL: its result is the outcome of the service's last attempt, and it
exists only when the service has finished.

Deciding on the first attempt (the old behaviour) made `retry: {type: none}`
on a client hide every retry of the service behind it, and made a retrying
client start a second call while the service was still retrying the first.
"""

from simulator.core.engine import Simulator
from simulator.core.models import TimeInterval
from simulator.core.types import DropReason
from simulator.faults.injection import PartialFailure
from simulator.policies.retry import FixedBackoffRetryPolicy, NoRetryPolicy
from simulator.policies.timeout import StaticTimeout
from simulator.runtime.client import ClientConfig, ClientRuntime
from simulator.runtime.service import ServiceConfig, ServiceRuntime
from simulator.utils.time import ms_to_ns


def make_service(fail_until_ms=None, max_attempts=None):
    """10 ms per attempt; attempts ending before `fail_until_ms` fail."""
    failures = []
    if fail_until_ms is not None:
        failures.append(PartialFailure(
            duration=TimeInterval(begin=0, end=ms_to_ns(fail_until_ms)), p_fail=1.0,
        ))
    retry = None
    if max_attempts is not None:
        retry = FixedBackoffRetryPolicy(max_attempts=max_attempts, delay=0)
    cfg = ServiceConfig(
        name="svc",
        latency_median=ms_to_ns(10),
        latency_lognorm_sigma=0.0,
        workers=10,
        partial_failures=failures,
        retry=retry,
    )
    return ServiceRuntime(cfg=cfg).bind(seed=1, record_events=True)


def run_one(client):
    sim = Simulator(seed=1)
    client.start_request(sim)
    sim.run()
    return sim


def at_ms(timestamp_ns, expected_ms):
    return abs(timestamp_ns - ms_to_ns(expected_ms)) < ms_to_ns(0.01)


class TestClientWithItsOwnPolicy:
    def test_no_retry_client_sees_the_services_final_result(self):
        """The service fails twice and succeeds on its third attempt: one successful call."""
        svc = make_service(fail_until_ms=25, max_attempts=3)
        client = ClientRuntime(cfg=ClientConfig(retry=NoRetryPolicy()), service=svc)
        run_one(client)

        assert len(svc.events) == 3
        (root,) = client.roots
        assert [(a.success, a.drop_reason) for a in root.attempts] == [(True, DropReason.NONE)]
        assert client.attempts_total == 1
        # The call lasts from the submission to the end of the third attempt.
        assert root.attempts[0].interval.begin == 0
        assert at_ms(root.attempts[0].interval.end, 30)

    def test_no_retry_client_sees_the_failure_when_the_service_gives_up(self):
        svc = make_service(fail_until_ms=10_000, max_attempts=3)
        client = ClientRuntime(cfg=ClientConfig(retry=NoRetryPolicy()), service=svc)
        run_one(client)

        assert len(svc.events) == 3
        (root,) = client.roots
        assert [(a.success, a.drop_reason) for a in root.attempts] == [
            (False, DropReason.SERVER_FAILURE),
        ]
        assert at_ms(root.attempts[0].interval.end, 30)

    def test_retrying_client_does_not_start_a_call_while_the_service_retries(self):
        """
        Client: 2 attempts, 5 ms apart. Service: 3 attempts, the first two fail.
        The first call succeeds at 30 ms; there is no second call.
        """
        svc = make_service(fail_until_ms=22, max_attempts=3)
        client = ClientRuntime(
            cfg=ClientConfig(retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5))),
            service=svc,
        )
        run_one(client)

        assert len(svc.events) == 3
        (root,) = client.roots
        assert [(a.success, a.drop_reason) for a in root.attempts] == [(True, DropReason.NONE)]
        assert at_ms(root.attempts[0].interval.end, 30)

    def test_each_client_attempt_is_one_complete_call(self):
        """Both retry: 2 client calls of 2 service attempts each."""
        svc = make_service(fail_until_ms=10_000, max_attempts=2)
        client = ClientRuntime(
            cfg=ClientConfig(retry=FixedBackoffRetryPolicy(max_attempts=2, delay=ms_to_ns(5))),
            service=svc,
        )
        run_one(client)

        assert len(svc.events) == 4
        (root,) = client.roots
        assert [a.drop_reason for a in root.attempts] == [DropReason.SERVER_FAILURE] * 2
        assert client.attempts_total == 2
        # First call 0-20 ms, 5 ms of backoff, second call 25-45 ms.
        assert at_ms(root.attempts[0].interval.end, 20)
        assert at_ms(root.attempts[1].interval.begin, 25)
        assert at_ms(root.attempts[1].interval.end, 45)

    def test_a_call_the_service_never_attempted_is_a_deadline(self):
        """Deadline already over at submission: the service makes no attempt and the
        client must still get a result (it used to wait forever)."""
        svc = make_service()
        client = ClientRuntime(
            cfg=ClientConfig(retry=NoRetryPolicy(), timeout=StaticTimeout(global_timeout=0)),
            service=svc,
        )
        run_one(client)

        assert len(svc.events) == 0
        (root,) = client.roots
        assert [(a.success, a.drop_reason) for a in root.attempts] == [(False, DropReason.DEADLINE)]


class TestClientWithoutAPolicy:
    def test_it_still_records_every_attempt_the_service_makes(self):
        """No client policy ("legacy" mode): the service's retry loop is the client's."""
        svc = make_service(fail_until_ms=25, max_attempts=3)
        client = ClientRuntime(cfg=ClientConfig(), service=svc)
        run_one(client)

        (root,) = client.roots
        assert [a.drop_reason for a in root.attempts] == [
            DropReason.SERVER_FAILURE, DropReason.SERVER_FAILURE, DropReason.NONE,
        ]
        assert client.attempts_total == 3
