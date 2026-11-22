from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, NamedTuple, Optional

from simulator.core.types import TimePoint, TimeDuration, DropReason
from simulator.core.models import Request, RootRequest, TimeInterval
from simulator.core.engine import Simulator
from simulator.runtime.service import ServiceRuntime
from simulator.metrics.collector import Metrics
from simulator.policies.retry import RetryPolicy, RetryContext
from simulator.policies.timeout import Timeout


@dataclass(frozen=True)
class ClientConfig:
    name: str = "client"
    retry: Optional['RetryPolicy'] = None
    timeout: Optional['Timeout'] = None
    e2e_retry_budget: Optional[int] = None  # Arolla Level 2: max retries across all hops


@dataclass
class AttemptCtx:
    """One request of the client, including all its retry attempts."""
    root: RootRequest
    _retry_budget_remaining: Optional[list] = None  # Arolla Level 2: shared [B]


class _Call(NamedTuple):
    """Models the client call to the service (one submit_request)."""
    on_attempt_done: Callable[..., None]  # the service reports one of their attempts
    on_root_done: Callable[[], None]      # the service has finished the call
    answered: Callable[[], bool]          # has the call already given the client a result?
    walk_away: Callable[[], None]         # the client's timeout fired: stop following this call


@dataclass
class ClientRuntime:
    """
    Client runtime that generates requests and tracks results.

    The client sends requests to one service. With a client-side retry policy, it
    decides on retries itself. Without one, it relies on the service and
    records the attempts that the service may make.

    Attributes:
        cfg: Client configuration
        service: The service to send requests to
        roots: List of all completed root requests (for metrics)
        attempts_total: Total number of attempts made across all requests

    Methods:
        start_request: Initiate a new request to the service
        metrics: Get metrics for all completed requests
    """
    cfg: ClientConfig
    service: ServiceRuntime

    roots: List[RootRequest] = field(default_factory=list)
    attempts_total: int = 0

    def start_request(self, sim: Simulator):
        root_req = RootRequest()

        global_timeout = None
        
        # Priority: Client config > Service config
        if self.cfg.timeout is not None:
            global_timeout = self.cfg.timeout.get_global_timeout()
        elif self.service.cfg.timeout is not None:
            global_timeout = self.service.cfg.timeout.get_global_timeout()
        if global_timeout is not None:
            root_req.global_deadline = sim.timestep + global_timeout

        ctx = AttemptCtx(
            root=root_req,
            _retry_budget_remaining=(
                [self.cfg.e2e_retry_budget] if self.cfg.e2e_retry_budget is not None else None
            ),
        )
        self._make_attempt(sim, ctx, is_retry=False)

    def _make_attempt(self, sim: Simulator, ctx: AttemptCtx, is_retry: bool):
        """Send one call to the service. With an attempt timeout, the client gives
        up on the call if the service has not answered in time."""
        begin_time = sim.timestep
        if self.cfg.retry is None:
            call = self._follow_call_without_policy(ctx, sim)
        else:
            call = self._follow_call_with_policy(ctx, sim, begin_time)

        if self.cfg.timeout is not None:
            attempt_timeout = self.cfg.timeout.get_attempt_timeout()
            if attempt_timeout:
                sim.schedule(
                    sim.timestep + attempt_timeout,
                    partial(self._on_local_timeout, ctx, sim, call, begin_time),
                )

        self.service.submit_request(
            sim,
            on_attempt_done=call.on_attempt_done,
            on_root_done=call.on_root_done,
            global_deadline=ctx.root.global_deadline,
            is_retry=is_retry,
            retry_budget_remaining=ctx._retry_budget_remaining,
            tenant_id=self.cfg.name,
        )

    def _follow_call_without_policy(self, ctx: AttemptCtx, sim: Simulator) -> _Call:
        """The client has no retry policy, but the service may have a retry policy. The
        client records the attempts that the service reports.

        If the client timeout elapses before the service has finished, the request
        ends as a failure. Whatever the service reports afterwards is ignored.
        """
        return _Call(
            on_attempt_done=partial(self._record_attempt, ctx, sim),
            on_root_done=partial(self._on_root_done, ctx),
            answered=lambda: ctx.root.done,
            walk_away=partial(self._on_root_done, ctx),
        )

    def _follow_call_with_policy(self, ctx: AttemptCtx, sim: Simulator, begin_time: TimePoint) -> _Call:
        """The client has a retry policy.

        The service may have a server-side retry policy and will make the call on its own and reports each attempt.
        Only the results of the last service call are considered (on_root_done).

        If the client timeout elapses before the service has finished, the request
        ends as a failure. Whatever the service reports afterwards is ignored.
        """
        last_attempt = None
        over = False

        def on_attempt_done(success, svc_time, drop_reason, queue_size, _attempt_begin, attempt_deadline):
            nonlocal last_attempt
            last_attempt = (success, svc_time, drop_reason, queue_size, attempt_deadline)

        def on_root_done():
            nonlocal over
            if over:
                return  # the client already timed this call out
            over = True
            self._on_call_done(ctx, sim, begin_time, last_attempt)

        def answered():
            return over

        def walk_away():
            nonlocal over
            over = True

        return _Call(on_attempt_done, on_root_done, answered, walk_away)

    def _on_call_done(
        self, ctx: AttemptCtx, sim: Simulator, call_begin_time: TimePoint, last_attempt: Optional[tuple],
    ):
        """The service has finished a call: record its result as one client attempt."""
        if last_attempt is None:
            # The service made no attempt: the request's deadline had already passed.
            success, svc_time, drop_reason = False, 0, DropReason.DEADLINE
            queue_size, attempt_deadline = len(self.service.queue), None
        else:
            success, svc_time, drop_reason, queue_size, attempt_deadline = last_attempt

        self._record_attempt(
            ctx, sim,
            success=success,
            svc_time=svc_time,
            drop_reason=drop_reason,
            queue_size=queue_size,
            begin_time=call_begin_time,
            attempt_deadline=attempt_deadline,
        )

    def _on_local_timeout(self, ctx: AttemptCtx, sim: Simulator, call: _Call, begin_time: TimePoint):
        """The client attempt timeout has elapsed. If the call is still running at the
        service, the client gives up and records a timeout."""
        if ctx.root.done or call.answered():
            return
        self._record_attempt(
            ctx, sim,
            success=False,
            svc_time=0,
            drop_reason=DropReason.DEADLINE,
            queue_size=-1,  # the client cannot see the service's queue
            begin_time=begin_time,
            attempt_deadline=sim.timestep,
        )
        call.walk_away()

    def _record_attempt(
        self,
        ctx: AttemptCtx,
        sim: Simulator,
        success: bool,
        svc_time: TimeDuration,
        drop_reason: DropReason,
        queue_size: int,
        begin_time: TimePoint,
        attempt_deadline: Optional[TimePoint],
    ):
        """Record one attempt of the request. A client with a retry policy then
        may decide whether the request is over or to try another attempt."""
        if ctx.root.done:
            return  # in case we receive a late answer for a request that has already finished

        self.attempts_total += 1
        req = Request(
            service=self.service.cfg.name,
            interval=TimeInterval(begin=begin_time, end=sim.timestep),
            deadline=attempt_deadline,
        )
        req.success = success
        req.drop_reason = drop_reason
        req.queue_size_at_end = queue_size
        ctx.root.add_attempt(req)

        # Report the latency to the service timeout policy (for CSV parity).
        if self.service.cfg.timeout is not None:
            self.service.cfg.timeout.record_result(success, req.interval.end - req.interval.begin)

        if self.cfg.retry is None:
            return

        retry_ctx = RetryContext(attempt=len(ctx.root.attempts), now=sim.timestep)
        self.cfg.retry.record_attempt(retry_ctx, success)
        if success:
            self._on_root_done(ctx)
            return

        should_retry, delay = self.cfg.retry.next_delay(retry_ctx)
        next_start = sim.timestep + delay
        too_late = ctx.root.global_deadline is not None and next_start >= ctx.root.global_deadline
        if not should_retry or too_late:
            self._on_root_done(ctx)
            return
        sim.schedule(next_start, partial(self._make_attempt, sim, ctx, is_retry=True))

    def _on_root_done(self, ctx: AttemptCtx):
        if not ctx.root.done:
            ctx.root.done = True
            self.roots.append(ctx.root)

    def metrics(self) -> Metrics:
        return Metrics(roots=self.roots, attempts_total=self.attempts_total)
