from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional

from simulator.core.engine import Simulator
from simulator.core.types import DropReason, TimePoint

from .service_attempts import CallResult


class _ServiceDependencyMixin:
    """How a service calls its dependencies.

    Each dependency call counts once, when the dependency is done with it, even
    if the dependency retries on its own (see `submit_call`). Example: B fails
    at 10 ms, retries, and succeeds at 20 ms; the service calling B sees one
    success, at 20 ms.
    """

    # Severity ranking: higher value = more severe
    _DROP_SEVERITY = {
        DropReason.NONE: 0,
        DropReason.QUEUE_FULL: 1,
        DropReason.SERVER_FAILURE: 2,
        DropReason.DEADLINE: 3,
    }

    def _call_deps_parallel(
        self,
        sim: Simulator,
        deps: List["ServiceRuntime"],
        optionality: List[bool],
        on_all_done: Callable[[bool, DropReason], None],
        deadline: Optional[TimePoint],
        is_retry: bool,
        retry_budget_remaining: Optional[list] = None,
        tenant_id: Optional[str] = None,
    ):
        """Call all dependencies concurrently, and report once all of them are done."""
        remaining = len(deps)
        any_required_failed = False
        worst_reason = DropReason.NONE

        def on_dep_done(dep_idx: int, result: CallResult):
            nonlocal remaining, any_required_failed, worst_reason
            remaining -= 1
            is_optional = optionality[dep_idx] if dep_idx < len(optionality) else False
            if not result.success and not is_optional:
                any_required_failed = True
                if self._DROP_SEVERITY.get(result.drop_reason, 0) > self._DROP_SEVERITY.get(worst_reason, 0):
                    worst_reason = result.drop_reason
            if remaining == 0:
                on_all_done(not any_required_failed, worst_reason)

        for i, dep in enumerate(deps):
            dep.submit_call(
                sim,
                partial(on_dep_done, i),
                global_deadline=deadline,
                is_retry=is_retry,
                retry_budget_remaining=retry_budget_remaining,
                tenant_id=tenant_id,
            )

    def _call_deps_sequential(
        self,
        sim: Simulator,
        deps: List["ServiceRuntime"],
        optionality: List[bool],
        idx: int,
        on_all_done: Callable[[bool, DropReason], None],
        deadline: Optional[TimePoint],
        is_retry: bool,
        retry_budget_remaining: Optional[list] = None,
        tenant_id: Optional[str] = None,
    ):
        """Call dependencies one after another. Stop at the first required one that fails."""
        if idx >= len(deps):
            on_all_done(True, DropReason.NONE)
            return

        def on_dep_done(result: CallResult):
            is_optional = optionality[idx] if idx < len(optionality) else False
            if not result.success and not is_optional:
                on_all_done(False, result.drop_reason)
                return
            self._call_deps_sequential(
                sim, deps, optionality, idx + 1, on_all_done, deadline, is_retry,
                retry_budget_remaining=retry_budget_remaining,
                tenant_id=tenant_id,
            )

        deps[idx].submit_call(
            sim,
            on_dep_done,
            global_deadline=deadline,
            is_retry=is_retry,
            retry_budget_remaining=retry_budget_remaining,
            tenant_id=tenant_id,
        )
