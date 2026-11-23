from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional

from simulator.core.engine import Simulator
from simulator.core.types import DropReason, TimePoint

from .service_attempts import CallResult


class _ServiceDependencyMixin:

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
        """Call all dependencies concurrently. Report once all of them are done, or,
        with `dependency_fail_fast`, as soon as a required one has failed."""
        remaining = len(deps)
        any_required_failed = False
        worst_reason = DropReason.NONE
        first_reason = DropReason.NONE
        sending = True
        reported = False

        def report():
            nonlocal reported
            reported = True
            if self.dependency_fail_fast and any_required_failed:
                on_all_done(False, first_reason)
            else:
                on_all_done(not any_required_failed, worst_reason)

        def on_dep_done(dep_idx: int, result: CallResult):
            nonlocal remaining, any_required_failed, worst_reason, first_reason
            remaining -= 1
            is_optional = optionality[dep_idx] if dep_idx < len(optionality) else False
            if not result.success and not is_optional:
                if not any_required_failed:
                    first_reason = result.drop_reason
                any_required_failed = True
                if self._DROP_SEVERITY.get(result.drop_reason, 0) > self._DROP_SEVERITY.get(worst_reason, 0):
                    worst_reason = result.drop_reason
            if reported:
                return  # the attempt has already failed, nobody waits for this call
            fail_now = self.dependency_fail_fast and any_required_failed and not sending
            if remaining == 0 or fail_now:
                report()

        for i, dep in enumerate(deps):
            dep.submit_call(
                sim,
                partial(on_dep_done, i),
                global_deadline=deadline,
                is_retry=is_retry,
                retry_budget_remaining=retry_budget_remaining,
                tenant_id=tenant_id,
            )

        # A call that fails while the others are still being sent does not stop them.
        sending = False
        if self.dependency_fail_fast and any_required_failed and not reported:
            report()

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
