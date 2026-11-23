"""
A retry budget earns `budget_ratio` of a retry per success, whatever the ratio.

The deposit was computed as `int(budget_ratio * 100)` hundredths of a retry, so
a ratio of 0.0867 (the `diversity/` scenarios) ran as 0.08, 0.025 as 0.02, and
any ratio below 0.01 never earned a retry at all.
"""

import pytest

from simulator.policies.retry import FixedBackoffRetryPolicy, RetryContext
from simulator.policies.retry_controls import LimiterRetryBudgetPolicy, RetryBudgetPolicy


def client_budget(ratio):
    return RetryBudgetPolicy(
        inner=FixedBackoffRetryPolicy(max_attempts=100, delay=0),
        budget_ratio=ratio,
        min_retries_per_sec=0,
        max_retries=10,
    )


def retries_earned_client(ratio, successes):
    """Retries a drained client-side budget can pay for after `successes` first attempts."""
    budget = client_budget(ratio)
    while budget.next_delay(RetryContext(attempt=1, now=0))[0]:
        pass
    for _ in range(successes):
        budget.record_attempt(RetryContext(attempt=1, now=0), True)
    earned = 0
    while budget.next_delay(RetryContext(attempt=1, now=0))[0]:
        earned += 1
    return earned


def retries_earned_server(ratio, successes):
    """The same for the server-side budget."""
    budget = LimiterRetryBudgetPolicy(budget_ratio=ratio, max_retries=10)
    while budget.next_delay(RetryContext(attempt=1, now=0))[0]:
        pass
    for _ in range(successes):
        budget.add_result(True)
    earned = 0
    while budget.next_delay(RetryContext(attempt=1, now=0))[0]:
        earned += 1
    return earned


@pytest.mark.parametrize("earned", [retries_earned_client, retries_earned_server])
class TestDeposit:
    def test_a_ratio_with_more_than_two_decimals_is_not_rounded_down(self, earned):
        # 100 successes at 0.0867 are 8.67 retries: 8 whole ones. 12 are 1.04.
        assert earned(0.0867, 100) == 8
        assert earned(0.0867, 12) == 1   # 0.96 when the ratio ran as 0.08
        assert earned(0.025, 40) == 1    # 0.80 when it ran as 0.02

    def test_a_small_ratio_still_earns_retries(self, earned):
        assert earned(0.005, 199) == 0
        assert earned(0.005, 200) == 1   # never, when it ran as 0
        assert earned(0.005, 1000) == 5

    def test_round_ratios_are_what_they_were(self, earned):
        assert earned(0.1, 9) == 0
        assert earned(0.1, 10) == 1
        assert earned(0.5, 7) == 3
        assert earned(1.0, 4) == 4

    def test_the_balance_is_still_capped(self, earned):
        assert earned(0.5, 1000) == 10   # max_retries


def test_only_first_attempt_successes_fund_the_client_budget():
    budget = client_budget(0.5)
    while budget.next_delay(RetryContext(attempt=1, now=0))[0]:
        pass
    for _ in range(10):
        budget.record_attempt(RetryContext(attempt=2, now=0), True)   # a retry that succeeded
        budget.record_attempt(RetryContext(attempt=1, now=0), False)  # a first attempt that failed
    assert not budget.next_delay(RetryContext(attempt=1, now=0))[0]
