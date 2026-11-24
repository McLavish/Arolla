import statistics
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from simulator.core.types import TimeDuration


class Timeout(ABC):
    @abstractmethod
    def get_global_timeout(self) -> Optional[TimeDuration]:
        pass

    @abstractmethod
    def get_attempt_timeout(self) -> Optional[TimeDuration]:
        pass

    @abstractmethod
    def record_result(self, success: bool, latency: TimeDuration) -> None:
        pass


@dataclass
class StaticTimeout(Timeout):
    """Static timeouts"""

    global_timeout: Optional[TimeDuration] = None
    attempt_timeout: Optional[TimeDuration] = None

    def get_global_timeout(self) -> Optional[TimeDuration]:
        return self.global_timeout

    def get_attempt_timeout(self) -> Optional[TimeDuration]:
        return self.attempt_timeout

    def record_result(self, success: bool, latency: TimeDuration) -> None:
        pass


@dataclass
class AdaptiveTimeout(Timeout):
    """An attempt timeout that follows the latency of the recent successful attempts.

    Example with the defaults: the 95th percentile of the last 100 successful attempts
    is 40 ms, so the next attempt gets 2 * 40 = 80 ms. While fewer than half of the last
    50 attempts succeed, the timeout is raised further, up to 10 times.
    """

    # Configuration
    initial_global_timeout: Optional[TimeDuration] = None
    initial_attempt_timeout: Optional[TimeDuration] = None

    # Adaptation parameters
    target_percentile: float = 0.95  # Use p95 latency for timeout calculation
    safety_margin: float = 2.0  # Multiply percentile by this factor
    min_samples: int = 10  # Minimum samples before adapting
    window_size: int = 100  # Number of recent results to consider
    min_timeout: TimeDuration = 1_000_000  # 1ms minimum
    max_timeout: Optional[TimeDuration] = None  # No maximum by default

    # Success rate tracking
    success_rate_window: int = 50
    min_success_rate: float = 0.5  # If success rate drops below this, increase timeout

    # Internal state
    _latencies: Deque[TimeDuration] = field(default_factory=deque, init=False)
    _successes: Deque[bool] = field(default_factory=deque, init=False)
    # Global timeout is fixed (i.e. we want to guarantee overall request completes in time)
    _current_global_timeout: Optional[TimeDuration] = field(default=None, init=False)
    _current_attempt_timeout: Optional[TimeDuration] = field(default=None, init=False)

    def __post_init__(self):
        self._latencies = deque(maxlen=self.window_size)
        self._successes = deque(maxlen=self.success_rate_window)
        self._current_global_timeout = self.initial_global_timeout
        self._current_attempt_timeout = self.initial_attempt_timeout

    def _calculate_percentile_timeout(self) -> Optional[TimeDuration]:
        """Calculate timeout based on latency percentiles using statistics.quantiles"""
        if len(self._latencies) < self.min_samples:
            return None

        # Use the same method as statistics.quantiles
        p = int(round(self.target_percentile * 100))
        p = max(1, min(99, p))  # clamp to [1, 99]

        q = statistics.quantiles(self._latencies, n=100, method="inclusive")
        percentile_latency = q[p - 1]
        adapted_timeout = int(percentile_latency * self.safety_margin)

        # Apply min/max constraints
        adapted_timeout = max(adapted_timeout, self.min_timeout)
        if self.max_timeout is not None:
            adapted_timeout = min(adapted_timeout, self.max_timeout)

        return adapted_timeout

    def _get_success_rate(self) -> float:
        if not self._successes:
            return 1.0
        return sum(self._successes) / len(self._successes)

    def _update_timeouts(self) -> None:
        """Update current timeout values based on collected metrics"""
        timeout = self._calculate_percentile_timeout()
        success_rate = self._get_success_rate()

        if timeout is None:
            # Too few successes for a percentile, for example because the starting timeout
            # is too short. The timeout can then only be raised, from its starting value.
            timeout = self.initial_attempt_timeout
            if timeout is None or success_rate >= self.min_success_rate:
                return

        if success_rate < self.min_success_rate:
            timeout = int(timeout / max(success_rate, 0.1))
            if self.max_timeout is not None:
                timeout = min(timeout, self.max_timeout)
        self._current_attempt_timeout = timeout

    def get_global_timeout(self) -> Optional[TimeDuration]:
        return self._current_global_timeout

    def get_attempt_timeout(self) -> Optional[TimeDuration]:
        return self._current_attempt_timeout

    def record_result(self, success: bool, latency: TimeDuration) -> None:
        """Record result and update timeout values"""
        if success:  # how long a failed attempt took says little about how long a success takes
            self._latencies.append(latency)
        self._successes.append(success)
        self._update_timeouts()
