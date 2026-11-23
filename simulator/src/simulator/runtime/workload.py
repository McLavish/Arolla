from collections.abc import Callable
from dataclasses import dataclass, field
from typing import List, Optional
import random
from simulator.core.types import TimePoint, TimeDuration
from simulator.utils.time import ns_to_s, s_to_ns
from simulator.faults.injection import LoadSpike
from simulator.core.engine import Simulator
from simulator.faults.events import FaultEventsTracker


@dataclass
class Workload:
    base_rps: float
    duration_s: int
    load_spikes: List[LoadSpike] = field(default_factory=list)
    rng_seed: Optional[int] = None
    exact_arrivals: bool = False  # see `_drive_exact`
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed if self.rng_seed is not None else 0)

    def register_fault_events(self, tracker: FaultEventsTracker):
        """Register all load spike events with the tracker"""
        for load_spike in self.load_spikes:
            tracker.add_load_spike(load_spike.duration, load_spike.rps_multiplier)

    def rps_at(self, time: TimePoint) -> float:
        rps = self.base_rps
        for spike in self.load_spikes:
            if spike.duration.contains(time):
                rps *= spike.rps_multiplier
        return max(0.0, rps)

    def _sample_interarrival(self, time: TimePoint) -> TimeDuration:
        rps = self.rps_at(time)
        if rps <= 0.0:
            # No load right now - push a bit forward.
            return s_to_ns(0.5)
        # Use workload's own RNG so arrivals are identical across experiments
        return s_to_ns(self._rng.expovariate(rps))

    def drive(self, sim: Simulator, on_arrival: Callable[[Simulator], None]):
        if self.exact_arrivals:
            self._drive_exact(sim, on_arrival)
            return

        end_at = s_to_ns(self.duration_s)

        def tick():
            if sim.timestep >= end_at:
                return
            on_arrival(sim)
            ia = max(1, self._sample_interarrival(sim.timestep))
            sim.schedule(sim.timestep + ia, tick)

        # start at t=0
        sim.schedule(0, tick)

    def _drive_exact(self, sim: Simulator, on_arrival: Callable[[Simulator], None]):
        """Arrivals as an exact Poisson process whose rate follows the load spikes.

        Two differences from the default: the first request is drawn like every other
        (none is sent at t = 0), and a load spike starts and ends exactly on time.
        Without a load spike the requests arrive at the same instants as by default,
        minus the one at t = 0.
        """
        end_at = s_to_ns(self.duration_s)

        def schedule_next(after: TimePoint):
            arrival = self._next_arrival_after(after, end_at)
            if arrival is not None:
                sim.schedule(max(arrival, after + 1), arrive)

        def arrive():
            if sim.timestep >= end_at:
                return
            on_arrival(sim)
            schedule_next(sim.timestep)

        schedule_next(0)

    def _next_rate_change(self, time: TimePoint, end_at: TimePoint) -> TimePoint:
        """The first instant after `time` at which a load spike starts or ends, or `end_at`."""
        change = end_at
        for spike in self.load_spikes:
            for edge in (spike.duration.begin, spike.duration.end):
                if time < edge < change:
                    change = edge
        return change

    def _next_arrival_after(self, time: TimePoint, end_at: TimePoint) -> Optional[TimePoint]:
        """Draw the next arrival after `time`. None if it would land after `end_at`.

        One draw gives the number of expected arrivals that must pass before the next
        request. The rate is constant between two spike edges, so each stretch supplies
        `rate x length` of them, and the request lands in the stretch where they run out.

        Example: 1 request per second, and a spike of x1000 from 10.0 s to 10.1 s.
        A draw of 5 is made at 9.0 s. The second before the spike supplies 1, so 4 are
        left. The spike supplies 100, so the request lands 4 ms into it, at 10.004 s.
        Drawing the whole gap at the rate of 9.0 s would put it at 14 s, after the spike.
        """
        needed = self._rng.expovariate(1.0)
        while time < end_at:
            rate = self.rps_at(time)
            change = self._next_rate_change(time, end_at)
            supplied = rate * ns_to_s(change - time)
            if rate > 0.0 and needed <= supplied:
                return time + s_to_ns(needed / rate)
            needed -= supplied
            time = change
        return None
