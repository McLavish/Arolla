"""Runtime object builders (services/workloads) used by ConfigLoader."""

from __future__ import annotations

import random
from typing import List, Optional

from simulator.config.schema import ServiceConfigYAML, WorkloadConfig
from simulator.runtime.service import ServiceConfig, ServiceRuntime
from simulator.runtime.workload import Workload
from simulator.core.engine import Simulator
from simulator.utils.time import ms_to_ns

from .faults import build_latency_injections, build_partial_failures, build_load_spikes
from .policies import build_retry_policy, build_timeout_policy, build_load_limiter
from .seeds import get_stable_seed


def build_service(
    cfg: ServiceConfigYAML,
    sim: Simulator,
    dependencies: Optional[List[ServiceRuntime]] = None,
    dependency_optionality: Optional[List[bool]] = None,
    dependency_call_pattern: str = "sequential",
    seed: Optional[int] = None,
) -> ServiceRuntime:
    # Use a service-local RNG seed for policy construction to preserve isolation.
    svc_rng = random.Random(seed if seed is not None else 0)

    retry_policy = build_retry_policy(cfg.retry, svc_rng)
    timeout_policy = build_timeout_policy(cfg.timeout)
    load_limiter = build_load_limiter(
        cfg.circuit_breaker,
        cfg.rate_limiter,
        retry_budget=cfg.retry_budget,
        global_retry_budget=cfg.global_retry_budget,
        aimd_global_retry_budget=cfg.aimd_global_retry_budget,
        arolla_retry_budget=cfg.arolla_retry_budget,
    )

    service_cfg = ServiceConfig(
        name=cfg.name,
        latency_median=ms_to_ns(cfg.latency.median_ms),
        latency_lognorm_sigma=cfg.latency.lognorm_sigma,
        workers=cfg.workers,
        queue_capacity=cfg.queue_capacity,
        latency_injections=build_latency_injections(cfg.latency_injections),
        partial_failures=build_partial_failures(cfg.partial_failures),
        retry=retry_policy,
        timeout=timeout_policy,
        load_limiter=load_limiter,
    )

    return ServiceRuntime(
        cfg=service_cfg,
        dependencies=dependencies or [],
        dependency_optionality=dependency_optionality or [],
        dependency_call_pattern=dependency_call_pattern,
    ).bind(seed=seed, record_events=True)


def arrival_seed(rng_seed: Optional[int], position: int = 0, replica: int = 0) -> Optional[int]:
    """Seed for the arrival times of one client.

    `rng_seed` is the seed in the client's workload block. Two numbers tell
    apart the clients that share it:

    * `position`: which client entry it comes from, counting only the entries
      that target the same service, in file order (0, 1, 2, ...);
    * `replica`: which copy it is among that entry's `replicas` (0, 1, 2, ...).

    Each (position, replica) gets its own seed. Seeding them all with
    `rng_seed` made every client arrive at exactly the same instants, so twenty
    replicas behaved like one client sending twenty requests at a time.

    Two cases deliberately get the same stream as before:

    * client (0, 0) of a service keeps `rng_seed` itself, so a scenario with
      one client per service gives the same results as before;
    * the seed does not depend on which service the client targets, so policy
      arms laid out as parallel service stacks see identical traffic.
    """
    if position == 0 and replica == 0:
        return rng_seed
    base = rng_seed if rng_seed is not None else 0
    return get_stable_seed(base, f"arrivals:{position}:{replica}")


def build_workload(cfg: WorkloadConfig, position: int = 0, replica: int = 0) -> Workload:
    return Workload(
        base_rps=cfg.base_rps,
        duration_s=cfg.duration_s,
        load_spikes=build_load_spikes(cfg.load_spikes),
        rng_seed=arrival_seed(cfg.rng_seed, position, replica),
    )
