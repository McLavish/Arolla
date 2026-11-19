"""Runtime object builders (services/workloads) used by ConfigLoader."""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional

from simulator.config.schema import ServiceConfigYAML, WorkloadConfig
from simulator.runtime.service import ServiceConfig, ServiceRuntime
from simulator.runtime.workload import Workload
from simulator.core.engine import Simulator
from simulator.utils.time import ms_to_ns

from .faults import build_latency_injections, build_partial_failures, build_load_spikes
from .policies import build_retry_policy, build_timeout_policy, build_load_limiter


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
    """Seed of one client's arrival stream.

    Every client used to draw its arrivals from `rng_seed` alone, so all the
    replicas of a client entry, and all entries sharing one workload block,
    arrived at exactly the same instants. The stream now also depends on where
    the client sits:

    * `position`: the rank of its entry among the entries that target the same
      service, in file order;
    * `replica`: its index among the entry's replicas.

    Entries on DIFFERENT services still get the same streams, so policy arms
    laid out as parallel service stacks stay paired, request for request. The
    first client of a service keeps `rng_seed` itself, so a scenario with one
    client per service is unchanged.
    """
    if position == 0 and replica == 0:
        return rng_seed
    base = rng_seed if rng_seed is not None else 0
    digest = hashlib.md5(f"{base}:arrivals:{position}:{replica}".encode("utf-8")).hexdigest()
    return int(digest, 16) & 0xFFFFFFFF


def build_workload(cfg: WorkloadConfig, position: int = 0, replica: int = 0) -> Workload:
    return Workload(
        base_rps=cfg.base_rps,
        duration_s=cfg.duration_s,
        load_spikes=build_load_spikes(cfg.load_spikes),
        rng_seed=arrival_seed(cfg.rng_seed, position, replica),
    )
