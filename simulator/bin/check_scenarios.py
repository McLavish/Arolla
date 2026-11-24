"""Check that a change to the simulator keeps the results of the shipped scenarios.

    python simulator/bin/check_scenarios.py OLD_CHECKOUT NEW_CHECKOUT [--jobs N]

Both arguments are checkouts of the repository, for example `main` and a pull request.
Every scenario in OLD_CHECKOUT/simulator/experiments/yaml is run once with the simulator
code of each checkout. The script lists the scenarios whose results differ, and exits
with an error if there is at least one.

The results of a run are every attempt each client saw (start, end, success, reason),
every attempt each service recorded, and the state of each service's random generator
at the end. A change that does not change the behaviour, such as a refactor or a new
option that is off by default, gives exactly the same results.

To keep the check short, each scenario sends requests for at most 10 seconds (less for
a very busy one), and a sweep is cut to its first 60 combinations.

The script reads some internals of the simulator (`client.roots`, `service._events`,
`service._rng`). If those change, the script has to change with them.
"""

import copy
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

MAX_SECONDS = 10
MAX_REQUESTS = 300_000
MAX_SWEEP_POINTS = 60


def scenario_points(checkout: Path):
    """Every scenario to run, as (name, config): one per file, or one per sweep combination."""
    sys.path.insert(0, str(checkout / "simulator/bin"))
    import run_sweep  # the sweep helpers of OLD_CHECKOUT

    points = []
    for path in sorted((checkout / "simulator/experiments/yaml").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "services" not in raw:
            continue
        sweeps = list(run_sweep.find_sweeps(raw).values())
        combinations = list(itertools.product(*[s["values"] for s in sweeps])) if sweeps else [()]
        for index, combination in enumerate(combinations[:MAX_SWEEP_POINTS]):
            config = copy.deepcopy(raw)
            config.pop("sweeps", None)
            for sweep, value in zip(sweeps, combination):
                for dotted_path in sweep["paths"]:
                    run_sweep.set_nested_value(config, dotted_path, value)
            shorten(config)
            points.append((f"{path.relative_to(checkout)}#{index}", config))
    return points


def shorten(config):
    """Cut every workload to MAX_SECONDS, or to fewer seconds if the scenario would send
    more than MAX_REQUESTS requests. Example: at 100 000 requests per second, 3 seconds."""
    clients = config.get("clients") or []
    workloads = [client["workload"] for client in clients]
    rps = sum(client["workload"]["base_rps"] * client.get("replicas", 1) for client in clients)
    if config.get("workload"):
        workloads.append(config["workload"])
        rps += config["workload"]["base_rps"]
    seconds = max(1, min(MAX_SECONDS, int(MAX_REQUESTS / max(rps, 1))))
    for workload in workloads:
        workload["duration_s"] = min(workload["duration_s"], seconds)


def run_point(point):
    """Run one scenario and return a hash of its results, or the error it raised."""
    name, config = point
    sys.setrecursionlimit(1_000_000)  # long chains of callbacks in the busiest scenarios
    from simulator.config.loader import ConfigLoader
    from simulator.config.schema import ExperimentConfig

    try:
        sim, clients, workloads, _, services = ConfigLoader.build_simulation(ExperimentConfig(**config))
        for client, workload in zip(clients, workloads):
            workload.drive(sim, lambda s, c=client: c.start_request(s))
        sim.run()
    except Exception as error:  # a scenario that fails must show up in the report
        return name, f"error: {type(error).__name__}: {error}"

    digest = hashlib.sha256()
    for client in clients:
        digest.update(f"{client.cfg.name}|{len(client.roots)}\n".encode())
        for root in client.roots:
            digest.update(repr([
                (a.interval.begin, a.interval.end, a.success, a.drop_reason.name, a.deadline,
                 a.queue_size_at_end)
                for a in root.attempts
            ]).encode())
    for service_name in sorted(services):
        service = services[service_name]
        digest.update(f"{service_name}|{service.in_flight}|{len(service.queue)}\n".encode())
        digest.update(repr([
            (e[0], e[1], e[2], e[3].name, e[4], e[5], e[6]) for e in service._events
        ]).encode())
        digest.update(repr(service._rng.getstate()).encode())
        digest.update(repr(sorted(service.get_admission_stats().items())).encode())
    digest.update(f"end|{sim.timestep}\n".encode())
    return name, digest.hexdigest()


def results_of(checkout: Path, points_file: str, jobs: int):
    """Run every scenario with the code of `checkout`, in a fresh Python process."""
    env = dict(os.environ, PYTHONPATH=str(checkout / "simulator/src"))
    results_file = points_file + ".results"
    command = [sys.executable, __file__, "--run", points_file, results_file, str(jobs)]
    subprocess.run(command, env=env, check=True)
    results = json.load(open(results_file))
    os.unlink(results_file)
    return results


def report(lines):
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as file:
            file.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "--run":  # inside `results_of`: run the points, write the results
        points = json.load(open(sys.argv[2]))
        with ProcessPoolExecutor(max_workers=int(sys.argv[4])) as pool:
            results = dict(pool.map(run_point, points, chunksize=1))
        json.dump(results, open(sys.argv[3], "w"))
        return

    old, new = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else os.cpu_count()
    points = scenario_points(old)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as file:
        json.dump(points, file)
    before = results_of(old, file.name, jobs)
    after = results_of(new, file.name, jobs)
    os.unlink(file.name)

    changed = [name for name, _ in points if before[name] != after[name]]
    failing = [name for name, _ in points if after[name].startswith("error")]
    lines = [f"### Shipped scenarios: {len(points)} compared, {len(changed)} with different results"]
    lines += [f"- `{name}`" for name in changed]
    if failing:
        lines += ["", f"{len(failing)} scenarios fail with the new code:"]
        lines += [f"- `{name}`: {after[name]}" for name in failing]
    if changed:
        lines += ["", "If the change is meant to change results (a bug fix, for example), "
                      "say in the pull request which scenarios change and why."]
    report(lines)
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
