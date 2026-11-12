"""The speed gate of the DOLFINx backend: a time step on DOLFINx may take at most 5% longer
than on dolfin (docs: the merge plan's speed policy).

    python3 misc/speed_gate.py [--np 1] [--repeat 3] [--steps 30] [--warmup 10] [--out DIR]

Each case runs `repeat` times on each backend, the backends interleaved (dolfin,
dolfinx, dolfin, ...) so that a slow spell of the machine hits both, with telemetry on
(twoasis/common/telemetry.py). A run's time per step is the mean wall time of its steps
after the first `warmup` (which take more inner passes and include the JIT compilation);
a backend's is its fastest run's. The gate is that ratio, DOLFINx over dolfin, per
case. The fastest runs' blocks, solvers and iterations are reported, for information. Exit
status 1 when a case fails the gate.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = 1.05
# Two cases: matched properties with P2 (PeriodicBubble2D), and Hysing's case 1 with
# unequal densities and viscosities (Bubble2D). dt as the problems set it.
CASES = {"PeriodicBubble2D": 0.00025, "Bubble2D": 0.002}


def run(problem, dt, backend, steps, nproc, folder):
    env = dict(os.environ, TWOASIS_BACKEND=backend, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    args = ["twoasis", "TPfracStep", "problem=" + problem, "T={!r}".format(dt * (steps + 0.5)),
            "telemetry=true", "stat_interval=1000000", "save_step=1000000",
            "checkpoint=1000000", "plot_interval=1000000", "timestamps_interval=1000000",
            "folder=" + str(folder)]
    if nproc > 1:
        args = ["mpirun", "-np", str(nproc)] + args     # the ranks inherit `env`
    log = folder.with_suffix(".log")
    with open(log, "w") as f:
        proc = subprocess.run(args, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=folder.parent)
    if proc.returncode != 0:
        raise RuntimeError("{} on {} failed: see {}".format(problem, backend, log))
    (stats,) = folder.glob("data/*/Stats/telemetry.jsonl")
    return [json.loads(line) for line in open(stats)]


def summary(records, warmup):
    """Per-step means over the steps after warmup: wall, blocks and solver seconds,
    and the solvers' iterations."""
    r = records[warmup:]
    n = len(r)
    out = {"wall": sum(x["wall"] for x in r) / n, "passes": sum(x["passes"] for x in r) / n}
    for k in r[0]["blocks"]:
        out["block " + k] = sum(x["blocks"][k] for x in r) / n
    for s in r[0]["solvers"]:
        out["solver " + s] = sum(x["solvers"][s]["seconds"] for x in r) / n
        out["its " + s] = sum(x["solvers"][s]["its"] for x in r) / n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--cases", nargs="*", default=list(CASES))
    a = p.parse_args()
    out = Path(a.out or tempfile.mkdtemp(prefix="speed_gate."))
    out.mkdir(parents=True, exist_ok=True)
    print("runs in", out)
    failed = False
    for case in a.cases:
        best = {}
        for k in range(a.repeat):
            for backend in ("dolfin", "dolfinx"):
                folder = out / "{}_{}_np{}_{}".format(case, backend, a.np, k)
                s = summary(run(case, CASES[case], backend, a.steps, a.np, folder), a.warmup)
                print("  {} {} run {}: {:.4f} s/step".format(case, backend, k, s["wall"]))
                if backend not in best or s["wall"] < best[backend]["wall"]:
                    best[backend] = s
        d, x = best["dolfin"], best["dolfinx"]
        ratio = x["wall"] / d["wall"]
        ok = ratio <= GATE
        failed |= not ok
        print("{}: {:.4f} s/step on dolfin, {:.4f} on DOLFINx, ratio {:.3f} -> {}".format(
            case, d["wall"], x["wall"], ratio, "pass" if ok else "FAIL (gate {})".format(GATE)))
        for key in d:
            if key == "wall":
                continue
            if key.startswith(("its", "passes")):
                print("    {:28s} {:9.2f} {:9.2f}".format(key, d[key], x[key]))
            else:
                print("    {:28s} {:9.4f} {:9.4f}  x{:.2f}".format(
                    key, d[key], x[key], x[key] / d[key] if d[key] > 0 else float("nan")))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
