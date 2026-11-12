"""Per-step telemetry, the same on every backend: where a time step's time goes and how
the linear solvers behave.

With telemetry=True the driver writes one JSON line per time step to
Stats/telemetry.jsonl (rank 0):

    {"tstep": 12, "t": 0.024, "passes": 3, "wall": 0.81,
     "blocks": {"phase_field_assemble": 0.06, ..., "hooks": 0.01},
     "solvers": {"pressure": {"solves": 3, "its": 21, "pc_setups": 1, "seconds": 0.012},
                 ...}}

`wall` is the step's wall time and `blocks` the driver's parts of it, in seconds, each
the largest over the processes. `solvers` are the Krylov solvers by role (velocity,
pressure, phase_field, projection): solves, iterations, preconditioner setups (PETSc's
PCSetUp events in the solver's log stage, nested ones included, such as the Schur
preconditioner's inner AMG), the time in their solves and the part of it in those
setups (pc_seconds). Both backends' solvers are
PETSc KSPs, so the counts mean the same on each.

Telemetry off (the default), nothing is wrapped or timed: `solver` returns the solver
itself and `block` does nothing.
"""
import json
import time
from contextlib import contextmanager

import numpy as np

# Solver roles by the name of their settings in NS_parameters.
ROLES = {"velocity_krylov_solver": "velocity", "pressure_krylov_solver": "pressure",
         "phase_field_krylov_solver": "phase_field", "velocity_update_solver": "projection"}
BLOCKS = ("phase_field_assemble", "phase_field_solve", "first_inner_assemble",
          "velocity_assemble", "velocity_solve", "pressure_assemble", "pressure_solve",
          "pressure_update", "velocity_update", "hooks")


class _Off:
    """Telemetry switched off."""
    def solver(self, sol, name):
        return sol

    @contextmanager
    def block(self, name):
        yield

    def start_step(self):
        pass

    def end_step(self, tstep, t, passes):
        pass

    def close(self):
        pass


class _RecordedSolver:
    """A Krylov solver whose solves are counted, timed and logged in a PETSc stage of
    their own; everything else is the solver's."""

    def __init__(self, sol, record, stage):
        object.__setattr__(self, "_sol", sol)
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_stage", stage)

    def __getattr__(self, key):
        return getattr(self._sol, key)

    def __setattr__(self, key, value):
        setattr(self._sol, key, value)

    def solve(self, *args):
        self._stage.push()
        t0 = time.perf_counter()
        try:
            out = self._sol.solve(*args)
        finally:
            dt = time.perf_counter() - t0
            self._stage.pop()
        r = self._record
        r["solves"] += 1
        r["its"] += self._sol.ksp().getIterationNumber()
        r["seconds"] += dt
        return out


class Telemetry:
    """Telemetry written to `path` (on rank 0 of `comm`)."""
    def __init__(self, path, comm):
        from petsc4py import PETSc
        PETSc.Log.begin()
        self.path, self.comm = path, comm
        self._pcsetup = PETSc.Log.Event("PCSetUp")
        self._stages = {}
        self._solvers = {}
        self._pc_before = {}
        self._pc_time_before = {}
        self._blocks = dict.fromkeys(BLOCKS, 0.0)
        # appended to: a restart writes into its run's folder, after the steps before it
        self._file = open(path, "a") if comm.rank == 0 else None
        self._t0 = None

    def solver(self, sol, name):
        from petsc4py import PETSc
        role = ROLES.get(name, name)
        if role not in self._stages:
            self._stages[role] = PETSc.Log.Stage("twoasis " + role)
            self._solvers[role] = dict(solves=0, its=0, seconds=0.0)
            self._pc_before[role] = 0
        return _RecordedSolver(sol, self._solvers[role], self._stages[role])

    @contextmanager
    def block(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._blocks[name] = self._blocks.get(name, 0.0) + time.perf_counter() - t0

    def start_step(self):
        self._t0 = time.perf_counter()

    def end_step(self, tstep, t, passes):
        wall = time.perf_counter() - self._t0
        roles = sorted(self._solvers)
        info = [self._pcsetup.getPerfInfo(self._stages[r]) for r in roles]
        pcs = [i["count"] for i in info]
        pc_times = [i["time"] for i in info]
        local = np.array([wall] + list(self._blocks.values())
                         + [self._solvers[r]["seconds"] for r in roles]
                         + [pt - self._pc_time_before.get(r, 0.0) for r, pt in zip(roles, pc_times)])
        from mpi4py import MPI
        # Elementwise: the pickle-based allreduce would compare whole arrays.
        times = np.empty_like(local)
        self.comm.Allreduce(local, times, op=MPI.MAX)
        if self._file is not None:
            line = dict(tstep=int(tstep), t=float(t), passes=int(passes), wall=times[0],
                        blocks=dict(zip(self._blocks, times[1:1 + len(self._blocks)])),
                        solvers={})
            nb = 1 + len(self._blocks)
            for k, (r, pc) in enumerate(zip(roles, pcs)):
                s = self._solvers[r]
                line["solvers"][r] = dict(solves=s["solves"], its=s["its"],
                                          pc_setups=int(pc - self._pc_before[r]),
                                          seconds=times[nb + k],
                                          pc_seconds=times[nb + len(roles) + k])
            self._file.write(json.dumps(line) + "\n")
            self._file.flush()
        for r, pc, pt in zip(roles, pcs, pc_times):
            self._pc_before[r] = pc
            self._pc_time_before[r] = pt
            self._solvers[r].update(solves=0, its=0, seconds=0.0)
        self._blocks = dict.fromkeys(self._blocks, 0.0)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


#: The telemetry in force, set by the driver (start).
current = _Off()


def start(path=None, comm=None):
    """The run's telemetry, from now on `current`: writing to `path`, or off without
    one. Returns it."""
    global current
    current = Telemetry(path, comm) if path is not None else _Off()
    return current


def read(path):
    """The records of a telemetry.jsonl file, a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
