"""Temporal convergence of the BDF solver (slow; run with `pytest -m slow`), and a
quick check that IPCS applies the manufactured phase-field source.

Two studies: self-convergence of the periodic bubble, and errors against the
exact Taylor-Green solution.

Bubble: three runs at dt, dt/2, dt/4 to the same physical time; the observed order
p = log2(|q(dt) - q(dt/2)| / |q(dt/2) - q(dt/4)|) must be ~2 for BDF2.
Inner-loop and Krylov tolerances are tightened so temporal error dominates:
with the defaults, solver noise (~1e-6) would drown the finest difference.

Calibration on the reference platform gave orders 2.02-2.23 with the finest
differences >= 1.3e-6, two orders of magnitude above the 1e-8 precision of
tdata.dat, so the bounds below have a wide margin in both directions. A
first-order regression (e.g. a term lagged inconsistently) measures ~1.
"""
import json
import math

import numpy as np
import pytest

from conftest import BUBBLE, COLS, load_tdata, run_twoasis

TIGHT = dict(
    max_error=1e-9,
    max_iter=200,
    velocity_krylov_solver=json.dumps({"relative_tolerance": 1e-10, "maximum_iterations": 2000}),
    pressure_krylov_solver=json.dumps({"relative_tolerance": 1e-10, "maximum_iterations": 500}),
    phase_field_krylov_solver=json.dumps({"relative_tolerance": 1e-10, "maximum_iterations": 5000}),
    krylov_solvers=json.dumps({"absolute_tolerance": 1e-14}),
)


@pytest.mark.slow
def test_bdf2_temporal_order(tmp_path):
    T = 0.032
    dts = [0.008, 0.004, 0.002]
    finals = []
    for dt in dts:
        wd = tmp_path / "dt_{}".format(dt)
        wd.mkdir()
        params = dict(BUBBLE, dt=dt, checkpoint=1000000, **TIGHT)
        run_twoasis(wd, solver="BDF", T=T, **params)
        row = load_tdata(wd)[-1]
        assert abs(row[COLS["t"]] - T) < 1e-12
        finals.append(row)

    orders = {}
    for name in ["u0m", "E_kin", "E_int"]:
        q0, q1, q2 = (f[COLS[name]] for f in finals)
        d1, d2 = abs(q0 - q1), abs(q1 - q2)
        # Guard against differences collapsing into solver/print noise, which
        # would make the ratio meaningless rather than the scheme wrong.
        assert d2 > 1e-7, "{}: finest difference {:.3g} is below the noise floor".format(name, d2)
        orders[name] = math.log2(d1 / d2)

    assert all(1.7 < p < 2.6 for p in orders.values()), \
        "observed temporal orders not ~2: {}".format(
            {k: round(v, 2) for k, v in orders.items()})


# TaylorGreen2D has an analytic solution for the velocity, the pressure and the
# phase field, and writes the L2 errors to tdata.dat. The parameters make the
# solution decay fast enough (mu = 0.1, phi0 = 0.5) and the mesh fine enough
# (P2 velocity, N = 64) that the temporal error dominates the spatial one down
# to dt = 0.01. Calibration: velocity orders 1.94 and 1.91, phase-field orders
# 2.05 and 2.02.
TG_ERR_COLS = dict(u0=4, u1=5, p=6, phi=7)
TAYLOR_GREEN = dict(problem="TaylorGreen2D", solver="BDF", mu=[0.1, 0.1], phi0=0.5, N=64,
                    velocity_degree=2, plot_interval=1000000, timestamps_interval=1000000,
                    save_step=1000000, checkpoint=1000000, folder="results")


@pytest.mark.slow
def test_bdf2_taylor_green_exact(tmp_path):
    """BDF2 errors against the exact solution decrease as dt^2.

    initialize() provides the exact solution at t - dt, so the run must also
    take a second-order first step.
    """
    T = 0.2
    errors = []
    for dt in [0.04, 0.02, 0.01]:
        wd = tmp_path / "dt_{}".format(dt)
        wd.mkdir()
        proc = run_twoasis(wd, T=T, dt=dt, stat_interval=round(T / dt), **TAYLOR_GREEN, **TIGHT)
        assert "BDF1 on first timestep: False" in proc.stdout
        row = load_tdata(wd)[-1]
        assert abs(row[1] - T) < 1e-12
        errors.append(row)

    orders = {}
    for name in ["u0", "phi"]:
        e = [row[TG_ERR_COLS[name]] for row in errors]
        orders[name] = [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
    assert all(1.8 < p < 2.3 for ps in orders.values() for p in ps), \
        "observed orders against the exact solution not ~2: {}".format(
            {k: [round(p, 2) for p in v] for k, v in orders.items()})


def test_ipcs_taylor_green_phase_field(tmp_path):
    """IPCS applies the phase-field source term of the manufactured solution.
    Without it the phase-field error at t = 0.1 is 0.36; with it 0.018 (BDF2:
    5e-4)."""
    run_twoasis(tmp_path, T=0.1, dt=0.01, stat_interval=10,
                **dict(TAYLOR_GREEN, solver="IPCS", N=32))
    assert load_tdata(tmp_path)[-1, TG_ERR_COLS["phi"]] < 0.05
