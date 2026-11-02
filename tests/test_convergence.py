"""Temporal convergence of the BDF solver (slow; run with `pytest -m slow`).

Three runs at dt, dt/2, dt/4 to the same physical time; the observed order
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
