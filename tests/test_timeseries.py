"""Reading a state back from an XDMF time series (common/io.py, load_timeseries).

A helper process writes known periodic fields at two times with the solver's
own writer (save_tstep_solution_h5), and another process reads them back. The
series hold vertex values, so the fields read must equal the written ones at
every vertex, whatever the degrees and the number of processes on either side.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, mpirun_size

pytest.importorskip("h5py")

# The helper builds the solver's spaces on a periodic channel and either writes
# fields f(x, y, t) at t = 0.1, 0.2 or reads them and reports, per field, the
# largest difference from f at the vertices.
HELPER = r'''
import json, sys
import numpy as np
from dolfin import (as_vector, Expression, FiniteElement, Function, FunctionSpace, interpolate,
                    MixedElement, MPI, set_log_level, assign)
from twoasis.common.io import create_initial_folders, load_timeseries, save_tstep_solution_h5
from twoasis.common.utilities import AssignedVectorFunction
from twoasis.problems.TPfracStep.shared import PeriodicBox, rectangle_mesh
set_log_level(40)

mode, folder, res, vdeg, pdeg, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
mesh = rectangle_mesh(2.0, 1.0, res)
pb = PeriodicBox({0: (0, 2.0)})
V = FunctionSpace(mesh, "CG", vdeg, constrained_domain=pb)
Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pb)
el = FiniteElement("CG", mesh.ufl_cell(), pdeg)
W = FunctionSpace(mesh, MixedElement([el, el]), constrained_domain=pb)
VV = dict(u0=V, u1=V, p=Q, phig=W)
q_ = {k: Function(S) for k, S in VV.items()}
CODE = dict(u0="sin(pi*x[0])*x[1] + t", u1="cos(pi*x[0]) - 2*t*x[1]*x[1]",
            p="x[1]*sin(pi*x[0]) + 3*t", phi="tanh((x[1] - 0.5 - 0.1*cos(pi*x[0]))/0.1) * t",
            g="x[1]*x[1]*t + cos(pi*x[0])")
f = lambda name, t: Expression(CODE[name], t=t, degree=4)

if mode == "write":
    newfolder, tstepfiles = create_initial_folders(
        folder, None, ["u0", "u1", "p", "phig"], 0, lambda s: None, [], True)
    for t in (0.1, 0.2):
        for k in ("u0", "u1", "p"):
            q_[k].interpolate(f(k, t))
        for i, k in enumerate(("phi", "g")):
            assign(q_["phig"].sub(i), interpolate(f(k, t), W.sub(i).collapse()))
        u_ = as_vector([q_["u0"], q_["u1"]])
        save_tstep_solution_h5(t, q_, u_, newfolder, tstepfiles, pb, True, ["u0", "u1"],
                               AssignedVectorFunction, [], {})
    for file in tstepfiles.values():
        file.close()
else:
    result = {}
    for time in (None, 0.1):
        try:
            t = load_timeseries(folder, q_, VV, pb, time=time)
        except ValueError as e:
            result["error"] = str(e)
            break
        fields = dict(u0=q_["u0"], u1=q_["u1"], p=q_["p"],
                      phi=q_["phig"].split(deepcopy=True)[0], g=q_["phig"].split(deepcopy=True)[1])
        errors = {}
        for k, fn in fields.items():
            exact = interpolate(f(k, t), FunctionSpace(mesh, "CG", 1)).compute_vertex_values()
            local = np.abs(fn.compute_vertex_values(mesh) - exact).max()
            errors[k] = MPI.max(MPI.comm_world, float(local))
        result[str(time)] = dict(t=t, errors=errors)
    if MPI.rank(MPI.comm_world) == 0:
        json.dump(result, open(out, "w"))
'''


def helper(tmp_path, mode, folder, nprocs=1, res=0.0625, vdeg=2, pdeg=1):
    script = tmp_path / "helper.py"
    script.write_text(HELPER)
    out = tmp_path / "result_{}_{}_{}_{}.json".format(mode, nprocs, vdeg, pdeg)
    cmd = [sys.executable, str(script), mode, str(folder), str(res), str(vdeg), str(pdeg), str(out)]
    if nprocs > 1:
        cmd = ["mpirun", "-np", str(nprocs)] + cmd
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(out.read_text()) if mode == "read" else None


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    """Time series written in serial, with P2 velocity and a P1 phase field."""
    wd = tmp_path_factory.mktemp("series")
    helper(wd, "write", wd / "results")
    return wd, wd / "results" / "data" / "1" / "Timeseries"


def assert_exact(result, tol=1e-12):
    for time, t_expected in (("None", 0.2), ("0.1", 0.1)):
        assert result[time]["t"] == pytest.approx(t_expected)
        for name, err in result[time]["errors"].items():
            assert err < tol, "{} at t = {}: max vertex error {:.2e}".format(name, time, err)


def test_read_back(tmp_path, series):
    """Last time and a chosen time, into the spaces that wrote them."""
    _, folder = series
    assert_exact(helper(tmp_path, "read", folder))


def test_read_into_other_degrees(tmp_path, series):
    """Into P1 velocity and a P2 phase field: interpolated from the vertex values."""
    _, folder = series
    assert_exact(helper(tmp_path, "read", folder, vdeg=1, pdeg=2))


def test_read_in_parallel(tmp_path, series):
    """Written in serial, read on two processes (and the other way round)."""
    if mpirun_size(2) != 2:
        pytest.skip("mpirun -np 2 does not start a two-rank job here")
    _, folder = series
    assert_exact(helper(tmp_path, "read", folder, nprocs=2))
    helper(tmp_path, "write", tmp_path / "results_np2", nprocs=2)
    assert_exact(helper(tmp_path, "read", tmp_path / "results_np2" / "data" / "1" / "Timeseries"))


def test_other_mesh_is_an_error(tmp_path, series):
    """A series from a different mesh is reported, not read into the wrong points."""
    _, folder = series
    result = helper(tmp_path, "read", folder, res=0.05)
    assert "different mesh" in result["error"]


def test_match_points():
    """Matches within the tolerance, including points whose coordinates round to
    the other side of a multiple of the tolerance; a shifted mesh is an error."""
    from twoasis.common.io import match_points
    rng = np.random.default_rng(1)
    points = rng.random((1000, 3))
    order = rng.permutation(1000)
    tol = 1e-8
    queries = points[order] + rng.uniform(-1e-6, 1e-6, (1000, 3)) * tol  # round-off
    # Two points that straddle a rounding boundary of every coordinate.
    points[order[:2]] = 0.5 * tol - 1e-3 * tol
    points[order[1]] += 0.6
    queries[:2] = points[order[:2]] + 2e-3 * tol
    assert np.array_equal(match_points(points, queries, tol), order)
    with pytest.raises(ValueError, match="different mesh"):
        match_points(points, queries + 2 * tol, tol)
