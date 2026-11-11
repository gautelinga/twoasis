"""Reading a state back from an XDMF time series (common/io.py, load_timeseries).

A helper process writes known periodic fields at two times with the solver's
own writer (save_tstep_solution_h5), and another process reads them back. The
fields read must equal the written ones at every vertex, whatever the degrees
and the number of processes on either side. The series (XDMFFile2) also hold
the midside values of a P2 field, so a field read into the degree it was written
with must equal the written one at every dof.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, require_mpi

pytest.importorskip("h5py")

# The helper builds the solver's spaces on a periodic channel (2D) or box (3D)
# and either writes fields f(x, t) at t = 0.1, 0.2 or reads them and reports, per
# field, the largest difference from f at the vertices and at the dofs.
HELPER = r'''
import json, sys
import numpy as np
from dolfin import (as_vector, BoxMesh, Expression, FiniteElement, Function, FunctionSpace,
                    interpolate, MixedElement, MPI, Point, set_log_level, assign)
from twoasis.common.io import create_initial_folders, load_timeseries, save_tstep_solution_h5
from twoasis.common.utilities import AssignedVectorFunction
from twoasis.problems.TPfracStep.shared import PeriodicBox, rectangle_mesh
set_log_level(40)

mode, folder, res, vdeg, pdeg, out, dim = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
mesh = rectangle_mesh(2.0, 1.0, res) if dim == 2 else \
    BoxMesh(Point(0, 0, 0), Point(2.0, 1.0, 1.0), int(2 / res), int(1 / res), int(1 / res))
us = ["u{}".format(i) for i in range(dim)]
pb = PeriodicBox({0: (0, 2.0)})
V = FunctionSpace(mesh, "CG", vdeg, constrained_domain=pb)
Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pb)
el = FiniteElement("CG", mesh.ufl_cell(), pdeg)
W = FunctionSpace(mesh, MixedElement([el, el]), constrained_domain=pb)
VV = dict({u: V for u in us}, p=Q, phig=W)
q_ = {k: Function(S) for k, S in VV.items()}
CODE = dict(u0="sin(pi*x[0])*x[1] + t", u1="cos(pi*x[0]) - 2*t*x[1]*x[1]", u2="x[1]*sin(pi*x[0]) - t",
            p="x[1]*sin(pi*x[0]) + 3*t", phi="tanh((x[1] - 0.5 - 0.1*cos(pi*x[0]))/0.1) * t",
            g="x[1]*x[1]*t + cos(pi*x[0])")
f = lambda name, t: Expression(CODE[name], t=t, degree=4)

if mode == "write":
    newfolder, tstepfiles = create_initial_folders(
        folder, None, us + ["p", "phig"], 0, lambda s: None, [])
    for t in (0.1, 0.2):
        for k in us + ["p"]:
            q_[k].interpolate(f(k, t))
        for i, k in enumerate(("phi", "g")):
            assign(q_["phig"].sub(i), interpolate(f(k, t), W.sub(i).collapse()))
        u_ = as_vector([q_[u] for u in us])
        save_tstep_solution_h5(t, q_, u_, newfolder, tstepfiles, AssignedVectorFunction, {})
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
        fields = dict({u: q_[u] for u in us}, p=q_["p"],
                      phi=q_["phig"].split(deepcopy=True)[0], g=q_["phig"].split(deepcopy=True)[1])
        errors, dof_errors = {}, {}
        for k, fn in fields.items():
            exact = interpolate(f(k, t), FunctionSpace(mesh, "CG", 1)).compute_vertex_values()
            local = np.abs(fn.compute_vertex_values(mesh) - exact).max()
            errors[k] = MPI.max(MPI.comm_world, float(local))
            exact = interpolate(f(k, t), fn.function_space()).vector().get_local()
            local = np.abs(fn.vector().get_local() - exact).max() if len(exact) else 0.
            dof_errors[k] = MPI.max(MPI.comm_world, float(local))
        # against the physical pressure p + phi*g, at the pressure dofs
        I = lambda k: interpolate(f(k, t), Q).vector().get_local()
        P = I("p") + I("phi") * I("g")
        local = np.abs(q_["p"].vector().get_local() - P).max() if len(P) else 0.
        dof_errors["P"] = MPI.max(MPI.comm_world, float(local))
        result[str(time)] = dict(t=t, errors=errors, dof_errors=dof_errors)
    if MPI.rank(MPI.comm_world) == 0:
        json.dump(result, open(out, "w"))
'''


def helper(tmp_path, mode, folder, nprocs=1, res=0.0625, vdeg=2, pdeg=1, dim=2):
    script = tmp_path / "helper.py"
    script.write_text(HELPER)
    out = tmp_path / "result_{}_{}_{}_{}_{}.json".format(mode, nprocs, vdeg, pdeg, dim)
    cmd = [sys.executable, str(script), mode, str(folder), str(res), str(vdeg), str(pdeg), str(out),
           str(dim)]
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


def assert_exact(result, tol=1e-12, dofs=()):
    """Exact at the vertices, and at all dofs of the fields named in dofs."""
    for time, t_expected in (("None", 0.2), ("0.1", 0.1)):
        assert result[time]["t"] == pytest.approx(t_expected)
        for name, err in result[time]["errors"].items():
            assert err < tol, "{} at t = {}: max vertex error {:.2e}".format(name, time, err)
        for name in dofs:
            err = result[time]["dof_errors"][name]
            assert err < tol, "{} at t = {}: max dof error {:.2e}".format(name, time, err)


def test_read_back(tmp_path, series):
    """Last time and a chosen time, into the spaces that wrote them."""
    _, folder = series
    assert_exact(helper(tmp_path, "read", folder), dofs=["u0", "u1", "p", "phi", "g"])


def test_series_hold_the_physical_pressure(tmp_path, series):
    """The series hold the physical pressure p + phi*g; their params.dat marks it,
    and the reader gives back p. Without the mark the values are read as they
    are: here that is p + phi*g."""
    import pickle
    import shutil
    _, folder = series
    params = pickle.load(open(str(folder / "params.dat"), "rb"))
    assert params["timeseries_pressure"] == "physical"
    unmarked = tmp_path / "unmarked"
    shutil.copytree(str(folder), str(unmarked))
    params.pop("timeseries_pressure")
    pickle.dump(params, open(str(unmarked / "params.dat"), "wb"))
    result = helper(tmp_path, "read", unmarked)
    for time in ("None", "0.1"):
        assert result[time]["dof_errors"]["P"] < 1e-12
        assert result[time]["dof_errors"]["p"] > 1e-3


def test_read_into_other_degrees(tmp_path, series):
    """Into P1 velocity (from the P2 values at the vertices) and a P2 phase field
    (interpolated from the vertex values of the P1 one)."""
    _, folder = series
    assert_exact(helper(tmp_path, "read", folder, vdeg=1, pdeg=2), dofs=["u0", "u1", "p"])


def test_read_in_parallel(tmp_path, series):
    """Written in serial, read on two processes (and the other way round)."""
    require_mpi(2)
    _, folder = series
    dofs = ["u0", "u1", "p", "phi", "g"]
    assert_exact(helper(tmp_path, "read", folder, nprocs=2), dofs=dofs)
    helper(tmp_path, "write", tmp_path / "results_np2", nprocs=2)
    assert_exact(helper(tmp_path, "read", tmp_path / "results_np2" / "data" / "1" / "Timeseries"),
                 dofs=dofs)


def test_other_mesh_is_an_error(tmp_path, series):
    """A series from a different mesh is reported, not read into the wrong points."""
    _, folder = series
    result = helper(tmp_path, "read", folder, res=0.05)
    assert "not the one the field was written on" in result["error"]


@pytest.fixture(scope="module")
def series_3d(tmp_path_factory):
    """A 3D time series (XDMFFile2), with P2 velocity and a P1 phase field."""
    wd = tmp_path_factory.mktemp("series_3d")
    helper(wd, "write", wd / "results", res=0.25, dim=3)
    return wd, wd / "results" / "data" / "1" / "Timeseries"


def test_read_back_3d(tmp_path, series_3d):
    """The P2 velocity is read back at its midside nodes too, also on two processes."""
    _, folder = series_3d
    dofs = ["u0", "u1", "u2", "p", "phi", "g"]
    assert_exact(helper(tmp_path, "read", folder, res=0.25, dim=3), dofs=dofs)
    # Read into P1 velocity and a P2 phase field (interpolated from the vertices).
    assert_exact(helper(tmp_path, "read", folder, res=0.25, vdeg=1, pdeg=2, dim=3),
                 dofs=["u0", "u1", "u2", "p"])
    require_mpi(2)
    assert_exact(helper(tmp_path, "read", folder, nprocs=2, res=0.25, dim=3), dofs=dofs)
    helper(tmp_path, "write", tmp_path / "results_np2", nprocs=2, res=0.25, dim=3)
    assert_exact(helper(tmp_path, "read", tmp_path / "results_np2" / "data" / "1" / "Timeseries",
                        res=0.25, dim=3), dofs=dofs)
