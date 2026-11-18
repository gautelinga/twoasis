"""The DOLFINx backend against the dolfin backend: the BDF solver's operators and one time
step on the same mesh (tests/backend_worker.py, run once per backend), matched dof by
dof through their coordinates.

Operators and right-hand sides agree to round-off (they are the same integrals over the
same cells); the solution after the step to the Krylov solvers' tolerance (1e-12 here).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, once

pytest.importorskip("dolfinx")

WORKER = REPO_ROOT / "tests" / "backend_worker.py"
OPERATOR_TOL = 1e-12
SOLUTION_TOL = 1e-9

# Built before the first solve: the same integrals over the same cells, equal to
# round-off. name: (row space, column space) of a matrix, the space of a vector.
OPERATORS = {
    "Md": ("V", "V"), "divu_0": ("Q", "V"), "divu_1": ("Q", "V"), "gradp_0": ("V", "Q"),
    "gradp_M": ("V", "V"), "Mpf": ("W", "W"), "Mpf_form": ("W", "W"),
    "Apf_form": ("W", "W"), "Apf": ("W", "W"), "WGM": ("V", "Q"),
    "Fv": "V", "mass_Q_inv": "Q", "rho_": "V", "mu_": "V", "b_phig": "W"}
# Built from the phase field after its solve, and the solutions: equal to the
# solvers' tolerance.
SOLVED = {
    "A": ("V", "V"), "Ap": ("Q", "Q"), "rho_1": "V", "mu_1": "V", "b_tmp_u0": "V",
    "b_tmp_u1": "V", "b_u0": "V", "b_u1": "V", "b_p": "Q",
    "phig_1": "W", "x_u0": "V", "x_u1": "V", "x_p": "Q", "x_phig": "W"}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """{backend: (arrays, {space: permutation to the common order})}."""
    out = {}
    for backend in ("dolfin", "dolfinx"):
        def make(wd):
            env = dict(os.environ, TWOASIS_BACKEND=backend, PYTHONPATH=str(REPO_ROOT),
                       OMP_NUM_THREADS="1")
            proc = subprocess.run([sys.executable, str(WORKER), str(wd / "out.npz")], env=env,
                                  capture_output=True, text=True, timeout=900)
            assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
        data = dict(np.load(once(tmp_path_factory, "cross_" + backend, make) / "out.npz"))
        perms = {}
        for space in "VQW":
            k = np.round(data["keys_" + space], 9)
            perms[space] = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
            data["sorted_keys_" + space] = k[perms[space]]
        out[backend] = data, perms
    return out


def test_same_dofs(results):
    (d, _), (x, _) = results["dolfin"], results["dolfinx"]
    for space in "VQW":
        assert d["sorted_keys_" + space].shape == x["sorted_keys_" + space].shape, space
        assert np.array_equal(d["sorted_keys_" + space], x["sorted_keys_" + space]), space


def relative(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300)


def compare(results, name, spaces):
    (d, pd), (x, px) = results["dolfin"], results["dolfinx"]
    if isinstance(spaces, tuple):
        r, c = spaces
        return relative(d[name][np.ix_(pd[r], pd[c])], x[name][np.ix_(px[r], px[c])])
    return relative(d[name][pd[spaces]], x[name][px[spaces]])


@pytest.mark.parametrize("name", OPERATORS)
def test_operator(results, name):
    assert compare(results, name, OPERATORS[name]) < OPERATOR_TOL


@pytest.mark.parametrize("name", SOLVED)
def test_solved(results, name):
    assert compare(results, name, SOLVED[name]) < SOLUTION_TOL


def test_inner_passes(results):
    (d, _), (x, _) = results["dolfin"], results["dolfinx"]
    for k in (1, 2, 3):
        name = "udiff_{}".format(k)
        assert abs(d[name] - x[name]) <= 1e-8 * max(abs(d[name]), 1e-12), name
