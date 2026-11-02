"""Open boundaries: SimplePore2D, a meniscus driven through a pore throat by
pressure and phase-field Dirichlet conditions at the inlet and outlet.

The mesh is generated with meshes/lattice_mesh.py (needs gmsh and meshio; the
tests are skipped without them).
"""
import numpy as np

from conftest import COLS, assert_close, golden_compare


def test_runs(pore_bdf, pore_ipcs):
    for _, td in (pore_bdf, pore_ipcs):
        assert td.shape[0] == 5
        assert np.isfinite(td).all()
        assert td[-1, COLS["E_kin"]] < 1.0


def test_golden(request, pore_bdf):
    _, td = pore_bdf
    values = {name: float(td[-1, COLS[name]]) for name in ["u0m", "phim", "E_kin", "E_int"]}
    golden_compare(request, "pore_bdf", values)


def test_bdf_ipcs_agree(pore_bdf, pore_ipcs):
    """SimplePore2D uses BDF1 and equal fluid properties. For that case BDF and
    IPCS assemble the same equations, so they must agree to solver tolerance.
    A difference means one of them treats the open boundaries differently."""
    _, td_b = pore_bdf
    _, td_i = pore_ipcs
    for name in ["u0m", "phim", "E_kin", "E_int"]:
        assert_close("IPCS " + name, td_i[-1, COLS[name]], td_b[-1, COLS[name]], 1e-6, 1e-10)
