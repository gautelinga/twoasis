"""Open boundaries: SimplePore2D, a meniscus driven through a pore throat by
pressure and phase-field Dirichlet conditions at the inlet and outlet.

The mesh is generated with meshes/lattice_mesh.py (needs gmsh and meshio; the
tests are skipped without them).
"""
import numpy as np

from conftest import COLS, assert_last_close, golden_compare


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
    assert_last_close(td_i, td_b, [(name, 1e-6) for name in ["u0m", "phim", "E_kin", "E_int"]],
                      "IPCS")


def test_physical_pressure_at_the_ends(pore_bdf, pore_ipcs):
    """The ends prescribe the physical pressure p + phi*g = 0 (PhysicalPressureBC,
    followed every step): the last timestamp, which holds p + phi*g, is zero at
    the ends' vertices."""
    import dolfin
    from femtools.boundaries import read
    for wd, _ in (pore_bdf, pore_ipcs):
        case = wd / "results" / "data" / "1" / "Timestamps"
        mesh = dolfin.Mesh()
        with dolfin.HDF5File(mesh.mpi_comm(), str(case / "mesh.h5"), "r") as h5:
            h5.read(mesh, "mesh", False)
        P = dolfin.Function(dolfin.FunctionSpace(mesh, "P", 1))
        with dolfin.HDF5File(mesh.mpi_comm(), str(case / "up_5.h5"), "r") as h5:
            h5.read(P, "p")
        points, facets, markers = read(str(case / "boundaries.xdmf"))
        ends = np.unique(facets[np.isin(markers, (2, 3))])
        assert len(ends) > 0
        assert max(abs(P(*points[i][:2])) for i in ends) < 1e-10
        assert np.abs(P.vector().get_local()).max() > 1e-2
