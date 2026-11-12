"""Worker for tests/test_backend_cross.py: the BDF solver's operators and one time step on
the backend TWOASIS_BACKEND names, saved by dof coordinates.

    TWOASIS_BACKEND=dolfinx python3 tests/backend_worker.py out.npz

A bubble in a box periodic in x with walls at y = 0 and y = Ly, built as the driver
(TPfracStep.py) builds its namespace, with unequal densities and viscosities, a contact
angle, a nonzero wall velocity and gravity, so that every term of the scheme is there.
The solver's own functions (setup, the assembly and solve steps) run on it. Saved: every
operator and right-hand side as a dense array in the order of the backend's vectors,
the solution after the step, and for each space the key of each vector entry
(component, x, y), from which the test matches the two backends' dofs. Serial.
"""
import math
import sys

import numpy as np

import twoasis.backend as be
from twoasis.backend import (
    as_vector, assemble, Constant, DirichletBC, dx, FacetNormal, FiniteElement, Function,
    FunctionSpace, Measure, MixedElement, parameters, split, TestFunction, TestFunctions,
    TrialFunction, TrialFunctions, Vector)

LX, LY, NX, NY = 1.0, 2.0, 6, 12
VELOCITY_DEGREE, PRESSURE_DEGREE = 2, 1


def mesh_and_markers():
    """The rectangle mesh (the same cells on both backends: right diagonals), its
    periodic box, and facet markers with the walls marked 2."""
    if be.name == "dolfin":
        import dolfin as df
        from femtools.box import Faces
        from femtools.periodic import periodic_box_from_bounds
        mesh = df.RectangleMesh(df.Point(0., 0.), df.Point(LX, LY), NX, NY, "right")
        box = periodic_box_from_bounds({0: (0., LX)})
        markers = be.BoundaryMarkers(mesh)
        markers.set_all(0)
        Faces(1, 0., LY).mark(markers, 2)
        return mesh, box, markers
    import dolfinx.mesh
    mesh = dolfinx.mesh.create_rectangle(be.MPI.comm_world, [[0., 0.], [LX, LY]], [NX, NY],
                                         dolfinx.mesh.CellType.triangle,
                                         diagonal=dolfinx.mesh.DiagonalType.right)
    box = be.PeriodicBox({0: (0., LX)})
    markers = be.BoundaryMarkers(mesh)
    markers.set_all(0)
    markers.mark(lambda x: np.isclose(x[1], 0.) | np.isclose(x[1], LY), 2)
    return mesh, box, markers


def keys(V, n_sub):
    """(component, x, y) of each entry of V's vectors, in their order."""
    if be.name == "dolfin":
        X = V.tabulate_dof_coordinates().reshape((-1, 2))
        comp = np.zeros(len(X))
        for i in range(n_sub):
            comp[np.asarray(V.sub(i).dofmap().dofs())] = i
        return np.column_stack([comp, wrapped(X[:, :2])])
    from femtools.dolfinx.periodic import flat_collapse_map
    from twoasis.backend.dolfinx import unique_dofs
    U = unique_dofs(V)
    if n_sub == 0:
        X = V.tabulate_dof_coordinates()[:, :2]
        comp = np.zeros(len(X))
    else:
        n_local = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
        X, comp = np.zeros((n_local, 2)), np.zeros(n_local)
        for i in range(n_sub):
            Vi, parent = V.sub(i).collapse()
            parent = flat_collapse_map(parent)
            X[parent] = Vi.tabulate_dof_coordinates()[:, :2]
            comp[parent] = i
    own = U.owned_unique
    return np.column_stack([comp[own], wrapped(X[own])])


def wrapped(X):
    """Coordinates with x on the periodic face x = LX taken to x = 0: dolfin reports a
    periodic dof where its slave is, DOLFINx where its master is."""
    X = X.copy()
    X[np.isclose(X[:, 0], LX), 0] = 0.
    return X


def set_values(f, n_sub, *funcs):
    """f's values: funcs[i](x, y) at the dofs of component i."""
    k = keys(be.function_space(f), n_sub)
    values = np.zeros(len(k))
    for i, func in enumerate(funcs):
        mine = k[:, 0] == i
        values[mine] = func(k[mine, 1], k[mine, 2])
    f.vector().set_local(values)
    f.vector().apply("insert")


def dense(A):
    """A as a dense array (a converted copy: convert() alone works in place)."""
    from petsc4py import PETSc
    D = PETSc.Mat()
    be.as_backend_type(A).mat().convert("dense", D)
    return D.getDenseArray().copy()


def array(x):
    return x.get_local().copy()


def main(out):
    from twoasis.problems.TPfracStep import NS_parameters
    import twoasis.solvers.TPfracStep as shared
    from twoasis.common import utilities
    from twoasis.solvers.TPfracStep import BDF

    parameters["form_compiler"]["quadrature_degree"] = 2 * VELOCITY_DEGREE + 1
    mesh, box, subdomains = mesh_and_markers()

    ns = dict(NS_parameters)
    ns.update(rho=[1000., 100.], mu=[10., 1.], sigma=24.5, epsilon=0.08, M=1e-4, dt=0.002,
              bdf_order=2, rotational=True, constant_poisson_coefficient=False,
              initial_step_1st_order=False, tstep=0, t=0.0, AB_projection_pressure=False,
              scalar_components=[], u_components=["u0", "u1"], anderson_depth=0,
              velocity_tolerance_factor=0)
    ns["krylov_solvers"] = dict(ns["krylov_solvers"], relative_tolerance=1e-12,
                                absolute_tolerance=1e-14)
    V = Q = FunctionSpace(mesh, "CG", VELOCITY_DEGREE, constrained_domain=box)
    Q = FunctionSpace(mesh, "CG", PRESSURE_DEGREE, constrained_domain=box)
    el = FiniteElement("CG", mesh.ufl_cell(), VELOCITY_DEGREE)
    W = FunctionSpace(mesh, MixedElement([el, el]), constrained_domain=box)
    ds = Measure("ds", domain=mesh, subdomain_data=subdomains)
    u, v, p, q = TrialFunction(V), TestFunction(V), TrialFunction(Q), TestFunction(Q)
    phi, g = TrialFunctions(W)
    xi, eta = TestFunctions(W)
    sys_comp = ["u0", "u1", "p", "phig"]
    VV = dict(u0=V, u1=V, p=Q, phig=W)
    q_ = {ui: Function(VV[ui], name=ui) for ui in sys_comp}
    q_1 = {ui: Function(VV[ui], name=ui + "_1") for ui in sys_comp}
    q_2 = {ui: Function(VV[ui], name=ui + "_2") for ui in sys_comp}
    u_ = as_vector([q_[ui] for ui in ns["u_components"]])
    u_1 = as_vector([q_1[ui] for ui in ns["u_components"]])
    x_ = {ui: q_[ui].vector() for ui in sys_comp}
    x_1 = {ui: q_1[ui].vector() for ui in sys_comp}
    x_2 = {ui: q_2[ui].vector() for ui in sys_comp}
    phi_, g_ = split(q_["phig"])
    phi_1, g_1 = split(q_1["phig"])

    # Fields at three levels: a bubble, a shear flow, a pressure
    def bubble(s):
        def phi0(x, y):
            return np.tanh((np.hypot(x - 0.5 * LX, y - 0.4 * LY) - 0.3 * s) / (0.08 * np.sqrt(2)))
        return phi0

    def chem(x, y):
        return 0.3 * np.sin(2 * np.pi * x) * np.cos(np.pi * y / LY)

    for k, (f, s) in enumerate(((q_, 1.0), (q_1, 0.98), (q_2, 0.96))):
        set_values(f["u0"], 0, lambda x, y, s=s: s * (0.5 + y * (LY - y) * np.sin(2 * np.pi * x)))
        set_values(f["u1"], 0, lambda x, y, s=s: s * 0.3 * np.cos(2 * np.pi * x) * y * (LY - y))
        set_values(f["p"], 0, lambda x, y, s=s: s * (np.cos(2 * np.pi * x) + y))
        set_values(f["phig"], 2, bubble(s), lambda x, y, s=s: s * chem(x, y))

    wall = DirichletBC(V, 0.5, subdomains, 2)
    bcs = dict(u0=[wall], u1=[DirichletBC(V, 0., subdomains, 2)], p=[], phig=[])
    for ui in ("u0", "u1"):
        for x in (x_[ui], x_1[ui], x_2[ui]):
            [bc.apply(x) for bc in bcs[ui]]

    ns.update(
        mesh=mesh, V=V, Q=Q, W=W, VV=VV, u=u, v=v, p=p, q=q, phi=phi, g=g, xi=xi, eta=eta,
        q_=q_, q_1=q_1, q_2=q_2, u_=u_, u_1=u_1, x_=x_, x_1=x_1, x_2=x_2, p_=q_["p"],
        phi_=phi_, phi_1=phi_1, g_=g_, g_1=g_1, bcs=bcs, ds=ds, normal=FacetNormal(mesh),
        angles=[(math.pi / 3, 2)], sys_comp=sys_comp, subdomains=subdomains,
        acc=Constant((0., -0.98)), gradp_avg=Constant((-1.5, 0.)),
        b={ui: Vector(x_[ui]) for ui in sys_comp},
        b_tmp={ui: Vector(x_[ui]) for ui in sys_comp},
        du_={ui: x_[ui].copy() for ui in ns["u_components"]},
        dp_=Function(Q), fpf=None, bpf=None, NS_parameters=dict(ns),
        udiff=np.array([1e8]), unorm=np.array([0.]), inner_iter=0,
        assemble_matrix=utilities.assemble_matrix, homogenize=utilities.homogenize,
        GradFunction=utilities.GradFunction, PhiGradRHS=utilities.PhiGradRHS,
        DivRHS=utilities.DivRHS, TPsource=utilities.TPsource)
    v_ = ns["v"]
    ns["bgp0"] = {ui: v_ * ns["gradp_avg"][i] for i, ui in enumerate(ns["u_components"])}
    ns["b0"] = {ui: assemble(ns["bgp0"][ui] * dx) for ui in ns["u_components"]}
    ns["bg0"] = {ui: v_ * ns["acc"][i] for i, ui in enumerate(ns["u_components"])}

    res = dict(keys_V=keys(V, 0), keys_Q=keys(Q, 0), keys_W=keys(W, 2))
    ns.update(BDF.setup(**ns))
    ns["u_sol"], ns["p_sol"], ns["pf_sol"], ns["c_sol"] = BDF.get_solvers(**ns)
    res.update(Md=dense(ns["Md"]), Fv=array(ns["Fv"]), mass_Q_inv=array(ns["mass_Q_inv"]),
               rho_=array(ns["rho_"].vector()), mu_=array(ns["mu_"].vector()),
               divu_0=dense(ns["divu"].matvec[0][0]), divu_1=dense(ns["divu"].matvec[1][0]),
               gradp_0=dense(ns["gradp"]["u0"].matvec[0]),
               gradp_M=dense(ns["gradp"]["u0"].A),
               Mpf_0=dense(ns["Mpf"][0]), Mpf_1=dense(ns["Mpf"][1]),
               Apf_const=dense(ns["Apf_const"]),
               WGM=dense(be.weighted_gradient_matrix(V, Q, 0)))

    # One time step, as the driver takes it, with three inner passes.
    ns["t"] += ns["dt"]
    ns["tstep"] += 1
    BDF.phase_field_assemble(**ns)
    res.update(Apf=dense(ns["Apf"]), b_phig=array(ns["b"]["phig"]))
    shared.phase_field_solve(**ns)
    res.update(phig_1=array(x_["phig"]))
    for inner in range(1, 4):
        ns["inner_iter"] = inner
        ns["udiff"][0] = ns["unorm"][0] = 0.
        if inner == 1:
            BDF.assemble_first_inner_iter(**ns)
            res.update(A=dense(ns["A"]), Ap=dense(ns["Ap"]), rho_1=array(ns["rho_"].vector()),
                       mu_1=array(ns["mu_"].vector()),
                       b_tmp_u0=array(ns["b_tmp"]["u0"]), b_tmp_u1=array(ns["b_tmp"]["u1"]))
        for ui in ns["u_components"]:
            BDF.velocity_tentative_assemble(ui=ui, **ns)
            if inner == 1:
                res["b_" + ui] = array(ns["b"][ui])
            shared.velocity_tentative_solve(ui=ui, **ns)
        BDF.pressure_assemble(**ns)
        if inner == 1:
            res["b_p"] = array(ns["b"]["p"])
        BDF.pressure_solve(**ns)
        BDF.pressure_update(**ns)
        res["udiff_{}".format(inner)] = ns["udiff"][0]
    BDF.velocity_update(**ns)
    res.update({"x_" + ui: array(x_[ui]) for ui in sys_comp})
    np.savez(out, **res)


if __name__ == "__main__":
    main(sys.argv[1])
