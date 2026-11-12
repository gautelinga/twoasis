"""The legacy dolfin backend: the names twoasis's driver and solvers take from dolfin and
femtools, re-exported unchanged.

This list is the contract. A backend for another finite-element library provides the same
names with the same behaviour for what the driver (TPfracStep.py), the solvers
(solvers/TPfracStep) and the projections (common/utilities.py) use of them:

    UFL         ufl (the module), Coefficient, as_vector, split, dot, inner, grad,
                nabla_grad, div, lhs, rhs, solve (a linear variational problem), dx, Measure, FacetNormal, TrialFunction(s),
                TestFunction(s), FiniteElement, MixedElement
    spaces and functions
                FunctionSpace, Function, Constant, DirichletBC, Vector, Matrix,
                as_backend_type
    assembly    assemble, assemble_matrix (femtools.fast_assembly), apply_bc
                (femtools.fast_bc.apply), interpolate (femtools.fast_interpolation for
                a Function, dolfin's for an Expression),
                constant_nullspace, AssignedVectorFunction, set_subfunction, subfunction,
                subspace (femtools.mixed)
    algebra     norm, normalize, PETScKrylovSolver, ksp (femtools.ksp)
    runtime     MPI, parameters, Timer, list_timings, TimingClear, TimingType, XDMFFile,
                DOLFIN_EPS, DOLFIN_EPS_LARGE, BoundaryMarkers (femtools.boundaries),
                TimeSeriesFile (femtools.xdmf.XDMFFile2, set up for time series)
    checkpoints write_dof_tables, write_function_h5 (femtools.dolfin_layout: dolfin's
                HDF5 layout), space_id
    UFL, more   conditional, cos, exp, ge, gt, le, lt, sin, sqrt
    problems    Mesh, RectangleMesh, Point, Expression, SubDomain, SpatialCoordinate, Faces,
                read_mesh_file, write_mesh_h5, extent, mark_walls (meshes, boxes, walls),
                UnitSquareMesh, mesh_coordinates (writable), hmax
                (femtools.box), PeriodicBox (femtools.periodic.periodic_box_from_bounds)
    helpers     function_space, space_mesh, mesh_comm, geometric_dimension, element_degree
                (dolfin's methods that are DOLFINx properties), owned_dofs, fixed_dofs
                (index sets of a mixed space's components and of Dirichlet dofs),
                nodal_values (femtools.fast_interpolation.values), sync (no-op on dolfin)

Imports made inside functions for dolfin-only features (the periodic pairing check, the
weighted gradient matrix) stay where they are.
"""
from dolfin import (
    as_backend_type, as_vector, Constant, DirichletBC, div, DOLFIN_EPS, DOLFIN_EPS_LARGE, dot,
    dx, Expression, FacetNormal, FiniteElement, Function, FunctionSpace, grad, IndexMap, inner,
    lhs, list_timings, Matrix, Measure, Mesh, MixedElement, MPI, nabla_grad, norm, normalize,
    parameters, PETScKrylovSolver, Point, RectangleMesh, rhs, solve, SpatialCoordinate, split,
    UnitSquareMesh,
    TestFunction, TestFunctions, Timer, TimingClear, TimingType, TrialFunction, TrialFunctions,
    Vector, XDMFFile)

import numpy as np

from dolfin import conditional, cos, exp, ge, gt, le, lt, sin, sqrt, SubDomain

from femtools import ksp
from femtools.boundaries import BoundaryMarkers
from femtools.dolfin_layout import write_dofs as write_dof_tables
from femtools.dolfin_layout import write_function as write_function_h5
from femtools.box import Faces
from femtools.fast_assembly import assemble, assemble_matrix
from femtools.fast_bc import apply as apply_bc
from femtools.fast_interpolation import interpolate as _fast_interpolate
from femtools.fast_interpolation import values as nodal_values
from femtools.mixed import AssignedVectorFunction, set_subfunction, subfunction, subspace
from femtools.nullspace import constant_nullspace
# Periodic boundary conditions on a box, from each periodic axis's (lower, upper)
# bounds, e.g. PeriodicBox({0: (0, Lx)}) for periodicity in x.
from femtools.periodic import periodic_box_from_bounds as PeriodicBox

# The UFL that dolfin was built with: ufl_legacy in recent installations, ufl in older ones.
try:
    import ufl_legacy as ufl
    if not isinstance(Constant(0.), ufl.Coefficient):
        raise ImportError
except ImportError:
    import ufl
Coefficient = ufl.Coefficient

name = "dolfin"


# Neutral helpers: what the algorithm asks of spaces and meshes where dolfin and DOLFINx
# spell it differently (a method here, a property there), and what dolfin's own API
# reaches only through its dofmaps. The DOLFINx backend provides the same names.

def function_space(u):
    """The space of Function u."""
    return u.function_space()


def space_mesh(V):
    """The mesh of space V."""
    return V.mesh()


def mesh_comm(mesh):
    """The mesh's MPI communicator (mpi4py)."""
    return mesh.mpi_comm()


def geometric_dimension(mesh):
    return mesh.geometry().dim()


def element_degree(V):
    """The polynomial degree of V's element."""
    return V.ufl_element().degree()


def owned_dofs(W, i):
    """The owned dofs of component i of the mixed space W, as global indices of its
    vectors."""
    from petsc4py import PETSc
    return np.asarray(W.sub(i).dofmap().dofs(), dtype=PETSc.IntType)


def fixed_dofs(W, bcs):
    """The owned dofs of space W that the Dirichlet conditions `bcs` constrain, as
    global indices of its vectors."""
    start = W.dofmap().ownership_range()[0]
    owned = W.dofmap().index_map().size(IndexMap.MapSize.OWNED)
    return np.array([start + k for bc in bcs for k in bc.get_boundary_values() if k < owned],
                    dtype=np.int64)


def interpolate(f, V, out=None):
    """f interpolated into V, into `out` if given (a Function on V): a Function by
    femtools.fast_interpolation (a transfer matrix built once), anything else (an
    Expression) by dolfin."""
    if isinstance(f, Function):
        return _fast_interpolate(f, V, out=out)
    from dolfin import interpolate as dolfin_interpolate
    g = dolfin_interpolate(f, V)
    if out is None:
        return g
    out.vector().zero()
    out.vector().axpy(1., g.vector())
    return out


def TimeSeriesFile(comm, path):
    """A file for a field's time series, `write(function, t)` and `close()`: XDMF that
    keeps the midside nodes of a P2 field and compresses the data (femtools.xdmf's
    XDMFFile2), written incrementally."""
    from femtools.xdmf import XDMFFile2
    f = XDMFFile2(comm, path)
    f.parameters["rewrite_function_mesh"] = False
    f.parameters["flush_output"] = True
    return f


def space_id(V):
    """A key for space V, the same while V exists."""
    return V.id()


def read_mesh_file(fname, periodic=None):
    """Read the mesh stored as "mesh" in the HDF5 file fname.

    `periodic` gives the axes (x, y, z) the problem identifies. Under MPI the mesh
    is then read in blocks and partitioned with femtools.periodic_partition, so that
    no process holds a vertex of a periodic face without knowing that it is periodic
    (dolfin's periodic dofmap would give that vertex a wrong dof). Without it, the
    mesh is read by dolfin and post_import_problem repairs the partition, which
    costs another build of the mesh when cells have to move.
    """
    if periodic is not None and MPI.size(MPI.comm_world) > 1:
        from femtools.box import FACE_TOL
        from femtools.periodic_partition import read_mesh

        def info(s):
            if MPI.rank(MPI.comm_world) == 0:
                print(s)
        # L=None: the upper faces are at the largest coordinates, which is all the
        # partition needs.
        return read_mesh(fname, None, periodic, MPI.comm_world, log=info, tol=FACE_TOL)
    from dolfin import HDF5File
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh


def mesh_coordinates(mesh):
    """The mesh's vertex coordinates, writable (mesh.coordinates())."""
    return mesh.coordinates()


def hmax(mesh):
    """The largest cell diameter over all ranks (mesh.hmax())."""
    return mesh.hmax()


def write_mesh_h5(path, mesh):
    """The mesh in dolfin's HDF5 layout (femtools.dolfin_layout.write_mesh). Collective."""
    from femtools.dolfin_layout import write_mesh
    write_mesh(path, mesh)


def extent(mesh):
    """`(lower, upper)`: the mesh's bounding box over all ranks (femtools.periodic)."""
    from femtools.periodic import extent as femtools_extent
    return femtools_extent(mesh)


def mark_walls(boundaries, mesh, marker=1, **kwargs):
    """Mark the walls of the obstacles inside the mesh's box (femtools.box.mark_walls)."""
    from femtools.box import mark_walls as femtools_mark_walls
    return femtools_mark_walls(boundaries, mesh, marker, **kwargs)


def read_function_h5(path, u, name):
    """u from the Function `name` in path (dolfin's HDF5 layout, any number of
    processes)."""
    from dolfin import HDF5File
    with HDF5File(u.function_space().mesh().mpi_comm(), path, "r") as h5f:
        h5f.read(u, name)
    return u


def read_vector_function_h5(path, name, components):
    """The vector Function `name` in path (dolfin's HDF5 layout) into the scalar
    Functions `components` (all on one space, which gives the vector space its element
    and periodicity)."""
    from dolfin import FunctionAssigner, HDF5File, VectorFunctionSpace
    V = components[0].function_space()
    mesh = V.mesh()
    Vv = VectorFunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree(),
                             constrained_domain=V.dofmap().constrained_domain)
    u = Function(Vv)
    with HDF5File(mesh.mpi_comm(), path, "r") as h5f:
        h5f.read(u, name)
    FunctionAssigner([c.function_space() for c in components], Vv).assign(list(components), u)


def read_series_step(path, name, counter, u):
    """Step `counter` of field `name` of an XDMF time series (femtools.xdmf.XDMFFile2's
    layout, either backend's) into u (XDMFFile2.read_checkpoint)."""
    from femtools.xdmf import XDMFFile2
    with XDMFFile2(u.function_space().mesh().mpi_comm(), path) as xdmf:
        xdmf.read_checkpoint(u, name, counter)
    return u


def read_vector_series_step(path, name, counter, components, constrained_domain):
    """Step `counter` of the vector field `name` of an XDMF time series into the scalar
    Functions `components` (on one space, its element and periodicity the vector's)."""
    from dolfin import FunctionAssigner, VectorFunctionSpace
    V = components[0].function_space()
    el = V.ufl_element()
    uv = Function(VectorFunctionSpace(V.mesh(), el.family(), el.degree(),
                                      constrained_domain=constrained_domain))
    read_series_step(path, name, counter, uv)
    FunctionAssigner([V] * len(components), uv.function_space()).assign(list(components), uv)


def weighted_gradient_matrix(V, S, i):
    """The matrix of the derivative along x_i of a Function on S, onto V: the cells'
    mean gradients averaged onto V's dofs (femtools.gradient, compiled)."""
    from femtools.gradient import compiled_gradient_module
    DG = FunctionSpace(V.mesh(), 'DG', 0)
    G = assemble(TrialFunction(DG) * TestFunction(V) * dx())
    dg = Function(DG)
    dP = assemble(TrialFunction(S).dx(i) * TestFunction(DG) * dx())
    return compiled_gradient_module.compute_weighted_gradient_matrix(G, dP, dg)


def sync(f):
    """Nothing to do on dolfin, whose Functions have one layout; see the DOLFINx
    backend."""
    return f

__all__ = [
    "ufl", "Coefficient", "as_vector", "split", "dot", "inner", "grad", "nabla_grad", "div",
    "lhs", "rhs", "solve", "dx", "Measure", "FacetNormal", "TrialFunction", "TrialFunctions",
    "TestFunction", "TestFunctions", "FiniteElement", "MixedElement",
    "FunctionSpace", "Function", "Constant", "DirichletBC", "Vector", "Matrix",
    "as_backend_type",
    "assemble", "assemble_matrix", "apply_bc", "interpolate", "constant_nullspace",
    "AssignedVectorFunction", "set_subfunction", "subfunction", "subspace",
    "norm", "normalize", "PETScKrylovSolver", "ksp",
    "MPI", "parameters", "Timer", "list_timings", "TimingClear", "TimingType", "XDMFFile",
    "DOLFIN_EPS", "DOLFIN_EPS_LARGE", "BoundaryMarkers", "name",
    "function_space", "space_mesh", "mesh_comm", "geometric_dimension", "element_degree",
    "owned_dofs", "fixed_dofs", "nodal_values", "sync", "TimeSeriesFile",
    "write_dof_tables", "write_function_h5", "space_id", "read_function_h5",
    "read_vector_function_h5", "read_series_step", "read_vector_series_step",
    "weighted_gradient_matrix",
    "read_mesh_file", "write_mesh_h5", "extent", "mark_walls", "UnitSquareMesh",
    "mesh_coordinates", "hmax",
    "Mesh", "RectangleMesh", "Point", "Expression", "SpatialCoordinate", "Faces", "PeriodicBox",
    "SubDomain", "conditional", "cos", "exp", "ge", "gt", "le", "lt", "sin", "sqrt",
]
