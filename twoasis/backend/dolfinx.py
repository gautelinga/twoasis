"""The DOLFINx backend: the names of the contract (twoasis/backend/dolfin.py) on DOLFINx.

The algorithm keeps the dolfin object API it is written in; here that API is
implemented on DOLFINx and femtools.dolfinx. Every vector the algorithm sees, a
Function's `vector()` included, is a PETSc vector over the space's *unique* dofs
(femtools.dolfinx.unique): periodic slaves are folded onto their masters, as dolfin's
`constrained_domain` does, so norms, means, pointwise products and reciprocals need no
masks, and matrices are assembled folded (N_unique x N_unique).

A Function is a DOLFINx Function whose values live in two layouts: the compact vector,
which is what the algorithm reads and writes, and DOLFINx's own array (owned, ghost
and slave dofs), which the form assembly and the interpolation read. The compact
vector is the truth: `sync()` copies it into DOLFINx's array (one forward scatter),
and the backend calls it on every coefficient of a form before assembling it.
Anything that writes DOLFINx's array directly (interpolation, a component's values)
calls `restrict()` afterwards, which copies the unique values back.

Where dolfin's method names clash with DOLFINx's properties (`V.mesh()` against
`V.mesh`, `u.function_space()` against `u.function_space`, `mesh.geometry().dim()`
against `mesh.geometry.dim`), the algorithm calls the neutral helpers that both
backends provide: `space_mesh`, `function_space`, `geometric_dimension`,
`element_degree`, `owned_dofs`, `fixed_dofs`.

Supported so far: scalar and mixed Lagrange spaces (and DG), periodic boxes given as
a box spec (`lower`, `upper`, `periodic`), Dirichlet conditions on facets marked in a
`BoundaryMarkers` with a number, a `Constant` or a Function on the space as value.
"""
import time
from collections import defaultdict
from enum import Enum

import numpy as np
from mpi4py import MPI as _MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc  # noqa: F401  (fem.petsc)
import ufl
from dolfinx import fem
from ufl import (  # noqa: F401
    as_vector, conditional, cos, div, dot, dx, exp, FacetNormal, ge, grad, gt, inner, le, lhs,
    lt, nabla_grad, rhs, sin, split, sqrt, TestFunction, TestFunctions, TrialFunction,
    TrialFunctions)

from femtools import ksp  # noqa: F401
from femtools.dolfinx import assembly as _assembly
from femtools.dolfinx.periodic import flat_collapse_map, periodic_box
from femtools.dolfinx.unique import UniqueDofs
from femtools.lru import LRU

name = "dolfinx"
# What dolfin calls a Coefficient: a Function or a Constant (a ufl.Constant in DOLFINx),
# for isinstance checks.
Coefficient = (ufl.Coefficient, ufl.Constant)
DOLFIN_EPS = 3.0e-16
DOLFIN_EPS_LARGE = 1.0e-14

# dolfin's global parameters, as far as twoasis sets them. quadrature_degree is put on
# every integral that does not set its own, as dolfin does with the global one.
parameters = {
    "linear_algebra_backend": "PETSc",
    "form_compiler": {"quadrature_degree": None, "optimize": True, "cpp_optimize": True,
                      "representation": "uflacs", "cpp_optimize_flags": "-O3"},
}


# --- MPI -------------------------------------------------------------------------------

class MPI:
    """dolfin's MPI helpers on mpi4py communicators."""
    comm_world = _MPI.COMM_WORLD
    comm_self = _MPI.COMM_SELF

    @staticmethod
    def rank(comm):
        return comm.rank

    @staticmethod
    def size(comm):
        return comm.size

    @staticmethod
    def barrier(comm):
        comm.Barrier()

    @staticmethod
    def sum(comm, value):
        return comm.allreduce(value, op=_MPI.SUM)

    @staticmethod
    def min(comm, value):
        return comm.allreduce(value, op=_MPI.MIN)

    @staticmethod
    def max(comm, value):
        return comm.allreduce(value, op=_MPI.MAX)

    @staticmethod
    def avg(comm, value):
        return comm.allreduce(value, op=_MPI.SUM) / comm.size


# --- timers ----------------------------------------------------------------------------

class TimingClear(Enum):
    keep = 0
    clear = 1


class TimingType(Enum):
    wall = 0
    user = 1
    system = 2


_TIMINGS = defaultdict(lambda: [0, 0.0])  # task -> [repetitions, total wall time]


class Timer:
    """dolfin's Timer: started when made (with a task name, it is recorded for
    list_timings when stopped or destroyed); stop() returns the elapsed wall time in seconds,
    elapsed() the (wall, user, system) times so far."""

    def __init__(self, task=None):
        self.task = task
        self._total = 0.0
        self._t0 = None
        self.start()

    def start(self):
        self._total = 0.0
        self._t0 = time.perf_counter()

    def resume(self):
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is not None:
            self._total += time.perf_counter() - self._t0
            self._t0 = None
            if self.task:
                entry = _TIMINGS[self.task]
                entry[0] += 1
                entry[1] += self._total
        return self._total

    def __del__(self):
        # dolfin records a running Timer when it is destroyed, too.
        try:
            self.stop()
        except Exception:
            pass

    def elapsed(self):
        running = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        wall = self._total + running
        return wall, wall, 0.0


def timings():
    """{task: (repetitions, total wall time)} on this process."""
    return {task: tuple(v) for task, v in _TIMINGS.items()}


def list_timings(clear=TimingClear.keep, types=(TimingType.wall,)):
    """Print the wall times of the named Timers (average over the processes), as
    dolfin's list_timings does; clear them with TimingClear.clear. Collective."""
    comm = MPI.comm_world
    tasks = sorted(set().union(*comm.allgather(list(_TIMINGS))))
    rows = []
    for task in tasks:
        reps, total = _TIMINGS[task] if task in _TIMINGS else (0, 0.0)
        reps = comm.allreduce(reps, op=_MPI.MAX)
        total = comm.allreduce(total, op=_MPI.SUM) / comm.size
        rows.append((task, reps, total / max(reps, 1), total))
    if comm.rank == 0 and rows:
        width = max(len(r[0]) for r in rows)
        print("[MPI_AVG] Summary of timings {} | reps    wall avg    wall tot".format(
            " " * max(width - 28, 0)))
        print("-" * (width + 36))
        for task, reps, avg, total in rows:
            print("{:<{w}} | {:6d}  {:10.6f}  {:10.6f}".format(task, reps, avg, total, w=width))
    if clear == TimingClear.clear:
        _TIMINGS.clear()


# --- elements, meshes and spaces -------------------------------------------------------

_FAMILIES = {"CG": "Lagrange", "P": "Lagrange", "Lagrange": "Lagrange", "Q": "Lagrange",
             "DG": "DG", "Discontinuous Lagrange": "DG"}


def FiniteElement(family, cell, degree):
    """A scalar element by dolfin's family name ('CG', 'DG', ...)."""
    family = _FAMILIES.get(family, family)
    cellname = cell if isinstance(cell, str) else cell.cellname  # a property in UFL 2026
    cellname = cellname() if callable(cellname) else cellname
    if family == "DG":
        return basix.ufl.element("Lagrange", cellname, degree, discontinuous=True)
    return basix.ufl.element(family, cellname, degree)


def MixedElement(elements):
    return basix.ufl.mixed_element(list(elements))


class BoxSpec:
    """A periodic box: `lower` and `upper` corners and the `periodic` axes (3 each),
    with `tol` the absolute distance from a face that counts as on it. The dolfinx
    counterpart of femtools.periodic.periodic_box_from_bounds, which dolfin problems
    use; any object with these attributes serves."""

    def __init__(self, lower, upper, periodic, tol=1e-10):
        self.lower = [float(v) for v in lower]
        self.upper = [float(v) for v in upper]
        self.periodic = [bool(p) for p in periodic]
        self.tol = float(tol)


def PeriodicBox(bounds, tol=1e-10):
    """The box periodic along each axis in `bounds`, which maps the axis to its
    (lower, upper) bounds: {0: (0, Lx)} is periodic in x alone."""
    lower, upper = ([bounds[i][k] if i in bounds else 0. for i in range(3)] for k in (0, 1))
    return BoxSpec(lower, upper, [i in bounds for i in range(3)], tol)


Mesh = dolfinx.mesh.Mesh
SpatialCoordinate = ufl.SpatialCoordinate


class Point:
    """dolfin's Point(x, y, z=0)."""

    def __init__(self, *x):
        self.x = [float(v) for v in x] + [0.0] * (3 - len(x))

    def __getitem__(self, i):
        return self.x[i]


_DIAGONALS = {"right": "right", "left": "left", "crossed": "crossed",
              "right/left": "right_left", "left/right": "left_right"}


def RectangleMesh(p0, p1, nx, ny, diagonal="right"):
    """dolfin's RectangleMesh(Point, Point, nx, ny, diagonal): the same vertices and
    cells, on MPI.comm_world."""
    kind = getattr(dolfinx.mesh.DiagonalType, _DIAGONALS[diagonal])
    return dolfinx.mesh.create_rectangle(MPI.comm_world, [[p0[0], p0[1]], [p1[0], p1[1]]],
                                         [int(nx), int(ny)], dolfinx.mesh.CellType.triangle,
                                         diagonal=kind)


def read_mesh_file(fname, periodic=None):
    """The mesh stored as "mesh" in the HDF5 file fname (dolfin's layout), numbered as
    dolfin numbers it (femtools.dolfinx.mesh.read_mesh). `periodic` is dolfin's concern
    (its partition); DOLFINx pairs periodic dofs on any."""
    from femtools.dolfinx.mesh import read_mesh
    mesh = read_mesh(fname, MPI.comm_world)
    _DEFAULT_MESH[0] = mesh
    return mesh


def BoxMesh(p0, p1, nx, ny, nz):
    """dolfin's BoxMesh(Point, Point, nx, ny, nz): the same vertices and tetrahedra
    (DOLFINx's create_box numbers them as dolfin does), on MPI.comm_world."""
    return dolfinx.mesh.create_box(MPI.comm_world, [[p0[0], p0[1], p0[2]],
                                                     [p1[0], p1[1], p1[2]]],
                                   [int(nx), int(ny), int(nz)],
                                   dolfinx.mesh.CellType.tetrahedron)


def UnitSquareMesh(nx, ny, diagonal="right"):
    """dolfin's UnitSquareMesh(nx, ny, diagonal) (RectangleMesh of the unit square)."""
    return RectangleMesh(Point(0., 0.), Point(1., 1.), nx, ny, diagonal)


def mesh_coordinates(mesh):
    """The mesh's vertex coordinates, a writable (n, gdim) view (dolfin's
    mesh.coordinates()): change them before making spaces on the mesh."""
    return mesh.geometry.x[:, :mesh.geometry.dim]


def hmax(mesh):
    """The largest cell diameter over all ranks (dolfin's mesh.hmax())."""
    tdim = mesh.topology.dim
    n = mesh.topology.index_map(tdim).size_local
    h = mesh.h(tdim, np.arange(n, dtype=np.int32))  # a method of Mesh in DOLFINx 0.11
    return MPI.max(mesh.comm, float(h.max()) if n else 0.0)


def write_mesh_h5(path, mesh):
    """The mesh in dolfin's HDF5 layout (femtools.dolfinx.mesh.write_mesh). Collective."""
    from femtools.dolfinx.mesh import write_mesh
    write_mesh(path, mesh)


def extent(mesh):
    """`(lower, upper)`: the mesh's bounding box over all ranks, gdim components each."""
    from femtools.dolfinx.periodic import box_extent
    lo, hi = box_extent(mesh)
    return lo[:mesh.geometry.dim], hi[:mesh.geometry.dim]


def mark_walls(boundaries, mesh, marker=1, tol=1e-10):
    """Mark the walls of the obstacles inside the mesh's bounding box `marker`: every
    boundary facet except those on the box's faces, which are set to 0 (femtools.box's
    mark_walls)."""
    lo, hi = extent(mesh)

    def on_box(x):
        on = np.zeros(x.shape[1], dtype=bool)
        for i in range(len(lo)):
            on |= (np.abs(x[i] - lo[i]) < tol) | (np.abs(x[i] - hi[i]) < tol)
        return on

    boundaries.mark(lambda x: np.ones(x.shape[1], dtype=bool), marker)
    boundaries.mark(on_box, 0)


class Faces:
    """The boundary where coordinate `axis` equals one of `values` (femtools.box.Faces):
    Faces(1, 0, Ly).mark(markers, 2) marks the faces y = 0 and y = Ly."""

    def __init__(self, axis, *values, tol=1e-10):
        self.axis, self.values, self.tol = axis, values, tol

    def inside_points(self, x):
        on = np.zeros(x.shape[1], dtype=bool)
        for v in self.values:
            on |= np.abs(x[self.axis] - v) < self.tol
        return on

    def mark(self, markers, value):
        markers.mark(self, value)


class SubDomain:
    """dolfin's SubDomain for marking facets: a subclass defines inside(x, on_boundary)
    for one point x, and mark(markers, value) marks the boundary facets all of whose
    vertices are inside (with on_boundary True). Evaluated point by point in Python,
    which is slow on large meshes; Faces, or a vectorized predicate given to
    BoundaryMarkers.mark, is fast."""

    def inside(self, x, on_boundary):
        raise NotImplementedError

    def inside_points(self, x):
        return np.array([bool(self.inside(x[:, k], True)) for k in range(x.shape[1])],
                        dtype=bool)

    def mark(self, markers, value):
        markers.mark(self, value)


_DEFAULT_MESH = [None]


class _Space(fem.FunctionSpace):
    """A DOLFINx FunctionSpace that knows its periodic box (`_twoasis_box`) and, made by
    `sub`, the space it is a component of (`_twoasis_parent`): a Dirichlet condition on
    W.sub(0) acts through W's unique dofs."""

    def sub(self, i):
        S = super().sub(i)
        S = _Space(S.mesh, S.ufl_element(), S._cpp_object)
        S._twoasis_box = self._twoasis_box
        S._twoasis_parent = self
        return S


def FunctionSpace(mesh, family, degree=None, constrained_domain=None):
    """A space as dolfin makes one: FunctionSpace(mesh, 'CG', 2) or
    FunctionSpace(mesh, element), periodic when constrained_domain is a box spec."""
    element = FiniteElement(family, mesh.ufl_cell(), degree) if degree is not None else family
    V = fem.functionspace(mesh, element)
    V = _Space(V.mesh, V.ufl_element(), V._cpp_object)
    V._twoasis_box = constrained_domain
    _DEFAULT_MESH[0] = mesh
    return V


def unique_dofs(V):
    """The UniqueDofs of a space made by FunctionSpace or subspace (built once)."""
    U = getattr(V, "_twoasis_unique", None)
    if U is None:
        if hasattr(V, "_twoasis_parent"):
            raise ValueError("unique_dofs: a component W.sub(i) has W's unique dofs")
        if not hasattr(V, "_twoasis_box"):
            raise ValueError("a space not made by twoasis.backend (FunctionSpace or "
                             "subspace): its periodicity is unknown")
        box = V._twoasis_box
        # Mixed spaces numbered node by node, as dolfin numbers them (ILU depends on it).
        if box is None or not any(box.periodic):
            U = UniqueDofs(V, interleave=True)
        else:
            mesh = V.mesh
            gdim = mesh.geometry.dim
            extent = max(u - lo for u, lo in zip(box.upper[:gdim], box.lower[:gdim]))
            tol = max(float(getattr(box, "tol", 1e-10)) / extent, 1e-12)
            U = UniqueDofs(V, *periodic_box(V, upper=box.upper[:gdim],
                                            periodic=box.periodic[:gdim],
                                            lower=box.lower[:gdim], tol=tol),
                           interleave=True)
        V._twoasis_unique = U
    return U


# Neutral helpers for the names dolfin and DOLFINx spell differently.

def function_space(u):
    return u.function_space


def space_mesh(V):
    return V.mesh


def mesh_comm(mesh):
    return mesh.comm


def geometric_dimension(mesh):
    return mesh.geometry.dim


def element_degree(V):
    return V.ufl_element().embedded_superdegree


def owned_dofs(W, i):
    """The owned dofs of component i of the mixed space W, as global indices of its
    vectors."""
    U = unique_dofs(W)
    parent = flat_collapse_map(W.sub(i).collapse()[1])
    lo, hi = U.owned_range
    c = np.unique(U.compact[parent])
    return c[(c >= lo) & (c < hi)].astype(PETSc.IntType)


def fixed_dofs(W, bcs):
    """The owned dofs of W that the Dirichlet conditions `bcs` constrain, as global
    indices of its vectors."""
    return np.unique(np.concatenate(
        [bc.global_rows() for bc in bcs] or [np.zeros(0, PETSc.IntType)])).astype(
            PETSc.IntType)


# --- vectors and matrices --------------------------------------------------------------

class Vector:
    """dolfin's GenericVector API on a PETSc vector over unique dofs. Vector(x) is a
    copy of x, Vector(petsc_vec) wraps one."""

    def __init__(self, x=None):
        if isinstance(x, Vector):
            self._v = x._v.copy()
        elif isinstance(x, PETSc.Vec):
            self._v = x
        else:
            raise TypeError("Vector(x): a Vector to copy or a PETSc Vec")

    # access
    def vec(self):
        return self._v

    def mpi_comm(self):
        return self._v.getComm().tompi4py()

    def size(self):
        return self._v.getSize()

    def local_size(self):
        return self._v.getLocalSize()

    def __len__(self):
        return self._v.getLocalSize()

    def local_range(self):
        return self._v.getOwnershipRange()

    def get_local(self):
        return self._v.getArray(readonly=True).copy()

    def set_local(self, values):
        with self._v as a:
            a[:] = values

    def apply(self, mode=None):
        pass

    def update_ghost_values(self):
        pass

    def copy(self):
        return Vector(self)

    def __getitem__(self, index):
        return self._v.getArray(readonly=True)[index].copy()

    def __setitem__(self, index, value):
        if isinstance(value, Vector):
            value = value._v.getArray(readonly=True)
        with self._v as a:
            a[index] = value

    # algebra
    def zero(self):
        self._v.zeroEntries()

    def axpy(self, a, x):
        self._v.axpy(a, x._v)

    def norm(self, norm_type="l2"):
        kinds = {"l1": PETSc.NormType.NORM_1, "l2": PETSc.NormType.NORM_2,
                 "linf": PETSc.NormType.NORM_INFINITY}
        return self._v.norm(kinds[norm_type])

    def sum(self):
        return self._v.sum()

    def inner(self, x):
        return self._v.dot(x._v)

    def max(self):
        return self._v.max()[1]

    def min(self):
        return self._v.min()[1]

    def __mul__(self, other):
        out = self.copy()
        out *= other
        return out

    __rmul__ = __mul__

    def __imul__(self, other):
        if isinstance(other, Vector):
            self._v.pointwiseMult(self._v, other._v)
        else:
            self._v.scale(float(other))
        return self

    def __truediv__(self, other):
        return self * (1.0 / float(other))

    def __itruediv__(self, other):
        self._v.scale(1.0 / float(other))
        return self

    def __add__(self, other):
        out = self.copy()
        out += other
        return out

    def __sub__(self, other):
        out = self.copy()
        out -= other
        return out

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self._v.axpy(1.0, other._v)
        else:
            self._v.shift(float(other))
        return self

    def __isub__(self, other):
        if isinstance(other, Vector):
            self._v.axpy(-1.0, other._v)
        else:
            self._v.shift(-float(other))
        return self

    def __neg__(self):
        return self * -1.0


class Matrix:
    """dolfin's GenericMatrix API on a PETSc matrix over unique dofs. Matrix(A) is a
    copy of A (values, pattern and local-to-global maps), Matrix(petsc_mat) wraps one."""

    def __init__(self, A):
        if isinstance(A, Matrix):
            self._m = A._m.duplicate(copy=True)
        elif isinstance(A, PETSc.Mat):
            self._m = A
        else:
            raise TypeError("Matrix(A): a Matrix to copy or a PETSc Mat")
        # dolfin keeps a row's pattern when a Dirichlet condition zeroes it, so that
        # the matrix can be assembled into again.
        self._m.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)

    def mat(self):
        return self._m

    def mpi_comm(self):
        return self._m.getComm().tompi4py()

    def size(self, dim):
        return self._m.getSize()[dim]

    def zero(self):
        self._m.zeroEntries()

    def axpy(self, a, B, same_nonzero_pattern):
        structure = (PETSc.Mat.Structure.SAME_NONZERO_PATTERN if same_nonzero_pattern
                     else PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        self._m.axpy(a, B._m, structure)

    def norm(self, norm_type="frobenius"):
        kinds = {"l1": PETSc.NormType.NORM_1, "frobenius": PETSc.NormType.FROBENIUS,
                 "linf": PETSc.NormType.NORM_INFINITY}
        return self._m.norm(kinds[norm_type])

    def __mul__(self, x):
        y = self._m.createVecLeft()
        self._m.mult(x._v, y)
        return Vector(y)

    def set_nullspace(self, null):
        self._m.setNullSpace(null.petsc)


def as_backend_type(x):
    return x


# --- functions and constants -----------------------------------------------------------

class Function(fem.Function):
    """A DOLFINx Function with dolfin's `vector()`: the compact vector over the space's
    unique dofs (see the module's docstring for the two layouts)."""

    def __init__(self, V, name=None):
        super().__init__(V, name="f" if name is None else name)
        self._unique = unique_dofs(V)
        self._vector = Vector(self._unique.create_vector())

    def vector(self):
        return self._vector

    def sync(self):
        """The compact vector into DOLFINx's array (owned, ghost and slave dofs)."""
        self._unique.prolong(self._vector._v, self)
        return self

    def restrict(self):
        """DOLFINx's array into the compact vector, after it was written directly."""
        self._unique.restrict(self, self._vector._v)
        self._vector._v.stateIncrease()
        return self

    def sub(self, i):
        """A view of component i, whose values are this Function's (sync syncs this)."""
        self.sync()
        view = super().sub(i)
        view._twoasis_parent = self
        return view

    def split(self):
        self.sync()
        return super().split()

    def rename(self, name, label=None):
        self.name = name

    def assign(self, other):
        """The values of `other`, a Function on the same space (dolfin's assign)."""
        if not (isinstance(other, Function) and other.function_space is self.function_space):
            raise NotImplementedError("assign: a Function on the same space only")
        self._vector._v.array[:] = other._vector._v.array_r
        self._vector._v.stateIncrease()


def sync(f):
    """Bring DOLFINx's array of f up to date with its compact vector (if f is a
    backend Function, or a view of one's component); returns f."""
    if isinstance(f, Function):
        f.sync()
    elif hasattr(f, "_twoasis_parent"):
        f._twoasis_parent.sync()
    return f


class Constant(fem.Constant):
    """dolfin's Constant(value): on the last mesh a space was made on, unless given."""

    def __init__(self, value, mesh=None):
        mesh = _DEFAULT_MESH[0] if mesh is None else mesh
        if mesh is None:
            raise RuntimeError("Constant: no mesh yet (make a FunctionSpace first, or "
                               "pass mesh=)")
        super().__init__(mesh, np.asarray(value, dtype=PETSc.ScalarType))

    def values(self):
        return np.atleast_1d(np.asarray(self.value, dtype=float)).ravel().copy()

    def assign(self, value):
        """The value of a number, an array or another Constant."""
        self.value = value.value if isinstance(value, fem.Constant) else value


class Expression(Function):
    """dolfin's Expression(cpp_code, degree=..., **parameters): C++ code in x[0], x[1],
    x[2] and the parameters, compiled as dolfin compiles it
    (femtools.dolfinx.expression.CompiledExpression: <cmath>, near(), DOLFIN_EPS, pi;
    the conditional and logical operators and integer division as in C++); a tuple of
    strings is a vector. The parameters are attributes (expr.t = 1.0). Making one is
    collective (rank 0 compiles, once per code; the library is cached).

    interpolate and set_subfunction evaluate the code at the target's dofs. In a form it
    is, as in dolfin, its interpolant of the given degree (Lagrange, on the last mesh a
    space was made on), computed again when a parameter has changed."""

    def __init__(self, cpp_code, degree=2, element=None, **params):
        from femtools.dolfinx.expression import CompiledExpression
        object.__setattr__(self, "_c", CompiledExpression(cpp_code, MPI.comm_world, **params))
        object.__setattr__(self, "_dirty", True)
        mesh = _DEFAULT_MESH[0]
        if mesh is None:
            raise RuntimeError("Expression: no mesh yet (make a FunctionSpace first)")
        n = self._c.value_size
        if degree == 0:
            family, degree = "DG", 0
        else:
            family, degree = "Lagrange", degree if degree else 2
        el = basix.ufl.element(family if family == "Lagrange" else "Lagrange",
                               mesh.basix_cell(), degree, discontinuous=family == "DG",
                               shape=(n,) if n > 1 else ())
        V = fem.functionspace(mesh, el)
        V = _Space(V.mesh, V.ufl_element(), V._cpp_object)
        V._twoasis_box = None
        super().__init__(V, name="Expression")

    @property
    def _user_parameters(self):
        return self._c.parameters

    def __getattr__(self, key):
        c = self.__dict__.get("_c")
        if c is not None and key in c._names:
            return getattr(c, key)
        raise AttributeError(key)

    def __setattr__(self, key, value):
        c = self.__dict__.get("_c")
        if c is not None and key in c._names:
            setattr(c, key, value)
            object.__setattr__(self, "_dirty", True)
        else:
            object.__setattr__(self, key, value)

    def evaluate(self, x):
        """The values at points x (an array of shape (3, n))."""
        return self._c(x)

    def sync(self):
        if self._dirty:
            fem.Function.interpolate(self, self.evaluate)
            self.restrict()
            object.__setattr__(self, "_dirty", False)
        return self


# --- assembly --------------------------------------------------------------------------

CAPACITY = 256
_FORMS = LRU(CAPACITY)


def _with_quadrature_degree(form):
    degree = parameters["form_compiler"]["quadrature_degree"]
    if degree is None:
        return form
    return ufl.Form([itg.reconstruct(metadata=dict({"quadrature_degree": degree},
                                                   **itg.metadata()))
                     for itg in form.integrals()])


class _Compiled:
    def __init__(self, form):
        self.form = form  # held, so that its id is not reused while it is a key
        self.cpp = fem.form(_with_quadrature_degree(form))
        self.coefficients = [c for c in form.coefficients()
                             if isinstance(c, Function) or hasattr(c, "_twoasis_parent")]
        args = sorted(form.arguments(), key=lambda a: a.number())
        self.spaces = [a.ufl_function_space() for a in args]
        self.rank = len(args)
        self.comm = self.cpp.mesh.comm


def compiled(form):
    """The compiled form of a UFL form, made once per form object (while among the
    CAPACITY most recently used)."""
    entry = _FORMS.get(id(form))
    if entry is None or entry.form is not form:
        entry = _FORMS[id(form)] = _Compiled(form)
    for c in entry.coefficients:
        sync(c)
    return entry


def _new_matrix(c):
    rows = unique_dofs(c.spaces[0])
    cols = unique_dofs(c.spaces[1])
    return Matrix(_assembly.create_matrix(c.cpp, rows, cols))


def assemble(form, tensor=None):
    """dolfin's assemble: a Matrix, a Vector or a number, into `tensor` if given (a
    Matrix made for the same spaces, or a Vector over the test space's unique dofs)."""
    c = compiled(form)
    if c.rank == 0:
        return c.comm.allreduce(fem.assemble_scalar(c.cpp), op=_MPI.SUM)
    if c.rank == 1:
        U = unique_dofs(c.spaces[0])
        b = Vector(U.create_vector()) if tensor is None else tensor
        _assembly.assemble_vector(b._v, c.cpp, U)
        return b
    A = _new_matrix(c) if tensor is None else tensor
    _assembly.assemble_matrix(A._m, c.cpp)
    return A


MATRIX_CAPACITY = 64
_MATRICES = LRU(MATRIX_CAPACITY)


def assemble_matrix(form, bcs=()):
    """The matrix of the bilinear `form` with the Dirichlet conditions `bcs` applied,
    assembled once per equal form and conditions (femtools.fast_assembly's
    assemble_matrix): the same Matrix is returned to every caller."""
    key = (form, tuple(bcs))
    A = _MATRICES.get(key)
    if A is None:
        if len(form.arguments()) != 2:
            raise ValueError("assemble_matrix takes a bilinear form")
        A = assemble(form)
        for bc in bcs:
            bc.apply(A)
        _MATRICES[key] = A
    return A


# --- boundaries and Dirichlet conditions -----------------------------------------------

class BoundaryMarkers:
    """Integer markers on the facets of a mesh, as dolfin's facet MeshFunction:
    set_all(value), mark(where, value) with `where` a vectorized predicate of the
    coordinates (an array of shape (3, n)) or an object with such an `inside_points`;
    a boundary facet is marked when all its vertices satisfy it. `meshtags()` gives
    the boundary facets' markers for DOLFINx (Measure takes the BoundaryMarkers)."""

    def __init__(self, mesh, value=0):
        self.mesh = mesh
        tdim = mesh.topology.dim
        mesh.topology.create_entities(tdim - 1)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        imap = mesh.topology.index_map(tdim - 1)
        self.values = np.full(imap.size_local + imap.num_ghosts, value, dtype=np.int32)
        self.boundary = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        self._tags = None

    def set_all(self, value):
        self.values[:] = value
        self._tags = None

    def mark(self, where, value):
        predicate = getattr(where, "inside_points", where)
        facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh, self.mesh.topology.dim - 1, predicate)
        self.values[facets] = value
        self._tags = None

    def where(self, value):
        """The local facets (owned and ghost) marked `value`."""
        return np.flatnonzero(self.values == value).astype(np.int32)

    def write(self, path):
        """`path` (.xdmf) and its .h5: the exterior facets with a nonzero marker, as
        femtools.boundary_markers.BoundaryMarkers.write writes them (the file partrac
        reads). Collective."""
        from femtools.dolfinx.mesh import write_boundary_facets
        return write_boundary_facets(path, self.mesh, self.values)

    def meshtags(self):
        if self._tags is None:
            facets = np.sort(self.boundary).astype(np.int32)
            self._tags = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim - 1,
                                               facets, self.values[facets])
        return self._tags


def Measure(integral_type, domain=None, subdomain_data=None, **kwargs):
    """ufl.Measure, taking a BoundaryMarkers as subdomain data."""
    if isinstance(subdomain_data, BoundaryMarkers):
        subdomain_data = subdomain_data.meshtags()
    return ufl.Measure(integral_type, domain=domain, subdomain_data=subdomain_data, **kwargs)


class DirichletBC:
    """dolfin's DirichletBC(V, value, markers, marker) on a scalar space or a scalar
    component W.sub(i) of a mixed one, `value` a number, a Constant or (on a whole
    space) a Function on V; DirichletBC(bc) copies one. It acts on the
    compact vectors and matrices: `apply(A)` sets the constrained rows to the
    identity's (keeping the pattern, as dolfin does), `apply(b)` sets the values,
    `apply(A, b)` both. A periodic slave on the boundary is its master (on the
    boundary too, where the data agree)."""

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], DirichletBC):
            other = args[0]
            self.V, self.value, self._local = other.V, other.value, other._local
            self._unique = other._unique
            return
        if len(args) != 4 or not isinstance(args[2], BoundaryMarkers):
            raise NotImplementedError("DirichletBC(V, value, markers, marker): conditions "
                                      "on marked facets only, so far")
        V, value, markers, marker = args
        if V.num_sub_spaces or V.dofmap.index_map_bs != 1:
            raise NotImplementedError("DirichletBC on a scalar space or component only, so far")
        self.V = V
        self.value = value
        # a component's dofs are numbered as its mixed space's (its parent's)
        self._unique = U = unique_dofs(getattr(V, "_twoasis_parent", V))
        mesh = V.mesh
        dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, markers.where(marker))
        compact = U.compact[dofs]
        lo, hi = U.owned_range
        self._local = np.unique(compact[(compact >= lo) & (compact < hi)] - lo).astype(
            PETSc.IntType)

    def function_space(self):
        return self.V

    def homogenize(self):
        self.value = 0.0

    def set_value(self, value):
        self.value = value

    def global_rows(self):
        return (self._local + self._unique.owned_range[0]).astype(PETSc.IntType)

    def values(self):
        """The values at the constrained owned dofs (in the order of `_local`)."""
        g = self.value
        if isinstance(g, Function):
            if g.function_space is not self.V:
                raise NotImplementedError("DirichletBC: a Function value on another space")
            return g.vector()._v.getArray(readonly=True)[self._local].copy()
        if isinstance(g, fem.Constant):
            return np.full(len(self._local), float(np.asarray(g.value)))
        return np.full(len(self._local), float(g))

    def get_boundary_values(self):
        """{local dof: value}, the owned constrained dofs (in the compact numbering)."""
        return dict(zip(self._local.tolist(), self.values().tolist()))

    def apply(self, *tensors):
        for t in tensors:
            if isinstance(t, Matrix):
                t._m.zeroRows(self.global_rows(), diag=1.0)
            elif isinstance(t, Vector):
                with t._v as a:
                    a[self._local] = self.values()
            else:
                raise TypeError("DirichletBC.apply: a Matrix or a Vector")


def apply_bc(bc, *tensors):
    bc.apply(*tensors)


# --- interpolation, mixed components, null spaces --------------------------------------

def interpolate(f, V, out=None):
    """dolfin.interpolate(f, V), into out (a Function on V, made if None)."""
    out = Function(V) if out is None else out
    fem.Function.interpolate(out, _interpolant(f))
    return out.restrict()


def _interpolant(f):
    """What DOLFINx's Function.interpolate takes for f: an Expression's or a constant's
    values at points, or a Function (brought up to date)."""
    if isinstance(f, Expression):
        return f.evaluate
    if isinstance(f, (fem.Constant, int, float)):
        value = np.asarray(f.value if isinstance(f, fem.Constant) else f, dtype=float)
        if value.size != 1:
            raise NotImplementedError("interpolate: a scalar constant only")
        return lambda x: np.full(x.shape[1], float(value))
    return sync(f)


class _Component:
    """Component i of a mixed space W: its collapsed space (with W's periodicity) and
    the map from its dofs to W's."""

    def __init__(self, W, i):
        self.W = W
        space, parent = W.sub(i).collapse()
        space._twoasis_box = W._twoasis_box
        self.space = space
        self.map = flat_collapse_map(parent)
        self.f = None


_COMPONENTS = LRU(64)


def _component(W, i):
    c = _COMPONENTS.get((id(W), i))
    if c is None or c.W is not W:
        c = _COMPONENTS[(id(W), i)] = _Component(W, i)
    return c


def subspace(W, i):
    """The collapsed space of component i of the mixed space W (made once)."""
    return _component(W, i).space


def subfunction(q, i):
    """Component i of the mixed Function q as a Function on its collapsed space,
    the same Function for every call on the same space and component."""
    c = _component(q.function_space, i)
    if c.f is None:
        c.f = Function(c.space)
    q.sync()
    c.f.x.array[:] = q.x.array[c.map]
    return c.f.restrict()


def set_subfunction(q, i, value):
    """Set component i of the mixed Function q to `value`: a Function on its
    collapsed space, or anything Function.interpolate takes (a callable of the
    coordinates, a Function on another space of the mesh)."""
    c = _component(q.function_space, i)
    if not (isinstance(value, fem.Function) and value.function_space is c.space):
        f = Function(c.space)
        fem.Function.interpolate(f, _interpolant(value))
        value = f
    else:
        sync(value)
    q.sync()
    q.x.array[c.map] = value.x.array
    q.restrict()


_VALUES = LRU(16)


def nodal_values(f, V):
    """The values of f at V's owned (unique) dofs, as an array: interpolate(f, V)'s
    get_local(), into a Function kept per space."""
    out = _VALUES.get(id(V))
    if out is None or out.function_space is not V:
        out = _VALUES[id(V)] = Function(V)
    return interpolate(f, V, out=out).vector().get_local()


class AssignedVectorFunction(Function):
    """A vector Function on the vector space of the scalar Functions `u` (all on one
    space, periodic as it is); calling it copies their values in and returns it
    (femtools.mixed.AssignedVectorFunction)."""

    def __init__(self, u, name="Assigned Vector Function"):
        V = u[0].function_space
        Vv = fem.functionspace(V.mesh, basix.ufl.blocked_element(V.ufl_element(),
                                                                 shape=(len(u),)))
        Vv = _Space(Vv.mesh, Vv.ufl_element(), Vv._cpp_object)
        Vv._twoasis_box = V._twoasis_box
        super().__init__(Vv, name=name)
        self.u = list(u)
        self._maps = [flat_collapse_map(Vv.sub(i).collapse()[1]) for i in range(len(u))]

    def __call__(self):
        for m, ui in zip(self._maps, self.u):
            self.x.array[m] = sync(ui).x.array
        return self.restrict()


class _NullSpace:
    """dolfin's VectorSpaceBasis of the constants, on a PETSc NullSpace."""

    def __init__(self, petsc):
        self.petsc = petsc

    def orthogonalize(self, x):
        self.petsc.remove(x._v)


def constant_nullspace(A, V):
    """The constants on the scalar space V as a null space, attached to A."""
    null = _NullSpace(_assembly.constant_nullspace(unique_dofs(V)))
    A.set_nullspace(null)
    return null


def norm(x, norm_type="l2"):
    return x.norm(norm_type)


def normalize(x, normalization_type="average"):
    """Subtract the average of x from it (dolfin's normalize); returns the average."""
    if normalization_type != "average":
        raise NotImplementedError("normalize: 'average' only")
    average = x.sum() / x.size()
    x._v.shift(-average)
    return average


# --- solvers ---------------------------------------------------------------------------

class PETScKrylovSolver:
    """dolfin's PETScKrylovSolver on a petsc4py KSP: `ksp()`, `parameters` (applied at
    every solve, as dolfin does), set_reuse_preconditioner, solve(A, x, b)."""

    _COUNT = [0]

    def __init__(self, method=None, preconditioner=None):
        self._ksp = PETSc.KSP().create(MPI.comm_world)
        self.parameters = {}
        self._reuse = False
        if method is not None:
            # dolfin's PETScKrylovSolver(method, preconditioner), by femtools.ksp's names
            self._COUNT[0] += 1
            ksp.configure(self._ksp, "twoasis_krylov_{}_".format(self._COUNT[0]),
                          ksp.solver_options(method, preconditioner or "default"))

    def ksp(self):
        return self._ksp

    def set_reuse_preconditioner(self, reuse):
        self._reuse = bool(reuse)

    def _update(self):
        p, k = self.parameters, self._ksp
        if p.get("nonzero_initial_guess") is not None:
            k.setInitialGuessNonzero(bool(p["nonzero_initial_guess"]))
        rtol, atol, dtol, maxit = k.getTolerances()
        k.setTolerances(rtol=p.get("relative_tolerance", rtol),
                        atol=p.get("absolute_tolerance", atol),
                        divtol=p.get("divergence_limit", dtol),
                        max_it=p.get("maximum_iterations", maxit))
        if p.get("monitor_convergence") and not getattr(self, "_monitoring", False):
            def monitor(ksp, its, rnorm):
                if MPI.comm_world.rank == 0:
                    print("  {} KSP Residual norm {:.12e}".format(its, rnorm))
            k.setMonitor(monitor)
            self._monitoring = True

    def solve(self, A, x, b):
        k = self._ksp
        k.setOperators(A._m)
        k.getPC().setReusePreconditioner(self._reuse)  # what KSPSetReusePreconditioner does
        self._update()
        k.solve(b._v, x._v)
        reason = k.getConvergedReason()
        its = k.getIterationNumber()
        if reason < 0 and self.parameters.get("error_on_nonconvergence", True):
            raise RuntimeError("PETScKrylovSolver: the solve did not converge (PETSc reason "
                               "{}, {} iterations)".format(reason, its))
        if self.parameters.get("report") and MPI.comm_world.rank == 0:
            print("PETSc Krylov solver ({}, {}) converged in {} iterations.".format(
                k.getType(), k.getPC().getType(), its))
        return its


# --- linear variational problems ----------------------------------------------------

_LINEAR = LRU(64)


def solve(equation, u, bcs=None):
    """dolfin's solve(a == L, u, bcs) for a linear problem: the system assembled, each
    condition applied to its rows (bc.apply(A, b), as dolfin's LinearVariationalSolver
    does), and solved by LU (MUMPS). The matrix and the factorization's solver are kept
    per bilinear form object (the CAPACITY most recent)."""
    a, L = equation.lhs, equation.rhs
    bcs = [] if bcs is None else (list(bcs) if isinstance(bcs, (list, tuple)) else [bcs])
    entry = _LINEAR.get(id(a))
    if entry is None or entry[0] is not a:
        A = assemble(a)
        solver = PETSc.KSP().create(MPI.comm_world)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")
        entry = _LINEAR[id(a)] = (a, A, solver)
    else:
        assemble(a, tensor=entry[1])
    _, A, solver = entry
    b = assemble(L)
    for bc in bcs:
        bc.apply(A, b)
    solver.setOperators(A._m)
    solver.solve(b._v, u.vector()._v)
    if solver.getConvergedReason() < 0:
        raise RuntimeError("solve: LU failed (PETSc reason {})".format(
            solver.getConvergedReason()))
    return u


# --- output ----------------------------------------------------------------------------

class XDMFFile:
    """dolfin's XDMFFile(comm, path) for what the driver writes: facet markers (a
    BoundaryMarkers) and Functions, each with its mesh."""

    def __init__(self, comm, path):
        self.comm, self.path = comm, path
        self._file = self._series = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for f in (self._file, self._series):
            if f is not None:
                f.close()
        self._file = self._series = None

    def write(self, obj, t=None):
        """Facet markers (a BoundaryMarkers) through DOLFINx's XDMFFile; a Function as
        dolfin's TimeSeriesFile writes one (femtools.dolfinx.xdmf: P2 kept)."""
        if isinstance(obj, BoundaryMarkers):
            if self._file is None:
                self._file = dolfinx.io.XDMFFile(self.comm, self.path, "w")
                self._file.write_mesh(obj.mesh)
            self._file.write_meshtags(obj.meshtags(), obj.mesh.geometry)
        else:
            if self._series is None:
                from femtools.dolfinx.xdmf import XDMFSeries
                self._series = XDMFSeries(self.comm, self.path)
            self._series.write(sync(obj), 0.0 if t is None else float(t))


class TimeSeriesFile:
    """A field's time series, `write(function, t)` and `close()`: XDMF in the layout
    dolfin's TimeSeriesFile (femtools.xdmf.XDMFFile2) writes, word for word
    (femtools.dolfinx.xdmf.XDMFSeries): a P2 field keeps its midside nodes, the mesh is
    stored once, and both backends' series read, merge and continue alike."""

    def __init__(self, comm, path):
        from femtools.dolfinx.xdmf import XDMFSeries
        self.path = path
        self._series = XDMFSeries(comm, path)

    def write(self, obj, t):
        self._series.write(sync(obj), float(t))

    def close(self):
        self._series.close()


# --- checkpoints: dolfin's HDF5 layout (femtools.dolfinx.checkpoint) --------------------

def space_id(V):
    """A key for space V, the same while V exists."""
    return id(V)


def write_dof_tables(path, V, name="dofs", mode="w"):
    """V's dof tables in dolfin's layout (femtools.dolfin_layout.write_dofs)."""
    from femtools.dolfinx.checkpoint import write_dofs
    write_dofs(path, V, unique_dofs(V), name, mode)


def write_function_h5(path, u, name, mode="a", dofs=None):
    """u in dolfin's layout, as dolfin's HDF5File.write writes it
    (femtools.dolfin_layout.write_function): dolfin reads it, on any number of ranks."""
    from femtools.dolfinx.checkpoint import write_function
    V = u.function_space
    write_function(path, u.vector().get_local(), V, unique_dofs(V), name, mode, dofs)


def read_function_h5(path, u, name):
    """u from the Function `name` in path, in dolfin's layout (written by dolfin or by
    write_function_h5, on any number of ranks)."""
    from femtools.dolfinx.checkpoint import read_function
    V = u.function_space
    u.vector().set_local(read_function(path, name.strip("/"), V, unique_dofs(V)))
    return u


def read_vector_function_h5(path, name, components):
    """The vector Function `name` in path (dolfin's layout) into the scalar Functions
    `components` (all on one space, which gives the vector space its element and
    periodicity)."""
    u = AssignedVectorFunction(components, name=name)
    read_function_h5(path, u, name)
    u.sync()
    for m, c in zip(u._maps, components):
        c.x.array[:] = u.x.array[m]
        c.restrict()


def read_series_step(path, name, counter, u):
    """Step `counter` of field `name` of an XDMF time series (femtools.xdmf.XDMFFile2's
    layout, either backend's) into u (femtools.dolfinx.xdmf_read.read_nodal)."""
    from femtools.dolfinx.xdmf_read import read_nodal
    read_nodal(path, name, counter, u)
    return u.restrict()


def read_vector_series_step(path, name, counter, components, constrained_domain=None):
    """Step `counter` of the vector field `name` of an XDMF time series into the scalar
    Functions `components` (on one space, its element and periodicity the vector's)."""
    u = AssignedVectorFunction(components, name=name)
    read_series_step(path, name, counter, u)
    u.sync()
    for m, c in zip(u._maps, components):
        c.x.array[:] = u.x.array[m]
        c.restrict()


def weighted_gradient_matrix(V, S, i):
    """The matrix of the derivative along x_i of a Function on S, onto V, as femtools
    .gradient makes it on dolfin: the cells' mean gradients averaged onto V's dofs, a
    dof's cells (those of its row in the DG0-to-V mass matrix) weighted by
    w_c^2 / sum(w), w_c = 1/|c|."""
    DG = FunctionSpace(V.mesh, "DG", 0)
    G = assemble(TrialFunction(DG) * TestFunction(V) * dx)._m
    w = assemble(TestFunction(DG) * dx)._v
    w.reciprocal()
    indptr, cols, _ = G.getValuesCSR()
    need = np.unique(cols).astype(PETSc.IntType)
    local = PETSc.Vec().createSeq(len(need), comm=PETSc.COMM_SELF)
    scatter = PETSc.Scatter().create(w, PETSc.IS().createGeneral(need, comm=PETSc.COMM_SELF),
                                     local, None)
    scatter.scatter(w, local, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    wc = local.getArray(readonly=True)[np.searchsorted(need, cols)]
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    total = np.bincount(rows, weights=wc, minlength=len(indptr) - 1)
    A = G.duplicate()
    A.setValuesCSR(indptr, cols, wc * wc / total[rows])
    A.assemble()
    dP = assemble(TrialFunction(S).dx(i) * TestFunction(DG) * dx)._m
    return Matrix(A.matMult(dP))


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
    "write_dof_tables", "write_function_h5", "space_id",
    "read_function_h5", "read_vector_function_h5", "read_series_step",
    "read_vector_series_step", "weighted_gradient_matrix",
    "read_mesh_file", "write_mesh_h5", "extent", "mark_walls", "UnitSquareMesh", "BoxMesh",
    "mesh_coordinates", "hmax",
    "Mesh", "RectangleMesh", "Point", "Expression", "SpatialCoordinate", "Faces", "PeriodicBox",
    "SubDomain", "conditional", "cos", "exp", "ge", "gt", "le", "lt", "sin", "sqrt",
    "BoxSpec",
]
