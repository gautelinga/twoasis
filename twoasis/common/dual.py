"""The dual mesh: the phase field on a uniform refinement of the flow mesh.

With `phase_refine=k` (k > 0) the Cahn-Hilliard system is solved on the flow mesh
refined uniformly k times (feluga' periodic Plaza: the same cells on dolfin and
DOLFINx, on any number of ranks), the Navier-Stokes equations on the flow mesh. The two
meet through sparse matrices built once (feluga.nested_transfer):

- the velocity onto the fine mesh, exactly (nested Lagrange spaces): the phase field's
  advection, and the statistics that weigh the velocity by the phase field;
- the capillary force, the integral of phi*grad(g) against the velocity's test
  functions, assembled on the fine mesh and taken to the flow mesh's test functions by
  the transpose of that prolongation, exactly;
- the phase field and chemical potential at the flow mesh's nodes (fine nodes too):
  the density and viscosity, nodal on the velocity space as on one mesh, and phi*g at the
  pressure nodes.

`current()` is the run's DualMesh, or None on one mesh. Problem modules are written as
for one mesh: an integral that mixes the two meshes -- the phase field with `ds(1)`, with
`SpatialCoordinate(mesh)`, with an Expression, the pressure, a contact angle's cos(theta)
-- is moved to the phase field's mesh as it is formed (`retarget`, called by the
backends' measures), exactly: the measure and its facet markers go to their refinement,
the coordinates, normals, Expressions and Constants are made there, and a flow-mesh
Function is replaced by its copy there, prolonged before each assembly (said once per
Function). A flow-mesh test or trial function cannot be moved so (it would change the
tensor's dofs): PhiGradRHS assembles the capillary force on the fine mesh itself.
"""

from feluga.lru import LRU
from twoasis.backend import (
    as_backend_type, BoundaryMarkers, Constant, element_degree, Function, FunctionSpace,
    function_space, geometric_dimension, is_lagrange, mesh_id, space_id as _space_key,
    subfunction)

_CURRENT = [None]


def current():
    """The run's DualMesh, or None."""
    return _CURRENT[0]


def start(mesh, passes, constrained_domain=None):
    """The run's DualMesh (`passes` > 0), or None on one mesh: what current() gives from
    now on, replacing an earlier run's in the same process."""
    _CURRENT[0] = DualMesh(mesh, passes, constrained_domain) if passes else None
    return _CURRENT[0]


def _petsc(vector):
    return as_backend_type(vector).vec()


def _updated(vector):
    """After its PETSc vector was written: the ghosts brought up to date (dolfin's;
    DOLFINx's compact vectors have none)."""
    as_backend_type(vector).update_ghost_values()


class DualMesh:
    """The flow mesh `mesh` and its phase-field mesh `fine`, `passes` uniform refinements
    of it, periodic across the faces of `constrained_domain` (a box, or None)."""

    def __init__(self, mesh, passes, constrained_domain=None):
        from feluga.nested_transfer import Nested, refine_uniformly
        d = geometric_dimension(mesh)
        box = constrained_domain
        if box is not None and not hasattr(box, "periodic"):
            raise ValueError("phase_refine: a periodic box (PeriodicBox) or none, not {}"
                             .format(type(box).__name__))
        axes = [] if box is None else [i for i in range(d) if box.periodic[i]]
        upper = None if box is None else list(box.upper[:d])
        lower = None if box is None else list(box.lower[:d])
        self.coarse, self.passes, self.constrained_domain = mesh, int(passes), box
        self.fine, ancestors = refine_uniformly(mesh, self.passes, upper, axes, lower)
        self.nest = Nested(mesh, self.fine, ancestors, self.passes)
        self._spaces, self._mats, self._vecs = {}, {}, {}
        self._twins = LRU(64)    # retarget's copies, by terminal (a form made anew each
        #                          step with a new Function would otherwise pile them up)
        self._ids = {_domain_id(mesh.ufl_domain()): "coarse",
                     _domain_id(self.fine.ufl_domain()): "fine"}

    # --- spaces and markers --------------------------------------------------------

    def fine_space(self, V):
        """The continuous Lagrange space of V's degree on the fine mesh (made once)."""
        key = ("fine", element_degree(V))
        if key not in self._spaces:
            self._spaces[key] = FunctionSpace(self.fine, "CG", key[1],
                                              constrained_domain=self.constrained_domain)
        return self._spaces[key]

    def markers(self, coarse):
        """Facet markers on the fine mesh: each fine facet in a facet of its ancestor
        cell has that facet's marker, the others 0 (feluga.nested_transfer
        .Nested.facet_values). Registered on `coarse`, so that a DirichletBC given
        `coarse` for a space on the fine mesh takes these."""
        fine = BoundaryMarkers(self.fine)
        fine.set_all(0)
        fine.set_values(self.nest.facet_values(coarse.array()))
        _registry(coarse)[mesh_id(self.fine)] = fine
        return fine

    # --- transfers -------------------------------------------------------------------

    def _prolongation(self, Vc, Vf):
        key = ("P", _space_key(Vc), _space_key(Vf))
        if key not in self._mats:
            self._mats[key] = (self.nest.prolongation(Vc, Vf), Vc, Vf)
        return self._mats[key][0]

    def _injection(self, Wf, component, Vc):
        key = ("R", _space_key(Wf), component, _space_key(Vc))
        if key not in self._mats:
            self._mats[key] = (self.nest.injection(Wf, Vc,
                                                   fine_component=component), Wf, Vc)
        return self._mats[key][0]

    def prolong(self, coarse, fine):
        """`fine` (a Function on the fine mesh) = `coarse` (one on the flow mesh) there."""
        P = self._prolongation(function_space(coarse), function_space(fine))
        P.mult(_petsc(coarse.vector()), _petsc(fine.vector()))
        _updated(fine.vector())
        return fine

    def update_twin(self, twin):
        """A flow-mesh Function's copy (`_twin`) prolonged again, if its source changed
        since (PETSc counts every write to a vector: its state)."""
        state = _petsc(twin._twoasis_source.vector()).stateGet()
        if getattr(twin, "_twoasis_state", None) != state:
            self.prolong(twin._twoasis_source, twin)
            twin._twoasis_state = state

    def inject(self, phig, component, out):
        """`out` (a Function on the flow mesh) = component `component` of `phig` (on the
        fine mesh) at out's nodes."""
        R = self._injection(function_space(phig), component, function_space(out))
        R.mult(_petsc(phig.vector()), _petsc(out.vector()))
        _updated(out.vector())
        return out

    def values_at(self, phig, component, V):
        """This rank's values of component `component` of `phig` at V's owned dofs."""
        R = self._injection(function_space(phig), component, V)
        y = self._vecs.get(id(R))
        if y is None:
            y = self._vecs[id(R)] = R.createVecLeft()
        R.mult(_petsc(phig.vector()), y)
        return y.getArray(readonly=True).copy()

    # --- copies on the phase field's mesh ----------------------------------------------

    def _twin(self, t):
        """The terminal `t` of the flow mesh, made on the phase field's mesh (retarget)."""
        twin = self._twins.get(t)
        if twin is not None:
            return twin
        classes = _ufl_module("classes")
        if isinstance(t, (classes.SpatialCoordinate, classes.FacetNormal)):
            twin = type(t)(self.fine.ufl_domain())
        elif isinstance(t, classes.Argument):
            raise ValueError("a test or trial function of the flow mesh in an integral with "
                             "the phase field: assemble it on the phase field's mesh and "
                             "restrict it (DualMesh.restrict_rhs, as PhiGradRHS does)")
        elif hasattr(t, "on_mesh"):                       # DOLFINx's Expression
            twin = t.on_mesh(self.fine)
        elif isinstance(t, classes.Constant):             # DOLFINx's (dolfin's have no mesh)
            twin = Constant(t.value, self.fine).follow(t)
        elif isinstance(t, classes.Coefficient):
            twin = self._function_twin(t)
        else:
            raise ValueError("{} of the flow mesh in an integral with the phase field: it "
                             "has no counterpart on the phase field's mesh".format(
                                 type(t).__name__))
        self._twins[t] = twin
        return twin

    def _function_twin(self, f):
        """A flow-mesh Lagrange Function's copy on the phase field's mesh, prolonged before
        each assembly with it."""
        from twoasis.problems import info_blue
        V = function_space(f)
        if len(f.ufl_shape):
            raise ValueError("a vector Function of the flow mesh in an integral with the "
                             "phase field: write it by components")
        if not is_lagrange(V):
            raise ValueError("a Function of the flow mesh in an integral with the phase "
                             "field: only continuous Lagrange ones have their copy there "
                             "(the nested spaces' prolongation), not {}".format(
                                 V.ufl_element()))
        twin = _Twin(self.fine_space(V))
        twin._twoasis_source = f
        self.update_twin(twin)
        name = f.name() if callable(getattr(f, "name", None)) else getattr(f, "name", "")
        info_blue("Dual mesh: {} moved to the phase field's mesh in an integral with it "
                  "(prolonged before each assembly)".format(name))
        return twin

    # --- checkpoints -----------------------------------------------------------------

    def _lattice(self, W):
        key = ("L", _space_key(W))
        if key not in self._mats:
            self._mats[key] = (self.nest.lattice(W), W)
        return self._mats[key][0]

    def write_function(self, path, f, name, mode="a"):
        """Function `f` on the fine mesh as dataset `name` of `path`, a row per flow-mesh
        cell (feluga.nested_transfer.Lattice): read back on any number of ranks and
        either backend, the fine mesh being made anew. Collective."""
        self._lattice(function_space(f)).write(path, name, _petsc(f.vector()), mode=mode,
                                               attrs={"layout": "dual-lattice"})

    def read_function(self, path, name, f):
        """`f` (a Function on the fine mesh) from dataset `name` of `path` as
        write_function wrote it. Collective."""
        self._lattice(function_space(f)).read(path, name, _petsc(f.vector()))
        _updated(f.vector())
        return f

    def restrict_rhs(self, b_fine, Vf, Vc, out):
        """`out` (a vector over Vc's dofs) = P^T b_fine: a right-hand side assembled
        against Vf's test functions, against Vc's."""
        P = self._prolongation(Vc, Vf)
        P.multTranspose(_petsc(b_fine), _petsc(out))
        return out

    def phase_on_flow_mesh(self, phig):
        """phi, component 0 of `phig`, at the flow mesh's nodes of phi's degree (a
        Function made once and refilled)."""
        key = ("phi_flow", _space_key(function_space(phig)))
        if key not in self._spaces:
            degree = element_degree(function_space(phig).sub(0))
            V = FunctionSpace(self.coarse, "CG", degree,
                              constrained_domain=self.constrained_domain)
            self._spaces[key] = (Function(V), function_space(phig))  # held: id in the key
        return self.inject(phig, 0, self._spaces[key][0])


def phase_on_flow_mesh(q_):
    """phi as a Function on the flow mesh, as files that pair it with the flow mesh
    (partrac's timestamps) want it: its own component of q_["phig"], or on the dual
    mesh its values at the flow mesh's nodes."""
    d = current()
    if d is None:
        return subfunction(q_["phig"], 0)
    return d.phase_on_flow_mesh(q_["phig"])


# --- integrals that mix the two meshes --------------------------------------------------

def _ufl_module(name):
    """`ufl.<name>` of the UFL the backend's forms are written in (ufl or ufl_legacy)."""
    import importlib
    from twoasis.backend import ufl
    return importlib.import_module(ufl.__name__ + "." + name)


def _domain_id(domain):
    return None if domain is None else domain.ufl_id()


def _terminal_domain(t):
    try:
        return _ufl_module("domain").extract_unique_domain(t)
    except (AttributeError, ValueError):
        return None


def retarget(measure, integrand):
    """`(measure, integrand)` moved to the phase field's mesh if the integral involves
    both meshes (or the phase field's with the flow mesh's measure), else None, UFL's own
    product then. See the module's docstring."""
    d = current()
    classes = _ufl_module("classes")
    terminals = [t for t in _ufl_module("corealg.traversal").unique_pre_traversal(integrand)
                 if isinstance(t, classes.Terminal)]
    where = {t: d._ids.get(_domain_id(_terminal_domain(t))) for t in terminals}
    on = set(where.values()) | {d._ids.get(_domain_id(measure.ufl_domain()))}
    if not {"coarse", "fine"} <= on:
        return None
    mapping = {t: d._twin(t) for t, w in where.items() if w == "coarse"}
    if mapping:
        integrand = _ufl_module("algorithms").replace(integrand, mapping)
    data = measure.subdomain_data()      # the flow mesh's facet markers, if any
    if data is not None:
        fine = on_space_mesh(d.fine, data)
        if fine is data:
            raise ValueError("an integral of the phase field over marked facets of the flow "
                             "mesh whose refinement is not known (DualMesh.markers)")
        data = fine
    moved = measure.reconstruct(domain=d.fine.ufl_domain(), subdomain_data=data)
    return moved, integrand


_FUNCTION_SYNC = getattr(Function, "sync", None)     # DOLFINx's: into its own array


class _Twin(Function):
    """A copy of a flow-mesh Function on the phase field's mesh, prolonged from it when
    it changed before each assembly with it (the backends' assemble syncs a form's
    coefficients)."""

    def sync(self):
        current().update_twin(self)
        return _FUNCTION_SYNC(self) if _FUNCTION_SYNC is not None else self


def _registry(markers):
    reg = getattr(markers, "_twoasis_refined", None)
    if reg is None:
        reg = {}
        markers._twoasis_refined = reg
    return reg


def on_space_mesh(V_mesh, markers):
    """The markers to use on a space on `V_mesh`: `markers`, or the fine markers
    DualMesh.markers registered on them for that mesh."""
    reg = getattr(markers, "_twoasis_refined", None)
    if reg:
        return reg.get(mesh_id(V_mesh), markers)
    return markers
