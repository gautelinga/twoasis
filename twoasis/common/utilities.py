__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-10-03"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from dolfin import (assemble, Function, Constant,
    TrialFunction,TestFunction, dx, Vector, Matrix,
    FunctionSpace, Timer, div, Form, inner, grad,
    VectorFunctionSpace, FunctionAssigner, DirichletBC, split)

# The UFL that dolfin was built with: ufl_legacy in recent installations, ufl in
# older ones.
try:
    from ufl_legacy import Coefficient
    from ufl_legacy.tensors import ListTensor
    if not isinstance(Constant(0.), Coefficient):
        raise ImportError
except ImportError:
    from ufl import Coefficient
    from ufl.tensors import ListTensor

# Create some dictionaries to hold work matrices
class Mat_cache_dict(dict):
    """Items in dictionary are matrices stored for efficient reuse."""

    def __missing__(self, key):
        form, bcs = key
        A = assemble(form)
        for bc in bcs:
            bc.apply(A)

        self[key] = A
        return self[key]


A_cache = Mat_cache_dict()


def assemble_matrix(form, bcs=[]):
    """Assemble matrix using cache register.
    """
    assert Form(form).rank() == 2
    return A_cache[(form, tuple(bcs))]


class OasisFunction(Function):
    """Function with more or less efficient projection methods
    of associated linear form.

    The matvec option is provided for letting the right hand side
    be computed through a fast matrix vector product. Both the matrix
    and the Coefficient of the required vector must be provided.

      method = "default"
        Solve the projection with `solver`, a Krylov solver (configured by the
        caller: the velocity_update_solver settings)

      method = "lumping"
        Solve through lumping of mass matrix

    """

    def __init__(self, form, Space, bcs=(),
                 name="x",
                 matvec=None,
                 method="default",
                 solver=None):

        Function.__init__(self, Space, name=name)
        self.form = form
        self.method = method
        self.bcs = list(bcs)
        # [matrix, vector] for a right-hand side matrix*vector; a fresh list per
        # instance (GradFunction fills it in place).
        self.matvec = list(matvec) if matvec is not None else [None, None]
        self.trial = trial = TrialFunction(Space)
        self.test = test = TestFunction(Space)
        Mass = inner(trial, test) * dx()
        self.bf = inner(form, test) * dx()
        self.rhs = Vector(self.vector())

        self.sol = solver
        if method.lower() == "default":
            self.A = A_cache[(Mass, tuple(bcs))]

        elif method.lower() == "lumping":
            self.A = A_cache[(Mass, tuple(bcs))]
            if Space.ufl_element().degree() >= 2:
                # Mass lumping is only accurate for P1; use the consistent mass
                # matrix for higher degrees.
                self.method = "default"
            else:
                ones = Function(Space)
                ones.vector()[:] = 1.
                self.ML = self.A * ones.vector()
                self.ML.set_local(1. / self.ML.get_local())
                self.ML.apply("insert")

    def assemble_rhs(self):
        """
        Assemble right hand side (form*test*dx) in projection
        """
        if not self.matvec[0] is None:
            mat, func = self.matvec
            self.rhs.zero()
            self.rhs.axpy(1.0, mat * func.vector())

        else:
            assemble(self.bf, tensor=self.rhs)

    def __call__(self, assemb_rhs=True):
        """
        Compute the projection
        """
        timer = Timer("Projecting {}".format(self.name()))

        if assemb_rhs:
            self.assemble_rhs()

        for bc in self.bcs:
            bc.apply(self.rhs)

        if self.method.lower() == "default":
            self.sol.solve(self.A, self.vector(), self.rhs)

        else:
            self.vector().zero()
            self.vector().axpy(1.0, self.rhs * self.ML)


class GradFunction(OasisFunction):
    """
    Function used for projecting gradients.

    Typically used for computing pressure gradient on velocity function space.

    """

    def __init__(self, p_, Space, i=0, bcs=[], name="grad", method={}, solver=None):

        assert len(p_.ufl_shape) == 0
        assert i >= 0 and i < Space.mesh().geometry().dim()

        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False)

        OasisFunction.__init__(self, p_.dx(i), Space, bcs=bcs, name=name,
                               method=solver_method, solver=solver)

        self.i = i
        Source = p_.function_space()
        if not low_memory_version:
            self.matvec = [
                A_cache[(self.test * TrialFunction(Source).dx(i) * dx, ())], p_]

        if solver_method.lower() == "gradient_matrix":
            from femtools.gradient import compiled_gradient_module
            DG = FunctionSpace(Space.mesh(), 'DG', 0)
            G = assemble(TrialFunction(DG) * self.test * dx())
            dg = Function(DG)
            dP = assemble(TrialFunction(p_.function_space()).dx(i)
                          * TestFunction(DG) * dx())
            self.WGM = compiled_gradient_module.compute_weighted_gradient_matrix(
                G, dP, dg)

    def assemble_rhs(self, u=None):
        """
        Assemble right hand side trial.dx(i)*test*dx.

        Possible Coefficient u may replace p_ and makes it possible
        to use this Function to compute both grad(p) and grad(dp), i.e.,
        the gradient of pressure correction.

        """
        if isinstance(u, Coefficient):
            self.matvec[1] = u
            self.bf = u.dx(self.i) * self.test * dx()

        if not self.matvec[0] is None:
            mat, func = self.matvec
            self.rhs.zero()
            self.rhs.axpy(1.0, mat * func.vector())
        else:
            assemble(self.bf, tensor=self.rhs)

    def __call__(self, u=None, assemb_rhs=True):
        if isinstance(u, Coefficient):
            self.matvec[1] = u
            self.bf = u.dx(self.i) * self.test * dx()

        if self.method.lower() == "gradient_matrix":
            self.vector().zero()
            self.vector().axpy(1.0, self.WGM * self.matvec[1].vector())
        else:
            OasisFunction.__call__(self, assemb_rhs=assemb_rhs)


class DivRHS:
    """The right-hand side of the pressure equation, the integral of div(u_)*q
    over the domain (q a test function on Q), as a Vector `rhs` that
    assemble_rhs fills: by products of the velocity components with the cached
    matrices of q*v.dx(i)*dx, or, with low_memory, by assembling the form."""

    def __init__(self, u_, Q, low_memory=False):
        q = TestFunction(Q)
        self.form = div(u_) * q * dx
        self.rhs = assemble(self.form)
        V = u_[0].function_space()
        self.matvec = None if low_memory else [
            (A_cache[(q * TrialFunction(V).dx(i) * dx, ())], u_[i])
            for i in range(Q.mesh().geometry().dim())]

    def assemble_rhs(self):
        if self.matvec is None:
            assemble(self.form, tensor=self.rhs)
        else:
            self.rhs.zero()
            for mat, u in self.matvec:
                self.rhs.axpy(1.0, mat * u.vector())


class PhiGradRHS:
    """The capillary force's component i, the integral of phi*g.dx(i)*v over the
    domain (v a test function on V), as a Vector `rhs` that assemble_rhs fills."""

    def __init__(self, phig_, V, i):
        phi_, g_ = split(phig_)
        self.form = phi_ * g_.dx(i) * TestFunction(V) * dx
        self.rhs = assemble(self.form)

    def assemble_rhs(self):
        assemble(self.form, tensor=self.rhs)


class AssignedVectorFunction(Function):
    """Vector function used for postprocessing.

    Assign data from ListTensor components using FunctionAssigner.
    """

    def __init__(self, u, name="Assigned Vector Function"):

        self.u = u
        assert isinstance(u, ListTensor)
        V = u[0].function_space()
        mesh = V.mesh()
        family = V.ufl_element().family()
        degree = V.ufl_element().degree()
        constrained_domain = V.dofmap().constrained_domain
        Vv = VectorFunctionSpace(mesh, family, degree,
                                 constrained_domain=constrained_domain)

        Function.__init__(self, Vv, name=name)
        self.fa = [FunctionAssigner(Vv.sub(i), V) for i, _u in enumerate(u)]

    def __call__(self):
        for i, _u in enumerate(self.u):
            self.fa[i].assign(self.sub(i), _u)


class TPsource(Function):
    """Function used for computing the transposed source to the two-phase equation."""

    def __init__(self, mu, u, Space, bcs=[], name=""):

        Function.__init__(self, Space, name=name)

        dim = Space.mesh().geometry().dim()
        test = TestFunction(Space)
        self.bf = [inner(inner(grad(mu), u.dx(i)), test)
                   * dx for i in range(dim)]

    def assemble_rhs(self, i=0):
        """Assemble right hand side."""
        assemble(self.bf[i], tensor=self.vector())


def homogenize(bcs):
    b = []
    for bc in bcs:
        b0 = DirichletBC(bc)
        b0.homogenize()
        b.append(b0)
    return b


class _Component:
    """Component i of a mixed space W, collapsed into a space of its own, with a
    Function on it and the assigners to it from W and back."""

    def __init__(self, W, i):
        self.space = W.sub(i).collapse()
        self.f = Function(self.space)
        self.get = FunctionAssigner(self.space, W.sub(i))
        self.put = FunctionAssigner(W.sub(i), self.space)


# Collapsing a subspace builds a dofmap and a FunctionAssigner its own maps, so
# both are built once per mixed space and component and shared by every caller.
_COMPONENTS = {}


def _component(W, i):
    key = (W.id(), i)
    if key not in _COMPONENTS:
        _COMPONENTS[key] = _Component(W, i)
    return _COMPONENTS[key]


def subspace(W, i):
    """The collapsed space of component i of the mixed space W (built once)."""
    return _component(W, i).space


def subfunction(q, i):
    """Component i of the mixed Function q, copied into a Function on its
    collapsed space (subspace). The same Function is returned, with the current
    values, on every call for the same mixed space and component."""
    c = _component(q.function_space(), i)
    c.get.assign(c.f, q.sub(i))
    return c.f


def set_subfunction(q, i, value):
    """Set component i of the mixed Function q to value: a Function on the
    component's collapsed space (subspace), or anything else interpolate takes,
    such as an Expression or a Function on another space, which is interpolated
    there first."""
    c = _component(q.function_space(), i)
    if not (isinstance(value, Function) and value.function_space().id() == c.space.id()):
        c.f.interpolate(value)
        value = c.f
    c.put.assign(q.sub(i), value)


def compute_ind(q_, phase=-1):
    """Smoothed indicator (0 to 1) of the phase phi = phase (+1 or -1)."""
    phi = subfunction(q_["phig"], 0)
    ind_ = Function(phi.function_space())
    c_ = 0.5*(1 + phase * phi.vector().get_local())
    c_[c_ < 0] = 0
    c_[c_ > 1] = 1
    ind_.vector().set_local(3*c_**2 - 2*c_**3)
    ind_.vector().apply("insert")
    return ind_


def check_periodic_pairing(constrained_domain, spaces):
    """Stop a parallel run if dolfin's periodic dofmap has given a node on a
    periodic face a dof of its own instead of its partner's (it can, depending on
    the partition; post_import_problem repairs the partition to avoid it). The box
    must come from PeriodicBox, which carries its bounds. Spaces of degree 1 and 2
    are checked (a mixed space when its parts have one degree);
    femtools.periodic_partition.check_space_pairing raises PairingError.
    """
    from dolfin import MPI
    mesh = spaces[0].mesh()
    if MPI.size(mesh.mpi_comm()) == 1 or not hasattr(constrained_domain, "upper"):
        return
    from femtools.periodic_partition import check_space_pairing
    for V in (V for V in spaces if V.ufl_element().degree() <= 2):
        check_space_pairing(V, constrained_domain.upper, constrained_domain.periodic,
                            mesh.mpi_comm(), what=str(V.ufl_element()),
                            lower=constrained_domain.lower)
