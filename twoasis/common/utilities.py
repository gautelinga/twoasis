__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-10-03"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from dolfin import (assemble, Function, Constant,
    TrialFunction,TestFunction, dx, Vector, Matrix,
    FunctionSpace, Timer, div, Form, inner, grad,
    VectorFunctionSpace, FunctionAssigner, PETScKrylovSolver,
    PETScPreconditioner, DirichletBC, split)

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


# Create some dictionaries to hold solvers used for projection
class Solver_cache_dict(dict):
    """Items in dictionary are Linear algebra solvers stored for efficient reuse.
    """

    def __missing__(self, key):
        assert len(key) == 4
        form, bcs, solver_type, preconditioner_type = key
        prec = PETScPreconditioner(preconditioner_type)
        sol = PETScKrylovSolver(solver_type, prec)
        #sol.prec = prec
        #sol = KrylovSolver(solver_type, preconditioner_type)

        #sol.parameters["preconditioner"]["structure"] = "same"
        sol.parameters["error_on_nonconvergence"] = False
        sol.parameters["monitor_convergence"] = False
        sol.parameters["report"] = False
        self[key] = sol
        return self[key]


A_cache = Mat_cache_dict()
Solver_cache = Solver_cache_dict()


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
        Solve projection with regular linear algebra using solver_type
        and preconditioner_type

      method = "lumping"
        Solve through lumping of mass matrix

    """

    def __init__(self, form, Space, bcs=(),
                 name="x",
                 matvec=None,
                 method="default",
                 solver_type="cg",
                 preconditioner_type="default"):

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

        if method.lower() == "default":
            self.A = A_cache[(Mass, tuple(bcs))]
            self.sol = Solver_cache[(Mass, tuple(
                bcs), solver_type, preconditioner_type)]

        elif method.lower() == "lumping":
            self.A = A_cache[(Mass, tuple(bcs))]
            if Space.ufl_element().degree() >= 2:
                # Mass lumping is only accurate for P1; use the consistent mass
                # matrix for higher degrees.
                self.method = "default"
                self.sol = Solver_cache[(Mass, tuple(
                    bcs), solver_type, preconditioner_type)]
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

    def __init__(self, p_, Space, i=0, bcs=[], name="grad", method={}):

        assert len(p_.ufl_shape) == 0
        assert i >= 0 and i < Space.mesh().geometry().dim()

        solver_type = method.get('solver_type', 'cg')
        preconditioner_type = method.get('preconditioner_type', 'default')
        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False)

        OasisFunction.__init__(self, p_.dx(i), Space, bcs=bcs, name=name,
                               method=solver_method, solver_type=solver_type,
                               preconditioner_type=preconditioner_type)

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


class PhiGradFunction(OasisFunction):
    """
    Function used for projecting gradients.

    Typically used for computing pressure gradient on velocity function space.

    """

    def __init__(self, phig_, Space, i=0, bcs=[], name="phigrad", method={}):

        assert len(phig_.ufl_shape) == 1 and phig_.ufl_shape[0] == 2
        assert i >= 0 and i < Space.mesh().geometry().dim()

        solver_type = method.get('solver_type', 'cg')
        preconditioner_type = method.get('preconditioner_type', 'default')
        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False) # !!!

        phi_, g_ = split(phig_)

        OasisFunction.__init__(self, phi_ * g_.dx(i), Space, bcs=bcs, name=name,
                               method=solver_method, solver_type=solver_type,
                               preconditioner_type=preconditioner_type)
        self.i = i

    def assemble_rhs(self):
        """
        Assemble right hand side trial.dx(i)*test*dx.
        """
        assemble(self.bf, tensor=self.rhs)

    def __call__(self):
        OasisFunction.__call__(self)


class DivFunction(OasisFunction):
    """
    Function used for projecting divergence of vector.

    Typically used for computing divergence of velocity on pressure function space.

    """

    def __init__(self, u_, Space, bcs=[], name="div", method={}):

        solver_type = method.get('solver_type', 'cg')
        preconditioner_type = method.get('preconditioner_type', 'default')
        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False)

        OasisFunction.__init__(self, div(u_), Space, bcs=bcs, name=name,
                               method=solver_method, solver_type=solver_type,
                               preconditioner_type=preconditioner_type)

        Source = u_[0].function_space()
        if not low_memory_version:
            self.matvec = [[A_cache[(self.test * TrialFunction(Source).dx(i) * dx, ())], u_[i]]
                           for i in range(Space.mesh().geometry().dim())]

        if solver_method.lower() == "gradient_matrix":
            from femtools.gradient import compiled_gradient_module
            DG = FunctionSpace(Space.mesh(), 'DG', 0)
            G = assemble(TrialFunction(DG) * self.test * dx())
            dg = Function(DG)
            self.WGM = []
            st = TrialFunction(Source)
            for i in range(Space.mesh().geometry().dim()):
                dP = assemble(st.dx(i) * TestFunction(DG) * dx)
                A = Matrix(G)
                self.WGM.append(compiled_gradient_module.compute_weighted_gradient_matrix(A, dP, dg))

    def assemble_rhs(self):
        """
        Assemble right hand side (form*test*dx) in projection
        """
        if not self.matvec[0] is None:
            self.rhs.zero()
            for mat, vec in self.matvec:
                self.rhs.axpy(1.0, mat * vec.vector())

        else:
            assemble(self.bf, tensor=self.rhs)

    def __call__(self, assemb_rhs=True):

        if self.method.lower() == "gradient_matrix":
            # Note that assembling rhs is not necessary using gradient_matrix
            if assemb_rhs:
                self.assemble_rhs()
            self.vector().zero()
            for i in range(self.function_space().mesh().geometry().dim()):
                self.vector().axpy(
                    1., self.WGM[i] * self.matvec[i][1].vector())

        else:
            OasisFunction.__call__(self, assemb_rhs=assemb_rhs)


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


_SUBFUNCTIONS = {}


def subfunction(q, i):
    """Component i of the mixed Function q, copied into a Function on its own
    (collapsed) space. The space, the Function and the assigner are built once
    per mixed space and component, and the same Function is returned (with the
    current values) on every call."""
    W = q.function_space()
    key = (W.id(), i)
    if key not in _SUBFUNCTIONS:
        f = Function(W.sub(i).collapse())
        _SUBFUNCTIONS[key] = (f, FunctionAssigner(f.function_space(), W.sub(i)))
    f, assigner = _SUBFUNCTIONS[key]
    assigner.assign(f, q.sub(i))
    return f


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
