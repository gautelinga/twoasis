__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-10-03"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from femtools.fast_assembly import assemble, assemble_matrix
from femtools.fast_bc import apply as apply_bc
from femtools.mixed import AssignedVectorFunction, set_subfunction, subfunction, subspace
from dolfin import (Function, Constant, TrialFunction, TestFunction, dx, Vector,
    FunctionSpace, Timer, div, inner, grad, DirichletBC, split)

# The UFL that dolfin was built with: ufl_legacy in recent installations, ufl in
# older ones.
try:
    from ufl_legacy import Coefficient
    if not isinstance(Constant(0.), Coefficient):
        raise ImportError
except ImportError:
    from ufl import Coefficient

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
            self.A = assemble_matrix(Mass, bcs)

        elif method.lower() == "lumping":
            self.A = assemble_matrix(Mass, bcs)
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
            apply_bc(bc, self.rhs)

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
                assemble_matrix(self.test * TrialFunction(Source).dx(i) * dx), p_]

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
            (assemble_matrix(q * TrialFunction(V).dx(i) * dx), u_[i])
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


def phi_g(q_):
    """phi*g at the dofs of the pressure space (an array of this process's
    values). The capillary force being -phi*grad(g), the solver's pressure p is
    the physical pressure minus phi*g. phi and g are each interpolated onto the
    pressure space (femtools.fast_interpolation, a transfer matrix built once),
    so phi*g is exact at its nodes."""
    from femtools.fast_interpolation import values
    Q = q_['p'].function_space()
    phi, g = q_['phig'].split()
    return values(phi, Q) * values(g, Q)


def physical_pressure(q_, out):
    """The physical pressure p + phi*g into out, a Function on the pressure space;
    returns out."""
    x = out.vector()
    x.set_local(q_['p'].vector().get_local() + phi_g(q_))
    x.apply('insert')
    return out


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
                            lower=constrained_domain.lower,
                            tol=getattr(constrained_domain, "tol", None))
