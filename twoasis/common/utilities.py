__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-10-03"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from twoasis.backend import (
    apply_bc, as_backend_type, assemble, assemble_matrix, AssignedVectorFunction, Coefficient, Constant,
    DirichletBC, div, dx, element_degree, Function, function_space, FunctionSpace,
    geometric_dimension, grad, inner, nodal_values, set_subfunction, space_mesh, split,
    subfunction, subspace, TestFunction, Timer, TrialFunction, Vector, weighted_gradient_matrix)

def _vec(v):
    """The PETSc Vec of a backend Vector (or the Vec itself)."""
    return v if hasattr(v, "pointwiseMult") else as_backend_type(v).vec()


def _work(owner, slot, create):
    """A PETSc Vec kept with the PETSc object `owner` under `slot`, made by `create()`
    the first time."""
    name = "twoasis_" + slot
    w = owner.query(name)
    if w is None:
        w = create()
        owner.compose(name, w)
    return w


def _mat(A):
    """The PETSc Mat of a backend Matrix (or the Mat itself)."""
    return A if hasattr(A, "createVecLeft") else as_backend_type(A).mat()


def nonzero_part(A):
    """A PETSc Mat of A's layout holding only A's nonzero entries: for products with a
    matrix stored on a larger pattern than its values need (one block of a mixed
    space's matrix, say). Its products equal A's."""
    import numpy as np
    from petsc4py import PETSc
    M = _mat(A)
    indptr, cols, vals = M.getValuesCSR()
    keep = vals != 0
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    kept = np.zeros(len(indptr), dtype=indptr.dtype)
    kept[1:] = np.cumsum(np.bincount(rows[keep], minlength=len(indptr) - 1))
    return PETSc.Mat().createAIJ(M.getSizes(), csr=(kept, cols[keep], vals[keep]),
                                 comm=M.getComm())


def mult(A, x, slot="mult"):
    """A x, in a PETSc Vec kept with A under `slot`: the next product with A in that slot
    overwrites it."""
    M = _mat(A)
    w = _work(M, slot, M.createVecLeft)
    M.mult(_vec(x), w)
    return w


def mass_norm2(M, x):
    """x^T M x: with M a mass matrix, the square of the L2 norm of the field whose
    dofs x holds. Collective."""
    return _vec(x).dot(mult(M, x, "mass_norm2"))


def pointwise(u, v, slot="pointwise"):
    """The pointwise product of vectors u and v, in a PETSc Vec kept with u under
    `slot`."""
    U = _vec(u)
    w = _work(U, slot, U.duplicate)
    w.pointwiseMult(U, _vec(v))
    return w


def axpy(y, a, w):
    """y += a*w, w a PETSc Vec (from mult or pointwise)."""
    _vec(y).axpy(a, w)


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
        self.projection_name = name
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
            if element_degree(Space) >= 2:
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
            axpy(self.rhs, 1.0, mult(mat, func.vector()))

        else:
            assemble(self.bf, tensor=self.rhs)

    def __call__(self, assemb_rhs=True):
        """
        Compute the projection
        """
        timer = Timer("Projecting {}".format(self.projection_name))

        if assemb_rhs:
            self.assemble_rhs()

        for bc in self.bcs:
            apply_bc(bc, self.rhs)

        if self.method.lower() == "default":
            self.sol.solve(self.A, self.vector(), self.rhs)

        else:
            self.vector().zero()
            axpy(self.vector(), 1.0, pointwise(self.rhs, self.ML))


class GradFunction(OasisFunction):
    """
    Function used for projecting gradients.

    Typically used for computing pressure gradient on velocity function space.

    """

    def __init__(self, p_, Space, i=0, bcs=[], name="grad", method={}, solver=None):

        assert len(p_.ufl_shape) == 0
        assert i >= 0 and i < geometric_dimension(space_mesh(Space))

        solver_method = method.get('method', 'default')
        low_memory_version = method.get('low_memory_version', False)

        OasisFunction.__init__(self, p_.dx(i), Space, bcs=bcs, name=name,
                               method=solver_method, solver=solver)

        self.i = i
        Source = function_space(p_)
        if not low_memory_version:
            self.matvec = [
                assemble_matrix(self.test * TrialFunction(Source).dx(i) * dx), p_]

        if solver_method.lower() == "gradient_matrix":
            self.WGM = weighted_gradient_matrix(Space, Source, i)

    def assemble_rhs(self, u=None):
        """
        Assemble right hand side trial.dx(i)*test*dx.

        Possible Coefficient u may replace p_ and makes it possible
        to use this Function to compute both grad(p) and grad(dp), i.e.,
        the gradient of pressure correction.

        """
        if isinstance(u, Coefficient):
            self._source(u)
        if not self.matvec[0] is None:
            mat, func = self.matvec
            self.rhs.zero()
            axpy(self.rhs, 1.0, mult(mat, func.vector()))
        else:
            assemble(self.bf, tensor=self.rhs)

    def _source(self, u):
        """Differentiate u from now on; the form is made only where it is assembled
        (no matrix) and only when u is another Function."""
        if u is not self.matvec[1]:
            self.matvec[1] = u
            if self.matvec[0] is None:
                self.bf = u.dx(self.i) * self.test * dx()

    def __call__(self, u=None, assemb_rhs=True):
        if isinstance(u, Coefficient):
            self._source(u)
        if self.method.lower() == "gradient_matrix":
            self.vector().zero()
            axpy(self.vector(), 1.0, mult(self.WGM, self.matvec[1].vector()))
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
        V = function_space(u_[0])
        self.matvec = None if low_memory else [
            (assemble_matrix(q * TrialFunction(V).dx(i) * dx), u_[i])
            for i in range(geometric_dimension(space_mesh(Q)))]

    def assemble_rhs(self):
        if self.matvec is None:
            assemble(self.form, tensor=self.rhs)
        else:
            self.rhs.zero()
            for mat, u in self.matvec:
                axpy(self.rhs, 1.0, mult(mat, u.vector()))


class PhiGradRHS:
    """The capillary force's component i, the integral of phi*g.dx(i)*v over the
    domain (v a test function on V), as a Vector `rhs` that assemble_rhs fills.

    On the dual mesh (phig on a refinement of V's mesh) it is assembled there, against
    the test functions of V's element on that mesh, and taken to V's by the transpose
    of the prolongation (DualMesh.restrict_rhs): V's basis functions are those fine ones
    combined, so this is the same integral, every quadrature point on the fine mesh."""

    def __init__(self, phig_, V, i):
        from twoasis.common import dual
        phi_, g_ = split(phig_)
        self.dual = dual.current()
        self.V = V
        if self.dual is None:
            self.form = phi_ * g_.dx(i) * TestFunction(V) * dx
            self.rhs = assemble(self.form)
            return
        self.Vf = self.dual.fine_space(V)
        self.form = phi_ * g_.dx(i) * TestFunction(self.Vf) * dx
        self.fine = assemble(self.form)
        self.rhs = Vector(Function(V).vector())
        self.assemble_rhs()

    def assemble_rhs(self):
        if self.dual is None:
            assemble(self.form, tensor=self.rhs)
            return
        assemble(self.form, tensor=self.fine)
        self.dual.restrict_rhs(self.fine, self.Vf, self.V, self.rhs)


class TPsource(Function):
    """Function used for computing the transposed source to the two-phase equation."""

    def __init__(self, mu, u, Space, bcs=[], name=""):

        Function.__init__(self, Space, name=name)

        dim = geometric_dimension(space_mesh(Space))
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
    pressure space (feluga.fast_interpolation, a transfer matrix built once),
    so phi*g is exact at its nodes."""
    from twoasis.common import dual
    Q = function_space(q_['p'])
    d = dual.current()
    if d is not None:     # on the dual mesh: at Q's nodes, which are the fine mesh's too
        return d.values_at(q_['phig'], 0, Q) * d.values_at(q_['phig'], 1, Q)
    phi, g = q_['phig'].split()
    return nodal_values(phi, Q) * nodal_values(g, Q)


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
    ind_ = Function(function_space(phi))
    c_ = 0.5*(1 + phase * phi.vector().get_local())
    c_[c_ < 0] = 0
    c_[c_ > 1] = 1
    ind_.vector().set_local(3*c_**2 - 2*c_**3)
    ind_.vector().apply("insert")
    return ind_
