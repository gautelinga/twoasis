"""The velocity and pressure of a BDF time step solved as one system
(velocity_pressure_solve="coupled").

Within a time step the phase field, the density, the viscosity and the momentum matrix
are fixed, and the inner passes of the pressure-correction scheme iterate towards the
solution of

    [ A    W G ] [u]   [b]
    [ D    C   ] [p] = [c]

with A = diag(A_1, ..., A_d) the momentum matrices of the velocity components, G_i the
pressure gradient's matrices (GradFunction), D_i the divergence's (DivRHS), W the
nodal 1/rho with zeros on the velocity's Dirichlet rows, and C zero but for ones on the
pressure's Dirichlet rows (where D's rows are zero and c holds the prescribed values).
The passes reach it only slowly where the viscous term dominates. Here the system is
solved directly: FGMRES, with the block upper-triangular preconditioner

    [ A^   W G ]^-1        S^-1 = alpha0 Ap^-1 + diag(mu / m_Q),
    [ 0   -S^  ]

where A^ is one BoomerAMG cycle per velocity component (one setup when the components
share A), Ap^-1 one BoomerAMG cycle on the pressure matrix, and m_Q the lumped pressure
mass. S^ approximates the Schur complement of the generalised Stokes operator alpha0 M
- div(mu grad) (Cahouet & Chabard, Int. J. Numer. Methods Fluids 8, 1988): alpha0 Ap^-1
where the time derivative dominates, mu/m_Q where the viscous term does. For constant
viscosity it is spectrally equivalent to the Schur complement whatever h and dt, and the
number of iterations is meant not to grow as the mesh is refined. It is the pressure step of a pass
with the rotational update; the coupled solution itself has no rotational term.
"""

from twoasis.backend import (
    apply_bc, as_backend_type, assemble, dx, fixed_dofs, Function, interpolate, ksp, normalize,
    TestFunction)
from twoasis.common import telemetry
from twoasis.problems import info_red


def solver_options(cfg, dim):
    """PETSc options for the coupled solver from its settings `cfg`: the outer
    solver_type, restarted every 100 iterations (fgmres, so that the inner solves may
    also be Krylov ones, set in petsc_options), and one BoomerAMG cycle for each inner
    solve (femtools.ksp.BOOMERAMG_3D's settings in 3D); cfg's petsc_options override
    them."""
    amg = dict(ksp.BOOMERAMG, **(ksp.BOOMERAMG_3D if dim == 3 else {}))
    options = {"ksp_type": ksp.KSP_NAMES.get(cfg.get("solver_type", "fgmres"),
                                             cfg.get("solver_type", "fgmres")),
               "ksp_gmres_restart": 100}
    for inner in ("u_", "p_"):
        options[inner + "ksp_type"] = "preonly"
        options.update({inner + k: v for k, v in amg.items()})
    options.update(cfg.get("petsc_options", {}))
    return options


def _mat(A):
    return as_backend_type(A).mat()


def _vec(x):
    return as_backend_type(x).vec()


class BlockPreconditioner:
    """The preconditioner above, as a petsc4py python PC on the nested system, whose
    vectors are plain ones with the blocks at the index sets `isets` (u_0, ..., p)."""

    def __init__(self, Ku, WG, Kp, mu_m, fixed_p, null, isets):
        self.Ku, self.WG, self.Kp = Ku, WG, Kp
        self.mu_m, self.null, self.isets = mu_m, null, isets
        # the pressure's Dirichlet rows, as local indices: there S is the identity
        self.fixed_p = fixed_p - mu_m.getOwnershipRange()[0]
        self.alpha0 = 1.0
        self.rp = mu_m.duplicate()
        self.tp = mu_m.duplicate()
        self.tu = [G.createVecLeft() for G in WG]

    def apply(self, pc, r_all, z_all):
        # views of the blocks (each rank's part of a block is contiguous)
        r = [r_all.getSubVector(i) for i in self.isets]
        z = [z_all.getSubVector(i) for i in self.isets]
        rp, zp = self.rp, z[-1]
        r[-1].copy(rp)
        if self.null is not None:
            self.null.remove(rp)
        self.Kp.solve(rp, zp)
        zp.scale(self.alpha0)
        self.tp.pointwiseMult(rp, self.mu_m)
        zp.axpy(1.0, self.tp)
        if len(self.fixed_p):
            zp.getArray()[self.fixed_p] = r[-1].getArray(readonly=True)[self.fixed_p]
        if self.null is not None:
            self.null.remove(zp)
        for i, (k, G) in enumerate(zip(self.Ku, self.WG)):
            t = self.tu[i]
            G.mult(zp, t)
            t.aypx(-1.0, r[i])          # r_i - W G z_p
            k.solve(t, z[i])
        for i, rv, zv in zip(self.isets, r, z):
            r_all.restoreSubVector(i, rv)
            z_all.restoreSubVector(i, zv)


class CoupledVelocityPressure:
    """The nested system above and its solver, built once; `solve` refreshes the
    coefficients that change with the phase field and solves one time step's system,
    the result written into x_."""

    def __init__(self, u_components, x_, b, A, Ai, gradp, divu, Ap, p_null, bcs, Q, mu_,
                 homogenize, cfg, parameters):
        from petsc4py import PETSc
        if divu.matvec is None or any(gradp[ui].matvec[0] is None for ui in u_components):
            raise ValueError("velocity_pressure_solve='coupled' needs the gradient and "
                             "divergence matrices: low_memory_version=False")
        self.uc = list(u_components)
        self.x_, self.b, self.bcs, self.mu_ = x_, b, bcs, mu_
        comm = PETSc.COMM_WORLD

        # Masks: zero on each component's Dirichlet rows, one elsewhere
        self.masks = []
        for ui in self.uc:
            m = x_[ui].copy()
            m[:] = 1.0
            apply_bc(homogenize(bcs[ui]), m)
            self.masks.append(_vec(m).copy())
        self.w = [m.duplicate() for m in self.masks]

        # Blocks: W G_i refreshed per step from the constant G_i; D with zero rows and C
        # with ones at the pressure's Dirichlet dofs
        self.G = [_mat(gradp[ui].matvec[0]) for ui in self.uc]
        self.WG = [G.duplicate(copy=True) for G in self.G]
        self.fixed_p = fixed_dofs(Q, bcs['p']).astype(PETSc.IntType)
        D = [_mat(m) for m, _ in divu.matvec]
        C = None
        if comm.tompi4py().allreduce(len(self.fixed_p)):
            D = [Di.duplicate(copy=True) for Di in D]
            for Di in D:
                Di.zeroRows(self.fixed_p, diag=0.0)
            C = _mat(Ap).duplicate()
            C.zeroEntries()
            for row in self.fixed_p:
                C.setValue(row, row, 1.0)
            C.assemble()
        self.D, self.C = D, C
        Amats = [_mat(Ai[ui]) if Ai else _mat(A) for ui in self.uc]
        d = len(self.uc)
        rows = []
        for i in range(d):
            rows.append([Amats[i] if j == i else None for j in range(d)] + [self.WG[i]])
        rows.append(D + [C])
        self.K = PETSc.Mat().createNest(rows, comm=comm)
        self.K.setUp()
        self.isets = self.K.getNestISs()[0]

        # Null space: the constant pressures, when nothing fixes the pressure
        self.p_null = p_null
        null_p = None
        if p_null is not None:
            one = _vec(x_['p']).duplicate()
            one.set(1.0)
            one.normalize()
            null_p = PETSc.NullSpace().create(vectors=[one], comm=comm)
            z = self.K.createVecRight()
            z.set(0.0)
            zp = z.getSubVector(self.isets[-1])
            one.copy(zp)
            z.restoreSubVector(self.isets[-1], zp)
            z.normalize()
            self.K.setNullSpace(PETSc.NullSpace().create(vectors=[z], comm=comm))

        # mu over the lumped pressure mass (the pressure is of a lower degree than the
        # velocity, so mu_ is interpolated onto Q each step)
        self.inv_mass = _vec(assemble(TestFunction(Q) * dx)).copy()
        self.inv_mass.reciprocal()
        self.Q, self.mu_Q = Q, Function(Q)
        self.mu_m = _vec(x_['p']).duplicate()

        # Solvers: the outer FGMRES under the prefix coupled_krylov_solver_, its inner
        # ones under coupled_krylov_solver_u_ and coupled_krylov_solver_p_
        name = "coupled_krylov_solver"
        self.ksp = PETSc.KSP().create(comm)
        ksp.configure(self.ksp, name + "_", cfg)
        ksp.apply_parameters(self.ksp, parameters)
        self.ksp.setOperators(self.K)
        shared = Ai is None
        self.Ku = []
        for i in range(d):
            if shared and i > 0:
                self.Ku.append(self.Ku[0])
                continue
            k = PETSc.KSP().create(comm)
            k.setOptionsPrefix(name + "_u_")
            k.setOperators(Amats[i])
            k.setFromOptions()
            self.Ku.append(k)
        self.Kp = PETSc.KSP().create(comm)
        self.Kp.setOptionsPrefix(name + "_p_")
        self.Kp.setOperators(_mat(Ap))
        self.Kp.setFromOptions()
        self.pc = BlockPreconditioner(self.Ku, self.WG, self.Kp, self.mu_m, self.fixed_p,
                                      null_p, self.isets)
        pc = self.ksp.getPC()
        pc.setType("python")
        pc.setPythonContext(self.pc)
        self.X = self.K.createVecRight()
        self.B = self.K.createVecLeft()
        self.p0 = _vec(x_['p']).duplicate()

    def solve(self, b_tmp, rho_inv_, alpha, dp_, tstep, reuse_velocity, reuse_pressure):
        """Solve the step's system from the current x_ (the velocity of the last step,
        the pressure guess), writing the result into x_; dp_ becomes the pressure's
        change. Returns the number of FGMRES iterations."""
        from petsc4py import PETSc
        x_, b, uc = self.x_, self.b, self.uc
        # Coefficients of this step
        rho_inv = _vec(rho_inv_.vector())
        for i, G in enumerate(self.G):
            self.w[i].pointwiseMult(self.masks[i], rho_inv)
            G.copy(self.WG[i], PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            self.WG[i].diagonalScale(L=self.w[i])
        interpolate(self.mu_, self.Q, out=self.mu_Q)
        self.mu_m.pointwiseMult(_vec(self.mu_Q.vector()), self.inv_mass)
        self.pc.alpha0 = float(alpha[0])
        for k in dict((id(k), k) for k in self.Ku).values():
            k.getPC().setReusePreconditioner(reuse_velocity)
        self.Kp.getPC().setReusePreconditioner(reuse_pressure)

        # Right-hand side and first guess, in the nested layout
        for ui in uc:
            b[ui].zero()
            b[ui].axpy(1.0, b_tmp[ui])
            apply_bc(self.bcs[ui], b[ui])
        b['p'].zero()
        apply_bc(self.bcs['p'], b['p'])
        for key, iset in zip(uc + ['p'], self.isets):
            for src, dst in ((b[key], self.B), (x_[key], self.X)):
                view = dst.getSubVector(iset)
                _vec(src).copy(view)
                dst.restoreSubVector(iset, view)
        _vec(x_['p']).copy(self.p0)

        self.ksp.solve(self.B, self.X)
        its = self.ksp.getIterationNumber()
        reason = self.ksp.getConvergedReason()
        telemetry.current.count("coupled", its)
        if reason < 0:
            if reason != PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT:
                raise RuntimeError("timestep {}: the coupled velocity-pressure solve failed "
                                   "(PETSc reason {}, {} iterations)".format(tstep, reason,
                                                                             its))
            info_red("Warning: timestep {}: the coupled velocity-pressure solve stopped at "
                     "its {} iterations without converging".format(tstep, its))

        # The result into x_; dp_ the pressure's change over the step's solve
        for key, iset in zip(uc + ['p'], self.isets):
            view = self.X.getSubVector(iset)
            view.copy(_vec(x_[key]))
            self.X.restoreSubVector(iset, view)
        if self.p_null is not None:
            normalize(x_['p'])
        for key in uc + ['p']:
            as_backend_type(x_[key]).update_ghost_values()  # the assemblies read ghosts
        dp = dp_.vector()
        dp.zero()
        dp.axpy(1.0, x_['p'])
        _vec(dp).axpy(-1.0, self.p0)
        as_backend_type(dp).update_ghost_values()
        return its
