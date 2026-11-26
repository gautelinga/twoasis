"""The velocity, pressure, phase field and chemical potential of a BDF time step solved
as one linear system (chns_solve="coupled").

The split step solves the phase field first, advected by the extrapolated velocity, and
then the velocity and pressure with the capillary force -phi grad g on the right-hand
side. That is explicit in the capillary force, and the step is stable only below a
time step that falls as 1/sqrt(sigma). Here the two are one system, linear in the
unknowns: with phi* the phase field extrapolated to the new time level (beta, as the
convecting velocity),

    [ A    W G   W C ] [u   ]   [b   ]
    [ D    Cp    B   ] [p   ] = [c   ]
    [ T    0     P   ] [phig]   [b_pf]

where A, W G, D and Cp are the blocks of the coupled velocity-pressure system
(coupled.py), P the (phi, g) matrix of the split step without its advection term, and

    C_i: int phi* d_i(g) v dx                                (the capillary force)
    T_i: -int phi* u_i d_i(xi) dx + int u_i n_i phi* xi ds   (the conservative advection)

with g and u_i the unknowns. B holds the physical pressure's boundary conditions
(PhysicalPressureBC: p + phi g = P on open ends, p the solver's pressure): their rows of
the pressure are p + phi* g = P, implicit in g, B the phi* at the boundary vertices (a
boundary form with vertex quadrature, its rows scaled by the lumped boundary mass) and
the right-hand side P. Lagged instead (p = P - phi* g*, the boundary values set before
the step, as the split step does with its new g), the meniscus's flux through the open
ends would be integrated explicitly, and the coupled step would become unstable once the
time step passes the inverse of that mode's frequency. The linearisation is the step's
own (phi* for phi^(n+1)), of second order. The density, the viscosity and the convecting
velocity are those of the extrapolated fields, as in the split step. With matched
densities (W the constant 1/rho) C^T = -T but for the boundary term, which vanishes
where the velocity is zero, so the force and the advection exchange energy exactly; the
linear coupled scheme of Garcke, Hinze & Kahle (Appl. Numer. Math. 99, 2016) rests on
this. With variable density W weights the force by 1/rho at the velocity's nodes and the
exchange is no longer exact.

The system is solved by chns_solver's method:

- "direct": the nest of the blocks is converted to one AIJ matrix each step and
  factorised (MUMPS LU by default, chns_solver's petsc_options).
- "iterative" (the default): FGMRES on the nest, with the block upper-triangular
  preconditioner

      [ K_NS   W C ]^-1       K_NS = [ A  W G ]      S ~ P - T K_NS^-1 W C
      [ 0      S   ]                 [ D  Cp  ]

  S is the Schur complement of the phase field, approximated by P. Where the time
  derivative dominates K_NS (K_NS^-1 ~ M_u^-1/alpha0), T M_u^-1 W C would add the mobility
  W phi*^2/alpha0 to P (Shen & Yang's stabilised scheme), but the pressure takes
  the gradient part of the force, and with it most of that response: the added mobility
  overestimates it, more so at large time steps, and P alone is the better approximation.
  S is solved by the phase-field solver (PhaseFieldSchur) and K_NS by FGMRES with the
  coupled velocity-pressure preconditioner (coupled.py), each to a loose tolerance
  (inner_tolerance, INNER_ITERATIONS); the outer FGMRES's tolerances are chns_solver's
  (KRYLOV_TOLERANCES), tested against the initial residual. The outer solver's PETSc
  options are chns_solver's petsc_options, the inner ones those under ns_ and pf_.

  The flow solve's tolerance is set against the outer residual's velocity and pressure
  part, not against the flow solve's own right-hand side: that holds the capillary force
  of the phase field's correction, mostly balanced by a pressure gradient and far larger
  than the outer residual, so that a test relative to it would end with an error far
  above the outer residual. The flow solve starts from the pressure that takes that
  balanced part: Ap p0 = -D M_u^-1 r_u, r_u the flow
  solve's momentum right-hand side and M_u the lumped velocity mass, the pressure of the
  right-hand side's Helmholtz projection (Ap ~ -D M_u^-1 W G), by one cycle of the
  pressure preconditioner's Ap^-1.

  chns_solver's inner "lu" solves both inner systems by LU instead (MUMPS), for
  diagnosis: the outer iterations then measure the block approximation alone; "lu_ns"
  and "lu_pf" only the flow's or only the phase field's.
"""

from twoasis.backend import (
    as_backend_type, assemble, dx, fixed_dofs, ksp, split, TestFunction, TestFunctions,
    TrialFunction, TrialFunctions)
from twoasis.common import telemetry
from twoasis.problems import info_red
from twoasis.solvers.TPfracStep import INITIAL_RESIDUAL, prepare_phase_field_preconditioner
from twoasis.solvers.TPfracStep.coupled import finish_solve, from_nest, momentum_rhs, to_nest


def _mat(A):
    return as_backend_type(A).mat()


def _vec(x):
    return as_backend_type(x).vec()


LU_OPTIONS = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
INNER_ITERATIONS = 15           # the iterative method's inner solves, the flow's and the
                                # phase field's: iteration limit, each stopping at
                                # inner_tolerance or here, whichever comes first

# chns_solver's settings (see the module docstring), and the values each may take
DEFAULTS = dict(
    method="iterative",
    inner_tolerance=1e-1,       # the iterative method's inner solves: relative tolerance
    inner="krylov",
    petsc_options={},
)
CHOICES = dict(
    method=("iterative", "direct"),
    inner=("krylov", "lu", "lu_ns", "lu_pf"),
)
# the outer solver's Krylov parameters (krylov_solver_parameters, feluga.ksp.apply_parameters)
KRYLOV_KEYS = ("relative_tolerance", "absolute_tolerance", "divergence_limit",
               "maximum_iterations", "monitor_convergence", "nonzero_initial_guess")


def settings(cfg):
    """chns_solver's settings `cfg` completed with DEFAULTS; ValueError for a key or a
    value it does not know."""
    unknown = set(cfg) - set(DEFAULTS) - set(KRYLOV_KEYS)
    if unknown:
        raise ValueError("chns_solver: unknown keys {}".format(sorted(unknown)))
    cfg = dict(DEFAULTS, **cfg)
    for key, allowed in CHOICES.items():
        if cfg[key] not in allowed:
            raise ValueError("chns_solver {} {!r}: one of {}".format(key, cfg[key], allowed))
    return cfg


def _pressure_null_space(X, iset):
    """The constant pressures (block `iset` of vectors like X) as a PETSc null space."""
    from petsc4py import PETSc
    z = X.duplicate()
    z.set(0.0)
    zp = z.getSubVector(iset)
    zp.set(1.0)
    z.restoreSubVector(iset, zp)
    z.normalize()
    return PETSc.NullSpace().create(vectors=[z], comm=X.getComm())


def _lu(comm, prefix, null):
    """A KSP that solves by LU (MUMPS), under the options prefix `prefix`, with null
    pivots allowed if `null`."""
    from petsc4py import PETSc
    k = PETSc.KSP().create(comm)
    options = dict(LU_OPTIONS)
    if null:
        options["mat_mumps_icntl_24"] = 1
    ksp.configure(k, prefix, options)
    return k


class CHNSPreconditioner:
    """The block upper-triangular preconditioner above, as a petsc4py python PC on the
    nest: S^-1 on the (phi, g) block by the KSP kpf, then K_NS^-1 on the velocity and
    pressure, the capillary force of that phi and g (and B's boundary pressure) taken
    off their residual, by the
    KSP kns on the velocity-pressure nest (vector rn, zn in its layout)."""

    def __init__(self, owner):
        self.o = owner
        self.its = [0, 0]

    def apply(self, pc, r, z):
        o = self.o
        isets, vis, d = o.isets, o.vp.isets, len(o.vp.uc)
        r_pf, z_pf = r.getSubVector(isets[-1]), z.getSubVector(isets[-1])
        o.kpf_apply.solve(r_pf, z_pf)
        self.its[0] += o.kpf_apply.getIterationNumber()
        for i in range(d):
            ri, dst = r.getSubVector(isets[i]), o.rn.getSubVector(vis[i])
            o.WC[i].mult(z_pf, dst)
            dst.aypx(-1.0, ri)                      # r_u - W C z_pf
            r.restoreSubVector(isets[i], ri)
            o.rn.restoreSubVector(vis[i], dst)
        rp, dst = r.getSubVector(isets[d]), o.rn.getSubVector(vis[d])
        if o.Bpg is not None:
            o.Bpg.mult(z_pf, dst)
            dst.aypx(-1.0, rp)                      # r_p - B z_pf
        else:
            rp.copy(dst)
        r.restoreSubVector(isets[d], rp)
        o.rn.restoreSubVector(vis[d], dst)
        o.zn.set(0.0)
        if o.kns_apply is o.kns:                    # an LU solve starts from zero
            self.pressure_guess()
        self.set_ns_tolerance(r)
        o.kns_apply.solve(o.rn, o.zn)
        self.its[1] += o.kns_apply.getIterationNumber()
        for i in range(d + 1):
            src, zi = o.zn.getSubVector(vis[i]), z.getSubVector(isets[i])
            src.copy(zi)
            o.zn.restoreSubVector(vis[i], src)
            z.restoreSubVector(isets[i], zi)
        r.restoreSubVector(isets[-1], r_pf)
        z.restoreSubVector(isets[-1], z_pf)

    def set_ns_tolerance(self, r):
        """The flow solve's tolerance against the outer residual's flow part, not
        against its own right-hand side, which holds the capillary force of the phase
        field's correction, mostly balanced by a pressure gradient."""
        o = self.o
        isets, d = o.isets, len(o.vp.uc)
        ss = 0.0
        for i in range(d + 1):
            ri = r.getSubVector(isets[i])
            ss += ri.norm() ** 2
            r.restoreSubVector(isets[i], ri)
        rn = o.rn.norm()
        rtol = o.inner_tol * min(1.0, ss ** 0.5 / rn) if rn > 0 else o.inner_tol
        # PETSc tests against ||rn|| even from a nonzero first guess, hence the ratio
        o.kns_apply.setTolerances(rtol=rtol)

    def pressure_guess(self):
        """The flow solve's initial pressure (see the docstring) in zn, its velocity
        zero; from rn."""
        o = self.o
        vp, vis, d = o.vp, o.vp.isets, len(o.vp.uc)
        zp, s = o.zn.getSubVector(vis[d]), o.s_p
        s.set(0.0)
        for i in range(d):
            ri, t = o.rn.getSubVector(vis[i]), o.t_u[i]
            t.pointwiseMult(ri, o.minv_u[i])        # M_u^-1 r_u, zero on Dirichlet rows
            o.rn.restoreSubVector(vis[i], ri)
            vp.D[i].multAdd(t, s, s)
        s.scale(-1.0)                               # Ap p0 = -D M_u^-1 r_u
        if vp.pc.null is not None:
            vp.pc.null.remove(s)
        vp.Kp.solve(s, zp)
        if len(o.fixed_p):                          # C's rows: p = r_p there
            rp = o.rn.getSubVector(vis[d])
            zp.getArray()[o.fixed_p] = rp.getArray(readonly=True)[o.fixed_p]
            o.rn.restoreSubVector(vis[d], rp)
        if vp.pc.null is not None:
            vp.pc.null.remove(zp)
        o.zn.restoreSubVector(vis[d], zp)



class CoupledCHNS:
    """The system above and its solver, built once from the coupled velocity-pressure
    system `vp` (a CoupledVelocityPressure), the (phi, g) matrix Apf and the space
    W of (phi, g); `phig_s` is the Function on W that holds phi* (its g is unused).
    For the iterative method also: `pf_sol`, the phase-field solver (it solves S = P), and
    `parameters`, the outer FGMRES's Krylov parameters.
    `solve` reassembles the blocks that change with phi* and solves one step."""

    def __init__(self, vp, Apf, V, W, phig_s, normal, ds, bcs, cfg,
                 pf_sol=None, parameters=None, Q=None):
        from petsc4py import PETSc
        cfg = settings(cfg)
        self.vp, self.Apf, self.bcs = vp, Apf, bcs
        comm = PETSc.COMM_WORLD
        phi_s = split(phig_s)[0]
        _, g = TrialFunctions(W)
        xi, _ = TestFunctions(W)
        u, v = TrialFunction(V), TestFunction(V)
        d = len(vp.uc)
        self.c_forms = [phi_s * g.dx(i) * v * dx for i in range(d)]
        self.t_forms = [-phi_s * u * xi.dx(i) * dx + phi_s * u * normal[i] * xi * ds
                        for i in range(d)]
        self.Cm = [assemble(f) for f in self.c_forms]
        self.Tm = [assemble(f) for f in self.t_forms]
        self.WC = [_mat(C) for C in self.Cm]
        self.fixed_pf = fixed_dofs(W, bcs['phig']).astype(PETSc.IntType)

        # The physical pressure's boundary rows: p + phi* g = P (B, see the docstring)
        self.physical = [bc for bc in bcs['p'] if hasattr(bc, 'update_physical')]
        self.Bpg = None
        if self.physical:
            q = TestFunction(Q)
            dsv = ds(metadata={"quadrature_degree": 1, "quadrature_rule": "vertex"})
            self.b_form = phi_s * g * q * dsv
            self.Bm = assemble(self.b_form)
            w = _vec(assemble(q * dsv))
            rows = fixed_dofs(Q, self.physical).astype(PETSc.IntType)
            self.s_pg = w.duplicate()
            self.s_pg.set(0.0)
            self.s_pg.setValues(rows, 1.0 / w.getValues(rows))
            self.s_pg.assemble()
            self.Bpg = _mat(self.Bm)
            self.Bpg.diagonalScale(L=self.s_pg)

        A = vp.Amats
        rows = []
        for i in range(d):
            rows.append([A[i] if j == i else None for j in range(d)]
                        + [vp.WG[i], self.WC[i]])
        rows.append(vp.D + [vp.C, self.Bpg])
        rows.append([_mat(T) for T in self.Tm] + [None, _mat(Apf)])
        self.K = PETSc.Mat().createNest(rows, comm=comm)
        self.K.setUp()
        self.isets = self.K.getNestISs()[0]
        self.Kaij = None
        self.method = cfg["method"]
        prefix = "chns_solver_"
        self.ksp = PETSc.KSP().create(comm)

        if self.method == "direct":
            options = dict(LU_OPTIONS)
            if vp.p_null is not None:
                # the constant pressures are in the null space: MUMPS's null pivot detection
                options["mat_mumps_icntl_24"] = 1
            options.update(cfg["petsc_options"])
            ksp.configure(self.ksp, prefix, options)
            return

        # Iterative: FGMRES on the nest
        self.X, self.B = self.K.createVecRight(), self.K.createVecLeft()
        if vp.p_null is not None:
            self.K.setNullSpace(_pressure_null_space(self.X, self.isets[-2]))
        options = dict({"ksp_type": "fgmres", "ksp_gmres_restart": 100}, **INITIAL_RESIDUAL)
        options.update({k: v for k, v in cfg["petsc_options"].items()
                        if not k.startswith(("ns_", "pf_"))})
        ksp.configure(self.ksp, prefix, options)
        ksp.apply_parameters(self.ksp, parameters or {})
        self.ksp.setOperators(self.K)
        self.pc = CHNSPreconditioner(self)
        pc = self.ksp.getPC()
        pc.setType("python")
        pc.setPythonContext(self.pc)

        tol, maxit = cfg["inner_tolerance"], INNER_ITERATIONS
        inner = {k[3:]: v for k, v in cfg["petsc_options"].items() if k.startswith("ns_")}
        # K_NS: FGMRES with the velocity-pressure preconditioner
        self.kns = PETSc.KSP().create(comm)
        ksp.configure(self.kns, prefix + "ns_", dict({"ksp_type": "fgmres"}, **inner))
        self.kns.setTolerances(rtol=tol, atol=0.0, max_it=maxit)
        self.inner_tol = tol
        self.inner = cfg["inner"]
        self.kns.setInitialGuessNonzero(True)
        self.fixed_p = vp.fixed_p - vp.mu_m.getOwnershipRange()[0]
        m = _vec(assemble(v * dx))
        self.minv_u = []
        for i in range(d):
            w = m.copy()
            w.reciprocal()
            w.pointwiseMult(w, vp.masks[i])
            self.minv_u.append(w)
        self.t_u = [w.duplicate() for w in self.minv_u]
        self.s_p = vp.mu_m.duplicate()
        self.kns.setOperators(vp.K)
        self.kns.getPC().setType("python")
        self.kns.getPC().setPythonContext(vp.pc)
        self.rn, self.zn = vp.K.createVecLeft(), vp.K.createVecRight()
        # S: the phase-field solver, on S's matrix
        self.kpf = pf_sol.ksp()
        self.kpf.setInitialGuessNonzero(False)
        self.kpf.setTolerances(rtol=tol, atol=0.0, max_it=maxit)
        for k, val in cfg["petsc_options"].items():
            if k.startswith("pf_"):
                PETSc.Options()[self.kpf.getOptionsPrefix() + k[3:]] = val
        self.kpf.setFromOptions()
        self.kns_apply, self.kpf_apply = self.kns, self.kpf

        # LU inner solves, for diagnosis (see the docstring)
        if self.inner in ("lu", "lu_ns"):
            self.kns_apply = _lu(comm, prefix + "ns_lu_", vp.p_null is not None)
            self.Kns_aij = None
        if self.inner in ("lu", "lu_pf"):
            self.kpf_apply = _lu(comm, prefix + "pf_lu_", False)

    def solve(self, b_tmp, b_pf, x_, b, rho_inv_, alpha, dp_, tstep, reuse_velocity=False,
              reuse_pressure=False, mobility=None, reuse_phase_field=False):
        """Solve the step's system, writing u, p and (phi, g) into x_; dp_ becomes the
        pressure's change. b_tmp: the momentum right-hand side without the capillary
        force; b_pf: the (phi, g) right-hand side, Apf and b_pf with the phase field's
        Dirichlet conditions applied. For the iterative method, `mobility` is
        (M, sigma_bar, epsilon), for the phase-field preconditioner's H. Returns the
        number of outer iterations (0 for the direct solve)."""
        from petsc4py import PETSc
        vp, uc = self.vp, self.vp.uc
        if self.method == "direct":
            vp.scale_gradients(rho_inv_)
        else:
            vp.prepare(rho_inv_, alpha, reuse_velocity, reuse_pressure)
        if self.Bpg is not None:
            assemble(self.b_form, tensor=self.Bm)
            self.Bpg.diagonalScale(L=self.s_pg)
            for bc in self.physical:
                bc.boundary_value.vector()[:] = bc.physical
        for i in range(len(uc)):
            assemble(self.c_forms[i], tensor=self.Cm[i])
            self.WC[i].diagonalScale(L=vp.w[i])  # 1/rho, zero on Dirichlet rows
            assemble(self.t_forms[i], tensor=self.Tm[i])
            _mat(self.Tm[i]).zeroRows(self.fixed_pf, diag=0.0)

        if self.method == "direct":
            # petsc4py's convert() without `out` converts in place: the nest would be gone
            if self.Kaij is not None:
                self.K.convert("aij", out=self.Kaij)
            else:
                self.Kaij = self.K.convert("aij", out=PETSc.Mat())
                self.X, self.B = self.Kaij.createVecs()
                if vp.p_null is not None:
                    self.Kaij.setNullSpace(_pressure_null_space(self.X, self.isets[-2]))
            self.ksp.setOperators(self.Kaij)
        else:
            self.kpf.setOperators(_mat(self.Apf))
            prepare_phase_field_preconditioner(self.kpf, reuse_phase_field, alpha[0], *mobility)
            if self.inner in ("lu", "lu_ns"):
                if self.Kns_aij is None:
                    self.Kns_aij = vp.K.convert("aij", out=PETSc.Mat())
                else:
                    vp.K.convert("aij", out=self.Kns_aij)
                self.kns_apply.setOperators(self.Kns_aij)
            if self.inner in ("lu", "lu_pf"):
                self.kpf_apply.setOperators(_mat(self.Apf))
            self.pc.its = [0, 0]

        momentum_rhs(b, b_tmp, self.bcs, uc)
        p0 = _vec(x_['p']).copy()
        keys = uc + ['p', 'phig']
        to_nest([b[ui] for ui in uc] + [b['p'], b_pf], self.B, self.isets)
        if self.method == "iterative":
            to_nest([x_[k] for k in keys], self.X, self.isets)

        self.ksp.solve(self.B, self.X)
        reason = self.ksp.getConvergedReason()
        its = self.ksp.getIterationNumber() if self.method == "iterative" else 0
        if reason < 0:
            if self.method == "iterative" and reason == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT:
                info_red("Warning: timestep {}: the coupled CH-NS solve stopped at its {} "
                         "iterations without converging".format(tstep, its))
            else:
                raise RuntimeError("timestep {}: the coupled CH-NS solve failed (PETSc "
                                   "reason {})".format(tstep, reason))
        telemetry.current.count("chns", its or 1)
        if self.method == "iterative":
            telemetry.current.count("chns_pf", self.pc.its[0])
            telemetry.current.count("chns_ns", self.pc.its[1])

        from_nest(self.X, self.isets, [x_[k] for k in keys])
        finish_solve(x_, keys, vp.p_null, p0, dp_)
        return its
