"""Adaptive time step for the BDF solver (option dt_adapt; off when None).

After each step the controller compares the new solution with its extrapolation from
the two levels before it, x* = beta0 x^n + beta1 x^(n-1) (the phi* and u_adv the step
was linearised about). Their difference, ~ dt (dt + dt_1)/2 x'', measures how far the
solution bent away from a straight line over the step; it is also the error of the
linearisation about phi*. Two measures, each 1 at its tolerance:

    e_phi = sqrt(2) epsilon max|phi^(n+1) - phi*| / tol_x
    e_u   = max_i max|u_i^(n+1) - u_adv_i| / (atol_u + rtol_u max_i max|u_i^(n+1)|)

phi = tanh(d / (sqrt(2) epsilon)) across the interface, so sqrt(2) epsilon |dphi| is
the interface's displacement (a length, whatever the domain's size); atol_u is the
velocity below which the velocity's error does not matter (a fraction of the slowest
flow of interest, e.g. 0.01 Ca). With e = max(e_phi, e_u), the next step is dt times

    f = min(grow, max(shrink, safety e^(-1/2))),

kept at dt when f is within `deadband`, and clipped to [dt_min, dt_max]. The step ratio
stays below grow = 2 < 1 + sqrt(2), BDF2's zero-stability bound.

While e <= fast_cut, f is also held within [1/ramp, ramp]: dt changes gradually. Every
change of dt is itself a disturbance. Where part of the solution follows the rest
quasi-statically (a meniscus that relaxes faster than dt, its position slaved to the
driving pressure), that part lags its quasi-static state by O(dt^2); a step of dt' after
steps of dt must move it by ~(dt^2 - dt'^2) within one step, and the velocity carries
the move, a disturbance ~ dt (1 - r^2)/r for r = dt'/dt that the velocity measure then
reads as error. Large changes thus make the error they react to, and dt see-saws; small
ones keep it below the tolerance. Above fast_cut the error is the solution's own (a
Haines jump starting), and dt is cut as fast as shrink allows. Only the interface's
measure is tested against fast_cut (fast_cut_on "e_phi"; "e" tests e): the velocity's is
the one a change of dt disturbs most, and a fast cut on it disturbs the velocity again,
whereas a jump's onset shows in both. The last steps before T
are evened out: when less than two steps are left, the next is half of what is left.
Steps are not rejected for their error; a BDF1 step (the first, without a level before
it) is not measured.

A step whose solve fails (a Krylov solve that diverges, or a solution that is not
finite) is redone from the same levels at retry_factor times its dt, at most
max_retries times, and not below dt_min. The steps after it then stay below half the
dt that failed, a ceiling that rises by ceiling_rise each step after, so that dt does
not grow straight back into the failure.
"""
import math

import numpy as np

from twoasis.backend import as_backend_type, owned_dofs

DEFAULTS = dict(
    tol_x=None,         # interface displacement per step; None: 1e-3 epsilon
    rtol_u=1e-3,
    atol_u=1e-7,
    dt_min=None,        # None: 1e-3 times the first dt
    dt_max=None,        # None: no upper bound
    safety=0.9,
    grow=2.0,
    shrink=0.5,
    deadband=(0.95, 1.05),
    ramp=1.1,           # the step ratio while e <= fast_cut: within [1/ramp, ramp]
    fast_cut=2.0,
    fast_cut_on="e_phi",  # the measure tested against fast_cut: "e_phi" or "e"
    retry_factor=0.25,
    max_retries=4,
    ceiling_rise=1.25,
)


class StepFailed(RuntimeError):
    """A step whose solution is not finite."""


def _petsc(x):
    return as_backend_type(x).vec()


class Controller:
    """The adaptive time step: `error` after a step, `next_dt` from it, `retry_dt`
    after a failed one."""

    def __init__(self, cfg, dt, epsilon, W, comm):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError("dt_adapt: unknown keys {}".format(sorted(unknown)))
        self.cfg = dict(DEFAULTS, **cfg)
        if self.cfg["tol_x"] is None:
            self.cfg["tol_x"] = 1e-3 * epsilon
        if self.cfg["dt_min"] is None:
            self.cfg["dt_min"] = 1e-3 * dt
        self.epsilon = epsilon
        self.comm = comm
        self._phi_global = owned_dofs(W, 0)
        self._phi_local = None
        if self.cfg["fast_cut_on"] not in ("e_phi", "e"):
            raise ValueError("dt_adapt fast_cut_on {!r}: 'e_phi' or 'e'".format(
                self.cfg["fast_cut_on"]))
        self.last = dict(e=float("nan"), e_phi=float("nan"), e_u=float("nan"))
        self.ceiling = None     # after a failed step: the largest dt to take
        self.wanted = None      # next_dt's: the step the error allows before the limits

    def _phi_index(self, vec):
        if self._phi_local is None:
            r0 = vec.getOwnershipRange()[0]
            self._phi_local = np.asarray(self._phi_global, dtype=int) - r0
        return self._phi_local

    def _max(self, value):
        from mpi4py import MPI
        return self.comm.allreduce(float(value), op=MPI.MAX)

    def error(self, x_, x_1, x_2, beta, u_components):
        """e, e_phi and e_u of the step just taken (beta: its extrapolation weights)."""
        b0, b1 = float(beta[0]), float(beta[1])
        new = _petsc(x_['phig']).getArray(readonly=True)
        old1 = _petsc(x_1['phig']).getArray(readonly=True)
        old2 = _petsc(x_2['phig']).getArray(readonly=True)
        i = self._phi_index(_petsc(x_['phig']))
        dphi = np.abs(new[i] - b0 * old1[i] - b1 * old2[i])
        dphi_max = self._max(dphi.max() if dphi.size else 0.0)
        du = umax = 0.0
        for ui in u_components:
            a = _petsc(x_[ui]).getArray(readonly=True)
            a1 = _petsc(x_1[ui]).getArray(readonly=True)
            a2 = _petsc(x_2[ui]).getArray(readonly=True)
            if a.size:
                du = max(du, float(np.abs(a - b0 * a1 - b1 * a2).max()))
                umax = max(umax, float(np.abs(a).max()))
        du, umax = self._max(du), self._max(umax)
        c = self.cfg
        e_phi = math.sqrt(2.0) * self.epsilon * dphi_max / c["tol_x"]
        e_u = du / (c["atol_u"] + c["rtol_u"] * umax)
        self.last.update(e=max(e_phi, e_u), e_phi=e_phi, e_u=e_u)
        return self.last["e"]

    def next_dt(self, dt, e, t, T, limit=None):
        """The step after one of size dt with error e (None: not measured). Also sets
        `wanted`, the step the error allows before the limits (safety e^(-1/2) dt;
        None when not measured). `limit(new, wanted)`, when given, returns the step to
        take instead of new, before the last steps are evened out (switch.Switch.next_dt:
        the split step's cap)."""
        c = self.cfg
        new = dt
        self.wanted = None
        if e is not None and np.isfinite(e):
            self.wanted = dt * c["safety"] * e ** -0.5 if e > 0 else math.inf
            f = c["safety"] * e ** -0.5 if e > 0 else c["grow"]
            f = min(c["grow"], max(c["shrink"], f))
            if (e if c["fast_cut_on"] == "e" else self.last["e_phi"]) <= c["fast_cut"]:
                f = min(c["ramp"], max(1.0 / c["ramp"], f))
            if not c["deadband"][0] <= f <= c["deadband"][1]:
                new = dt * f
        if self.ceiling is not None:
            new = min(new, self.ceiling)
            self.ceiling *= c["ceiling_rise"]
        new = max(c["dt_min"], new)
        if c["dt_max"] is not None:
            new = min(c["dt_max"], new)
        if limit is not None:
            new = limit(new, self.wanted)
        left = T - t
        if 0 < left < new * (1 + 1e-9):
            new = left
        elif 0 < left < 2 * new:
            new = 0.5 * left
        return new

    def check_finite(self, x_, keys):
        """Raise StepFailed unless every entry of x_[k] (k in keys) is finite."""
        bad = 0
        for k in keys:
            a = _petsc(x_[k]).getArray(readonly=True)
            bad += int(a.size and not np.isfinite(a).all())
        if self._max(bad) > 0:
            raise StepFailed("the step's solution is not finite")

    def retry_dt(self, dt, attempt, err):
        """The dt to redo a failed step with, or raise err when out of retries."""
        c = self.cfg
        new = dt * c["retry_factor"]
        if attempt > c["max_retries"] or new < c["dt_min"]:
            raise err
        self.ceiling = 0.5 * dt if self.ceiling is None else min(self.ceiling, 0.5 * dt)
        return new


def retryable(err):
    """Whether a failure is one a smaller step may cure: a solve that failed or
    diverged, or a solution that is not finite."""
    if isinstance(err, StepFailed):
        return True
    msg = str(err)
    return isinstance(err, RuntimeError) and any(
        s in msg for s in ("solve failed", "diverge", "did not converge", "failed to converge",
                           "DIVERGED"))
