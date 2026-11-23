__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from twoasis.backend import Constant, geometric_dimension

from twoasis.problems import NS_parameters

# Default parameters NSfracStep solver
NS_parameters.update(
    # Physical constants and solver parameters
    t=0.0,               # Time
    tstep=0,             # Timestep
    T=1.0,               # End time
    dt=0.01,             # Time interval on each timestep

    # Some discretization options
    # Use Adams Bashforth projection as first estimate for pressure on new timestep
    AB_projection_pressure=False,
    solver="BDF",  # or "IPCS", or "Naive" (a plain, slow version of IPCS for validation)
    bdf_order=2,
    # First BDF step: True = first order, False = second order, None = second order
    # only if initialize() (or the checkpoint) provides the solution at t - 2*dt.
    initial_step_1st_order=None,
    constant_poisson_coefficient=False,
    # Rotational pressure update (BDF and IPCS): after each pressure solve,
    # p = p* + phi - mu*div(u*), with u* the tentative velocity and phi the
    # pressure increment. It removes most of the boundary layer that the pressure
    # equation's Neumann condition leaves at walls and prescribed velocities, and
    # makes the inner iterations converge where they otherwise stall. The Naive
    # solver has none.
    rotational=True,
    # Convection of the velocity in the skew-symmetric form, (w.grad)u + 1/2 (div w) u
    # with w the convecting velocity: w is extrapolated from discretely
    # divergence-free fields, so it is not exactly divergence-free, and the plain form
    # (w.grad)u (skew_convection=False) lets convection add kinetic energy. The
    # stability argument in docs/notes/coupled-velocity-phase-field-stability.tex
    # needs the skew form.
    skew_convection=True,
    # Anderson acceleration of the inner passes on the pressure, with this many
    # earlier passes (0: off).
    anderson_depth=0,
    # For problems whose inner passes iterate: a tentative-velocity solve stops
    # once its residual has fallen by this factor (e.g. 0.1) from the one it
    # starts from (0: the solver's own relative tolerance). It pays where the
    # velocity solves dominate. With anderson_depth it takes more passes.
    velocity_tolerance_factor=0,
    # Rebuild the velocity and pressure preconditioners only every this many
    # timesteps, reusing them for the timesteps between (1: every timestep). The
    # pressure's is kept for the whole run anyway when its matrix is constant.
    velocity_preconditioner_steps=1,
    pressure_preconditioner_steps=1,
    # The phase field's preconditioner (also the coupled CH-NS step's S solve): rebuilt
    # every that many timesteps, and whenever the BDF coefficient alpha0 has changed by
    # more than 10 % since it was built (an adaptive dt)
    phase_field_preconditioner_steps=1,
    # The BDF solver's first guess for the split phase-field solve and the coupled solve
    # (velocity_pressure_solve="coupled"): "previous" (the last step's solution) or
    # "extrapolated" (beta0 x^n + beta1 x^(n-1), the BDF extrapolation, second order).
    # The stopping tests are relative to the initial residual, so a closer guess ends
    # closer to the exact solve at the same tolerance; it saves iterations only where
    # the absolute tolerance binds (slow flow) or with a looser relative tolerance
    initial_guess="previous",
    # Stop the run with an error when the velocity change of the inner passes has
    # grown this many passes in a row while larger than the first pass's change: the
    # passes diverge, and the step's result would be garbage (0: never). The last
    # checkpoint is kept.
    diverging_passes=5,
    # How a BDF step solves for its velocity and pressure: "passes", the
    # pressure-correction passes above (max_iter of them, or until max_error), or
    # "coupled", the velocity and pressure in one system to coupled_krylov_solver's
    # tolerance (twoasis/solvers/TPfracStep/coupled.py): the step the passes converge
    # to, in a number of iterations meant not to grow as the mesh is refined.
    # "coupled" needs an inf-sup stable pair, velocity_degree > pressure_degree, and
    # leaves rotational and the passes' settings unused.
    velocity_pressure_solve="passes",
    # How a BDF step couples the phase field to the flow: "split", the (phi, g) system
    # first, advected by the extrapolated velocity, then the flow with the capillary
    # force of the new phase field; "coupled", the phase field, chemical potential,
    # velocity and pressure in one linear system (twoasis/solvers/TPfracStep/
    # monolithic.py), solved by chns_solver's method; or "auto", split or coupled each
    # step by dt (twoasis/solvers/TPfracStep/switch.py), the split step's dt capped at
    # dt_split_max (required), its largest stable dt. "coupled" and "auto" solve the
    # velocity and pressure together (velocity_pressure_solve="coupled") and need
    # solver="BDF" and no dual mesh.
    chns_solve="split",
    dt_split_max=None,
    # Adaptive time step (solvers/TPfracStep/adapt.py): None (dt fixed) or a dict of
    # its settings, e.g. dict(tol_x=4e-5, rtol_u=1e-3, atol_u=1e-6, dt_max=25.6)
    dt_adapt=None,
    # The coupled CH-NS step's solver: settings that differ from monolithic.DEFAULTS
    # (twoasis/solvers/TPfracStep/monolithic.py), e.g. dict(method="direct"). method
    # "iterative" (the default: FGMRES with a block preconditioner; inner_tolerance for
    # its two inner solves, inner "lu", "lu_ns" or "lu_pf" for diagnosis) or "direct"
    # (MUMPS LU, for small problems).
    chns_solver={},
    backend="auto",      # dolfin, dolfinx, or auto: a restart's checkpoint's, else dolfin
                         # (chosen when twoasis.backend is imported; see its docstring)
    telemetry=False,     # write Stats/telemetry.jsonl: per-step times, Krylov iterations and
                         # preconditioner setups (twoasis/common/telemetry.py)
    quadrature_degree=None,        # quadrature degree for all forms; None means 2*velocity_degree + 1

    # Phase field parameters
    phase_field_degree=None,         # degree of phi; None means pressure_degree
    chemical_potential_degree=None,  # degree of g; None means phase_field_degree
    phase_refine=0,      # the dual mesh: solve the phase field on the flow mesh refined
                         # uniformly this many times (0: on the flow mesh itself); see
                         # twoasis/common/dual.py

    M=0.001,
    epsilon=0.04,
    sigma=1.0,
    theta=None,          # contact angle on the boundaries marked 1 (none if None),
                         # measured through the phase phi = +1

    # Body forces: the driving force per volume F0 (minus the mean pressure
    # gradient) and the acceleration (gravity) g0, one component per dimension;
    # None means zero.
    F0=None,
    g0=None,

    # Parameters used to tweak solver
    max_iter=1,                 # Number of inner pressure velocity iterations on timestep
    max_error=1e-8,             # Tolerance for inner iterations (pressure velocity iterations):
                                # stop when the velocity changes by less than this (its L2
    max_error_rel=0.,           # norm over the domain), or by less than max_error_rel times
                                # the velocity's L2 norm
    inner_norm='L2',            # the norm max_error is in (recorded with checkpoints)
                                # (e.g. 1e-3: integral statistics change by ~1e-5 or less;
                                # not for small velocities measured pointwise)
    iters_on_first_timestep=2,  # Number of iterations on each of the first 10 timesteps
    print_intermediate_info=10,
    print_velocity_pressure_convergence=False,

    # Parameters used to tweak output
    plot_interval=10,
    stat_interval=10,       # Write statistics (Stats/) each stat_interval timesteps
    timestamps_interval=10, # ... and fields to Timestamps/, in problems that write them
    checkpoint=10,       # Overwrite solution in Checkpoint folder each checkpoint
    checkpoint_minutes=None,  # ... and also once this many minutes of wall time have
                              # passed since the last checkpoint (None: by steps only)
    save_step=10,        # Store solution each save_step
    restart_folder=None, # If restarting solution, set the folder holding the solution to start from here

    # Stop simulations cleanly after the given number of seconds
    killtime=None,

    # Parameter set when enabling test mode
    testing=False,

    # Settings for all Krylov solvers. Each solver below can also set its own
    # relative_tolerance and maximum_iterations. A value set for one solver wins
    # over a value set here, and a value set here wins over the solver's default
    # in KRYLOV_TOLERANCES at the end of this file.
    krylov_solvers=dict(
        monitor_convergence=False,
        report=False,
        error_on_nonconvergence=True,   # stop the run if a linear solve does not converge
        nonzero_initial_guess=True,
        absolute_tolerance=1e-12),      # small, so the relative tolerance decides when to stop

    # Each solver's solver_type and preconditioner_type take PETSc's names
    # (dolfin's, such as 'bicgstab' or 'hypre_amg', are translated), and its
    # petsc_options any further PETSc options for that solver, e.g.
    # velocity_krylov_solver='{"petsc_options": {"sub_pc_type": "icc"}}'.
    # 'bjacobi' is block Jacobi with ILU(0) on each process's block.

    # Velocity update
    velocity_update_solver=dict(
        method='default',  # consistent mass matrix; 'lumping' lumps it for P1 (P2 and higher
                           # always use the consistent one)
        solver_type='cg',
        preconditioner_type='jacobi',  # optimal for a mass matrix, on any number of processes
        low_memory_version=False),

    velocity_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='auto'),  # 'hypre_amg' where the viscous term dominates
                                      # (dt*nu/h^2 above VISCOUS_DOMINATED, h the velocity's
                                      # smallest node spacing, nu the largest mu/rho), where
                                      # block Jacobi's iterations grow as the mesh is refined;
                                      # else 'bjacobi', which is cheaper there

    pressure_krylov_solver=dict(
        solver_type='gmres',
        preconditioner_type='hypre_amg'),  # scales better than 'petsc_amg' on large meshes;
                                           # femtools.ksp.BOOMERAMG_3D's settings in 3D
                                           # (as any BoomerAMG has there)

    phase_field_krylov_solver=dict(
        solver_type='fgmres',
        preconditioner_type='schur'),  # PhaseFieldSchur: iterations independent of h and dt

    # velocity_pressure_solve="coupled": FGMRES with a block preconditioner, one
    # BoomerAMG cycle per velocity component and on the pressure matrix. Its
    # petsc_options take the outer solver's PETSc options, and the inner ones under
    # u_ and p_ (e.g. {"u_pc_hypre_boomeramg_strong_threshold": 0.5}).
    coupled_krylov_solver=dict(
        solver_type='fgmres')
)

# Default relative tolerance and iteration limit for each Krylov solver, used
# when neither the solver's own settings nor krylov_solvers give a value.
KRYLOV_TOLERANCES = dict(
    # Loose: the outer pressure-velocity iteration controls the accuracy.
    velocity_krylov_solver=dict(relative_tolerance=1e-6, maximum_iterations=300),
    velocity_update_solver=dict(relative_tolerance=1e-8, maximum_iterations=200),
    # AMG converges in a few iterations; needing more means it has stalled.
    pressure_krylov_solver=dict(relative_tolerance=1e-6, maximum_iterations=100),
    # Tight, since it sets the accuracy of the interface. This one and the coupled
    # solver's are relative to the initial residual, the step's change (INITIAL_RESIDUAL
    # in twoasis/solvers/TPfracStep/__init__.py), not to the right-hand side.
    phase_field_krylov_solver=dict(relative_tolerance=1e-8, maximum_iterations=1000),
    # Tight: it sets the accuracy of the step's velocity and pressure.
    coupled_krylov_solver=dict(relative_tolerance=1e-8, maximum_iterations=200),
    # The coupled CH-NS step's iterative solver (chns_solver method "iterative"), likewise.
    chns_solver=dict(relative_tolerance=1e-8, maximum_iterations=200),
)

def velocity_tentative_hook(**NS_namespace):
    """Called just prior to solving for tentative velocity."""
    pass


def pressure_hook(**NS_namespace):
    """Called prior to pressure solve."""
    pass


def start_timestep_hook(**NS_namespace):
    """Called at start of new timestep"""
    pass


def temporal_hook(**NS_namespace):
    """Called at the end of each timestep. May return a dict of values to update
    in the solver namespace; stop=True in it ends the run after this step."""
    pass


def phase_field_hook(**NS_namespace):
    """Called prior to scalar solve."""
    pass

def mark_subdomains(**NS_namespace):
    return dict()

def contact_angles(theta, **NS_namespace):
    """Contact angles as a list of (angle, boundary marker): theta on the
    boundaries marked 1, if theta is set. An angle is measured through the
    phase phi = +1, so theta < pi/2 means that phase wets the boundary. A
    Function in place of the angle gives cos(theta) pointwise."""
    return [] if theta is None else [(theta, 1)]


def average_pressure_gradient(F0, mesh, **NS_namespace):
    """The mean pressure gradient, -F0: F0 is the force per volume it drives the
    flow with."""
    return Constant(tuple(-f for f in F0) if F0 is not None
                    else (0.,) * geometric_dimension(mesh))


def acceleration(g0, mesh, **NS_namespace):
    """Body acceleration g0 (e.g. gravity)."""
    return Constant(tuple(g0) if g0 is not None else (0.,) * geometric_dimension(mesh))
