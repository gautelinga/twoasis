__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from dolfin import Constant

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
    quadrature_degree=None,        # quadrature degree for all forms; None means 2*velocity_degree + 1

    # Phase field parameters
    phase_field_degree=None,         # degree of phi; None means velocity_degree
    chemical_potential_degree=None,  # degree of g; None means phase_field_degree
    phase_field_fieldsplit=False,  # use a Schur fieldsplit preconditioner for the (phi, g) solve

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
    max_error=1e-6,             # Tolerance for inner iterations (pressure velocity iterations):
                                # stop when the velocity changes by less than this (l2 norm
    max_error_rel=0.,           # of the dofs), or by less than max_error_rel times its norm
                                # (e.g. 1e-3: integral statistics change by ~1e-5 or less;
                                # not for small velocities measured pointwise)
    iters_on_first_timestep=2,  # Number of iterations on each of the first 10 timesteps
    use_krylov_solvers=True,    # Otherwise use LU-solver
    print_intermediate_info=10,
    print_velocity_pressure_convergence=False,

    # Parameters used to tweak output
    plot_interval=10,
    stat_interval=10,       # Write statistics (Stats/) each stat_interval timesteps
    timestamps_interval=10, # ... and fields to Timestamps/, in problems that write them
    checkpoint=10,       # Overwrite solution in Checkpoint folder each checkpoint
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

    # Velocity update
    velocity_update_solver=dict(
        method='default',  # consistent mass matrix; 'lumping' lumps it for P1 (P2 and higher
                           # always use the consistent one)
        solver_type='cg',
        preconditioner_type='jacobi',
        low_memory_version=False),

    velocity_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi'),

    pressure_krylov_solver=dict(
        solver_type='gmres',
        preconditioner_type='hypre_amg'),  # scales better than 'petsc_amg' on large meshes

    scalar_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi'),

    phase_field_krylov_solver=dict(
        solver_type='gmres',
        preconditioner_type='jacobi')
)

# Default relative tolerance and iteration limit for each Krylov solver, used
# when neither the solver's own settings nor krylov_solvers give a value.
KRYLOV_TOLERANCES = dict(
    # Loose: the outer pressure-velocity iteration controls the accuracy.
    velocity_krylov_solver=dict(relative_tolerance=1e-6, maximum_iterations=300),
    velocity_update_solver=dict(relative_tolerance=1e-8, maximum_iterations=200),
    # AMG converges in a few iterations; needing more means it has stalled.
    pressure_krylov_solver=dict(relative_tolerance=1e-6, maximum_iterations=100),
    # Tight, since it sets the accuracy of the interface. Jacobi is a weak
    # preconditioner for this system; with phase_field_fieldsplit ~100
    # iterations are enough.
    phase_field_krylov_solver=dict(relative_tolerance=1e-8, maximum_iterations=1000),
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
    return Constant(tuple(-f for f in F0) if F0 is not None else (0.,) * mesh.geometry().dim())


def acceleration(g0, mesh, **NS_namespace):
    """Body acceleration g0 (e.g. gravity)."""
    return Constant(tuple(g0) if g0 is not None else (0.,) * mesh.geometry().dim())
