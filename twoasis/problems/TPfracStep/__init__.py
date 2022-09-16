__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from twoasis.problems import *
import ufl

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
    solver="Naive",  # "IPCS", "BDF"
    bdf_order=2,
    constant_poisson_coefficient=False,

    # Phase field parameters
    phase_field_degree=1,
    chemical_potential_degree=1,
    M=0.001,
    epsilon=0.04,
    sigma=1.0,

    # Parameters used to tweak solver
    max_iter=1,                 # Number of inner pressure velocity iterations on timestep
    max_error=1e-6,             # Tolerance for inner iterations (pressure velocity iterations)
    iters_on_first_timestep=2,  # Number of iterations on first timestep
    use_krylov_solvers=True,    # Otherwise use LU-solver
    print_intermediate_info=10,
    print_velocity_pressure_convergence=False,

    # Parameters used to tweak output
    plot_interval=10,
    checkpoint=10,       # Overwrite solution in Checkpoint folder each checkpoint
    save_step=10,        # Store solution each save_step
    restart_folder=None, # If restarting solution, set the folder holding the solution to start from here
    output_timeseries_as_vector=True,  # Store velocity as vector in Timeseries

    # Stop simulations cleanly after the given number of seconds
    killtime=None,

    # Choose LES model and set default parameters
    # NoModel, Smagorinsky, Wale, DynamicLagrangian, ScaleDepDynamicLagrangian
    les_model='NoModel',

    # LES model parameters
    Smagorinsky=dict(Cs=0.1677),              # Standard Cs, same as OpenFOAM
    Wale=dict(Cw=0.325),
    DynamicSmagorinsky=dict(Cs_comp_step=1),  # Time step interval for Cs to be recomputed
    KineticEnergySGS=dict(Ck=0.08, Ce=1.05),

    # Choose Non-Newtonian model and set default parameters
    # NoModel, ModifiedCross
    nn_model='NoModel',

    # Non-Newtonian model parameters
    ModifiedCross = dict(
        lam=3.736,      # s
        m_param=2.406,  # for Non-Newtonian model
        a_param=0.34,   # for Non-Newtonian model
        mu_inf=0.00372, # Pa-s for non-Newtonian model
        mu_o=0.09,      # Pa-s for non-Newtonian model
        rho=1085        # kg/m^3
        ),

    # Parameter set when enabling test mode
    testing=False,

    # Solver parameters that will be transferred to dolfins parameters['krylov_solver']
    krylov_solvers=dict(
        monitor_convergence=False,
        report=False,
        error_on_nonconvergence=False,
        nonzero_initial_guess=True,
        maximum_iterations=200,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-8),

    # Velocity update
    velocity_update_solver=dict(
        method='default',  # "lumping", "gradient_matrix"
        solver_type='cg',
        preconditioner_type='jacobi',
        low_memory_version=False),

    velocity_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi'),

    pressure_krylov_solver=dict(
        solver_type='gmres',
        preconditioner_type='hypre_amg'),

    scalar_krylov_solver=dict(
        solver_type='bicgstab',
        preconditioner_type='jacobi'),

    phase_field_krylov_solver=dict(
        solver_type='gmres',
        preconditioner_type='jacobi')
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
    """Called at end of a timestep."""
    pass


def phase_field_hook(**NS_namespace):
    """Called prior to scalar solve."""
    pass

def mark_subdomains(**NS_namespace):
    return dict()

def contact_angles(**NS_namespace):
    return []