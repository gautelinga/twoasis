__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from dolfin import Constant, Mesh, MPI, parameters, Timer
import inspect
import pickle
import difflib
from os import path, sysconf
from collections import defaultdict


def getMemoryUsage(rss=True):
    """This process's resident (rss) or virtual memory in MB, from /proc (Linux;
    0 elsewhere), without starting a process under MPI."""
    try:
        with open("/proc/self/statm") as f:
            size, resident = (int(v) for v in f.read().split()[:2])
    except OSError:
        return 0.
    return (resident if rss else size) * sysconf("SC_PAGE_SIZE") / 2**20


parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
#parameters["form_compiler"]["quadrature_degree"] = 4
#parameters["form_compiler"]["cache_dir"] = "instant"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"
#parameters["mesh_partitioner"] = "ParMETIS"
#parameters["form_compiler"].add("no_ferari", True)
#set_log_active(False)

# Default parameters for all solvers
NS_parameters = dict(
    nu=0.01,             # Kinematic viscosity
    folder='results',    # Relative folder for storing results
    velocity_degree=2,   # default velocity degree
    pressure_degree=1    # default pressure degree
)

NS_expressions = {}

constrained_domain = None

# To solve for scalars provide a list like ['scalar1', 'scalar2']
scalar_components = []

# With diffusivities given as a Schmidt number defined by:
#   Schmidt = nu / D (= momentum diffusivity / mass diffusivity)
Schmidt = defaultdict(lambda: 1.)
Schmidt_T = defaultdict(lambda: 0.7)  # Turbulent Schmidt number (LES)

Scalar = defaultdict(lambda: dict(Schmidt=1.0,
                                  family="CG",
                                  degree=1))

# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0.
RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"


def info_blue(s, check=True):
    if MPI.rank(MPI.comm_world) == 0 and check:
        print(BLUE % s)


def info_green(s, check=True):
    if MPI.rank(MPI.comm_world) == 0 and check:
        print(GREEN % s)


def info_red(s, check=True):
    if MPI.rank(MPI.comm_world) == 0 and check:
        print(RED % s)


class OasisTimer(Timer):
    def __init__(self, task, verbose=False):
        Timer.__init__(self, task)
        info_blue(task, verbose)


class OasisMemoryUsage:
    def __init__(self, s):
        self.memory = 0
        self.memory_vm = 0
        self(s)

    def __call__(self, s, verbose=False):
        self.prev = self.memory
        self.prev_vm = self.memory_vm
        self.memory = MPI.sum(MPI.comm_world, getMemoryUsage())
        self.memory_vm = MPI.sum(MPI.comm_world, getMemoryUsage(False))
        if MPI.rank(MPI.comm_world) == 0 and verbose:
            info_blue('{0:26s}  {1:10d} MB {2:10d} MB {3:10d} MB {4:10d} MB'.format(s,
                        int(self.memory - self.prev), int(self.memory),
                        int(self.memory_vm - self.prev_vm), int(self.memory_vm)))


# Print memory use up til now
initial_memory_use = getMemoryUsage()
oasis_memory = OasisMemoryUsage('Start')


# Convenience functions
def initialize(**NS_namespace):
    """Initialize solution."""
    pass


def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)


def phase_field_source(**NS_namespace):
    """Phase field sources (tuple) -- mainly for testing purposes."""
    return None

def scalar_hook(**NS_namespace):
    """Called prior to scalar solve."""
    pass


def scalar_source(scalar_components, **NS_namespace):
    """Return a dictionary of scalar sources."""
    return dict((ci, Constant(0)) for ci in scalar_components)


def pre_solve_hook(**NS_namespace):
    """Called just prior to entering time-loop. Must return a dictionary."""
    return {}


def theend_hook(**NS_namespace):
    """Called at the very end."""
    pass


def problem_parameters(**NS_namespace):
    """Updates problem specific parameters, and handles restart"""
    pass


def checkpoint_parameters(restart_folder):
    """The parameters stored with the checkpoint in restart_folder (params.dat),
    after putting back the checkpoint of an interrupted swap (recover_checkpoint)."""
    from twoasis.common.io import recover_checkpoint
    if MPI.rank(MPI.comm_world) == 0:
        recover_checkpoint(restart_folder)
    MPI.barrier(MPI.comm_world)
    with open(path.join(restart_folder, "params.dat"), "rb") as f:
        return pickle.load(f)


def post_import_problem(NS_parameters, mesh, commandline_kwargs,
                        NS_expressions, constrained_domain=None, **NS_namespace):
    """Called after importing from problem.

    mesh and constrained_domain may be functions of the parameters; they are
    called here, once the command line has been merged into NS_parameters, so
    that e.g. a domain length given on the command line sets both the mesh and
    its periodicity. constrained_domain is also called with the mesh.
    """

    # On a restart, start from the parameters stored with the checkpoint (they
    # include t and tstep); the command line can still change any of them.
    if commandline_kwargs.get("restart_folder") is not None:
        restart_folder = path.abspath(commandline_kwargs["restart_folder"])
        NS_parameters.update(checkpoint_parameters(restart_folder))
        commandline_kwargs = dict(commandline_kwargs, restart_folder=restart_folder)

    # A key that is neither a default nor set by the problem is a mistake (a
    # misspelt name would otherwise leave the default in place, silently).
    unknown = sorted(k for k in commandline_kwargs if k not in NS_parameters and k != "problem")
    if unknown:
        raise ValueError("unknown parameter{} {}: not a default nor set by the problem{}".format(
            "s" if len(unknown) > 1 else "", ", ".join(unknown),
            "".join(" ({}: did you mean {}?)".format(k, " or ".join(m))
                    for k in unknown for m in [difflib.get_close_matches(k, NS_parameters)] if m)))

    # Update NS_parameters with all parameters modified through command line; a
    # dict updates the default dict (a solver's settings, say).
    for key, val in commandline_kwargs.items():
        if isinstance(val, dict) and isinstance(NS_parameters.get(key), dict):
            NS_parameters[key].update(val)
        else:
            NS_parameters[key] = val

    # If the mesh is a callable function, then create the mesh here.
    if callable(mesh):
        mesh = mesh(**NS_parameters)

    assert(isinstance(mesh, Mesh))

    constrained_domain = NS_expressions.pop("constrained_domain", constrained_domain)
    if inspect.isfunction(constrained_domain):
        constrained_domain = constrained_domain(mesh=mesh, **NS_parameters)

    # dolfin's partition can leave a process holding a vertex of a periodic face
    # without knowing that it is periodic, and the periodic dofmap then gives it a
    # wrong dof. femtools moves the few cells concerned to another process (most
    # partitions need no move, and the mesh is then used as it is).
    if MPI.size(mesh.mpi_comm()) > 1 and hasattr(constrained_domain, "upper"):
        from femtools.periodic_partition import repaired
        mesh = repaired(mesh, constrained_domain.upper, constrained_domain.periodic,
                        mesh.mpi_comm(), log=info_blue,
                        tol=getattr(constrained_domain, "tol", None))

    # Returned dictionary to be updated in the NS namespace
    d = dict(mesh=mesh, constrained_domain=constrained_domain)
    d.update(NS_parameters)
    d.update(NS_expressions)
    return d
