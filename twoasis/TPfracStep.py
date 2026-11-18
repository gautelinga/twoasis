#!/usr/bin/env python

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

"""
The main script of twoasis: it imports the problem (twoasis/problems/TPfracStep/<problem>.py)
and the solver (twoasis/solvers/TPfracStep/<solver>.py), builds one namespace from their
parameters and functions, and runs the time loop. Each step solves the Cahn-Hilliard
equation for the phase field (phi, g), then the Navier-Stokes equations by a fractional
step (tentative velocity, pressure correction, velocity update), with optional inner
iterations. Problem hooks (initialize, pre_solve_hook, temporal_hook, ...) are called
with the whole namespace.
"""
import importlib
from os import makedirs, path

from numpy import array
from twoasis.backend import name as backend_name
from twoasis.backend import (
    as_vector, assemble, BoundaryMarkers, check_periodic_pairing, collect_garbage, DOLFIN_EPS,
    dx, FacetNormal, FiniteElement, Function, FunctionSpace, geometric_dimension, list_timings,
    Measure, mesh_comm, MixedElement, MPI, parameters, set_default_mesh, split, TestFunction,
    TestFunctions, TimingClear, TimingType, TrialFunction, TrialFunctions, Vector, XDMFFile)

import twoasis.common
import twoasis.common.telemetry
import twoasis.problems
import twoasis.problems.TPfracStep
import twoasis.problems.TPfracStep.shared
import twoasis.solvers.TPfracStep
from twoasis.common import (
    create_initial_folders, init_from_restart, merge_visualization_files, parse_command_line,
    save_solution)
import twoasis.common.dual
from twoasis.common.utilities import Coefficient, phi_g
from twoasis.problems import info_green, info_red, initial_memory_use, oasis_memory, OasisTimer


def public_names(module):
    return {name: value for name, value in vars(module).items() if not name.startswith("_")}


commandline_kwargs = parse_command_line()

# Find the problem module
# The problem named on the command line, or on a restart the checkpoint's.
default_problem = "TaylorBubble2D"
problemname = commandline_kwargs.get('problem')
if problemname is None and commandline_kwargs.get('restart_folder') is not None:
    from twoasis.problems import checkpoint_parameters
    problemname = checkpoint_parameters(
        path.abspath(commandline_kwargs['restart_folder'])).get('problem')
if problemname is None:
    problemname = default_problem
problemspec = importlib.util.find_spec('.'.join(('twoasis.problems.TPfracStep', problemname)))
if problemspec is None:
    problemspec = importlib.util.find_spec(problemname)
if problemspec is None:
    raise RuntimeError(problemname + ' not found')

# Import the problem module
if MPI.rank(MPI.comm_world) == 0:
    print('Backend: ' + backend_name)
    print('Importing problem module '+problemname+':\n'+problemspec.origin)
problemmod = importlib.util.module_from_spec(problemspec)
problemspec.loader.exec_module(problemmod)

# Everything below runs in one namespace that is passed to the hooks and solver
# functions as keyword arguments. Start it with the helpers in twoasis.common and
# the default hooks and parameters, then let the problem module override them.
vars().update({name: getattr(twoasis.common, name) for name in twoasis.common.__all__})
vars().update(public_names(twoasis.problems))
vars().update(public_names(twoasis.problems.TPfracStep))
vars().update(public_names(problemmod))

# Update problem specific parameters
problem_parameters(**vars())

# Update current namespace with NS_parameters and commandline_kwargs ++
vars().update(post_import_problem(**vars()))
# The backend this run is on (backend=auto resolved), recorded with the parameters.
backend = NS_parameters["backend"] = backend_name

if scalar_components:
    raise NotImplementedError("passive scalars (scalar_components) are not supported by the "
                              "TPfracStep solvers")

# Use one quadrature degree for all forms. The form compiler's automatic estimate
# is very high for forms with variable coefficients (density, viscosity, the
# phase-field potential), which makes assembly slow for little gain in accuracy.
if quadrature_degree is None:
    quadrature_degree = 2 * velocity_degree + 1
parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"

# Import chosen functionality from solvers
# Each name in the solver's __all__ comes from the solver module, or else from
# the shared functions in twoasis.solvers.TPfracStep.
solver = importlib.import_module('.'.join(('twoasis.solvers.TPfracStep', solver)))
vars().update({name: getattr(solver, name) if hasattr(solver, name)
               else getattr(twoasis.solvers.TPfracStep, name) for name in solver.__all__})
if rotational and not hasattr(solver, "pressure_update"):
    if "rotational" in commandline_kwargs:
        raise ValueError("rotational=True: the {} solver has no rotational pressure "
                         "update".format(solver.__name__.split(".")[-1]))
    rotational = False
if velocity_pressure_solve not in ("passes", "coupled"):
    raise ValueError("velocity_pressure_solve={!r}: 'passes' or 'coupled'".format(
        velocity_pressure_solve))
if velocity_pressure_solve == "coupled":
    if not hasattr(solver, "coupled_solve"):
        raise ValueError("velocity_pressure_solve='coupled': the {} solver has none; "
                         "use solver='BDF'".format(solver.__name__.split(".")[-1]))
    if velocity_degree <= pressure_degree:
        raise ValueError("velocity_pressure_solve='coupled' needs an inf-sup stable pair, "
                         "velocity_degree > pressure_degree (here {} and {}): the coupled "
                         "equal-order system is not stabilised".format(velocity_degree,
                                                                       pressure_degree))

# Create lists of components solved for
dim = geometric_dimension(mesh)
u_components = ['u' + str(x) for x in range(dim)]
#u_components = ['u']
sys_comp = u_components + ['p'] + ['phig'] + scalar_components
uc_comp = u_components + scalar_components
uphi_components = u_components + ['phig']

# Set up initial folders for storing results
newfolder, tstepfiles = create_initial_folders(**vars())

# Per-step telemetry (twoasis/common/telemetry.py), started before the solvers are made.
if telemetry:
    if MPI.rank(MPI.comm_world) == 0:
        makedirs(path.join(newfolder, "Stats"), exist_ok=True)
    MPI.barrier(MPI.comm_world)
    tel = twoasis.common.telemetry.start(path.join(newfolder, "Stats", "telemetry.jsonl"),
                                         MPI.comm_world)
else:
    tel = twoasis.common.telemetry.start()

# Declare FunctionSpaces and arguments. Constants made without a mesh are the flow mesh's.
set_default_mesh(mesh)
V = Q = FunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
#V = VectorFunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
#S = FunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
if velocity_degree != pressure_degree:
    Q = FunctionSpace(mesh, 'CG', pressure_degree,
                      constrained_domain=constrained_domain)
# The phase field and the chemical potential use the pressure's degree by default: on one
# mesh the capillary force -phi*grad(g) then lies in the gradients the pressure can
# balance. With a higher degree it leaves bulk forces the pressure cannot cancel, which
# blow a run up where the densities or viscosities differ strongly or the interface meets
# walls. The dual mesh (phase_refine > 0) is richer than the pressure in the same way and
# needs a smaller step.
if phase_field_degree is None:
    phase_field_degree = pressure_degree
if chemical_potential_degree is None:
    chemical_potential_degree = phase_field_degree
# Recorded in the checkpoints, so a restart builds the spaces they were written in
NS_parameters.update(phase_field_degree=phase_field_degree,
                     chemical_potential_degree=chemical_potential_degree)
# The phase field's mesh: the flow mesh, or with phase_refine > 0 its uniform refinement
# (the dual mesh, twoasis/common/dual.py), where W lives.
dual = twoasis.common.dual.start(mesh, phase_refine, constrained_domain)
pf_mesh = dual.fine if dual else mesh
phi_el = FiniteElement('CG', pf_mesh.ufl_cell(), phase_field_degree)
g_el = FiniteElement('CG', pf_mesh.ufl_cell(), chemical_potential_degree)
W = FunctionSpace(pf_mesh, MixedElement([phi_el, g_el]),
                  constrained_domain=constrained_domain)
S_DG0 = FunctionSpace(mesh, "DG", 0, constrained_domain=constrained_domain)
check_periodic_pairing(constrained_domain, [V, Q] + (
    [W] if phase_field_degree == chemical_potential_degree else [W.sub(0), W.sub(1)]))

# Facet markers; also written as Timestamps/boundaries.xdmf for partrac
subdomains = BoundaryMarkers(mesh)
subdomains.set_all(0)
# mark_subdomains may return a dict of values to add to the namespace.
vars().update(mark_subdomains(**vars()) or {})
ds = Measure("ds", domain=mesh, subdomain_data=subdomains)
normal = FacetNormal(mesh)
# On the dual mesh, the markers' refinement, where integrals of the phase field over
# marked facets go (twoasis/common/dual.py, retarget)
if dual:
    dual.markers(subdomains)

#constrained_domain.mark(subd, 1)
#wall.mark(subd, 2)
with XDMFFile(mesh_comm(mesh), path.join(newfolder, "subdomains.xdmf")) as xdmff:
    xdmff.write(subdomains)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
#c = TrialFunction(S)
#ct = TestFunction(S)
c = TrialFunction(V)
ct = TestFunction(V)
phi, g = TrialFunctions(W)
xi, eta = TestFunctions(W)

# Use dictionary to hold all FunctionSpaces
VV = dict((ui, V) for ui in uc_comp)
#VV = dict((ci, S) for ci in scalar_components)
#VV['u'] = V
VV['p'] = Q
VV['phig'] = W

# Create dictionaries for the solutions at three timesteps
q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(VV[ui], name=ui + "_1")) for ui in sys_comp)
q_2 = dict((ui, Function(VV[ui], name=ui + "_2")) for ui in sys_comp)

# Create vectors of the segregated velocity components
u_  = as_vector([q_ [ui] for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt
#u_ = q_['u']
#u_1 = q_1['u']
#u_2 = q_2['u']

# Adams Bashforth projection of velocity at t - dt/2
U_AB = 1.5 * u_1 - 0.5 * u_2

# Create short forms for accessing the solution vectors
x_ = dict((ui, q_[ui].vector()) for ui in sys_comp)        # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)      # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in sys_comp)  # Solution vectors t - 2*dt
du_ = dict((ui, x_[ui].copy()) for ui in u_components)  # Velocity change in an inner iteration

# Create vectors to hold rhs of equations
b = dict((ui, Vector(x_[ui])) for ui in sys_comp)      # rhs vectors (final)
b_tmp = dict((ui, Vector(x_[ui])) for ui in sys_comp)  # rhs temp storage vectors

# Short forms pressure and scalars
p_ = q_['p']                # pressure at t
p_1 = q_1['p']              # pressure at t - dt
dp_ = Function(Q)           # pressure correction
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))
phig_ = q_['phig']
phig_1 = q_1['phig']
phig_2 = q_2['phig']
phi_, g_ = split(phig_)
phi_1, g_1 = split(phig_1)
phi_2, g_2 = split(phig_2)

print_solve_info = krylov_solvers['monitor_convergence']

# Boundary conditions
bcs = create_bcs(**vars())

angles = contact_angles(**vars())

if len(angles) > 0 and isinstance(angles[0][0], Function):
    with XDMFFile(mesh_comm(mesh), path.join(newfolder, "angles.xdmf")) as xdmff:
        xdmff.write(angles[0][0])


# Initialize solution, then, if restarting, overwrite the fields with the checkpoint
# (t and tstep came with the checkpoint's parameters).
_vars = initialize(**vars())
if isinstance(_vars, dict):
    vars().update(_vars)
init_from_restart(**vars())

#  Fetch linear algebra solvers
u_sol, p_sol, pf_sol, c_sol = get_solvers(**vars())

# Get body forces
gradp_avg = average_pressure_gradient(**vars())
assert(isinstance(gradp_avg, Coefficient))
bgp0 = dict((ui, v * gradp_avg[i]) for i, ui in enumerate(u_components))
b0 = dict((ui, assemble(bgp0[ui] * dx)) for ui in u_components)

acc = acceleration(**vars())
assert(isinstance(acc, Coefficient))
bg0 = dict((ui, v * acc[i]) for i, ui in enumerate(u_components))

# Get phase field sources (for testing)
fpf = phase_field_source(**vars())
bpf = (xi * fpf, assemble(xi * fpf * dx)) if isinstance(fpf, Coefficient) else None

# Get scalar sources
fs = scalar_source(**vars())
for ci in scalar_components:
    assert(isinstance(fs[ci], Coefficient))
    b0[ci] = assemble(v * fs[ci] * dx)

# Preassemble and allocate
vars().update(setup(**vars()))
anderson = (twoasis.solvers.TPfracStep.AndersonPressure(anderson_depth)
            if anderson_depth else None)

# Anything problem specific
vars().update(pre_solve_hook(**vars()))
# The PETSc objects that setup made and dropped (petsc4py keeps them until this)
collect_garbage()

# |div u| and |grad u| each stat_interval timesteps (Stats/divergence.dat)
divergence_stats = twoasis.problems.TPfracStep.shared.DivergenceStats(u_, mesh, newfolder)

tx = OasisTimer('Timestep timer')
tx.start()
total_timer = OasisTimer("Start simulations", True)
# Save the initial state (not on a restart: the checkpoint and the time series
# already hold it, and saving would overwrite the checkpoint's previous level).
stop = save_solution(**vars()) if restart_folder is None else False
while t < (T - tstep * DOLFIN_EPS) and not stop:
    t += dt
    tstep += 1
    inner_iter = 0
    udiff = array([1e8])  # Norm of velocity change over last inner iter
    unorm = array([0.])   # ... and of the velocity
    num_iter = max(iters_on_first_timestep, max_iter) if tstep <= 10 else max_iter
    tel.start_step()

    start_timestep_hook(**vars())

    # Solve for phase field
    t0 = OasisTimer('Solving phase field', print_solve_info)
    with tel.block("phase_field_assemble"):
        phase_field_assemble(**vars())
    phase_field_hook(**vars())
    with tel.block("phase_field_solve"):
        phase_field_solve(**vars())
    t0.stop()

    # A prescribed physical pressure p + phi*g follows the new phase field.
    physical_bcs = [bc for bc in bcs['p'] if hasattr(bc, 'update_physical')]
    if physical_bcs:
        phig_values = phi_g(q_)
        for bc in physical_bcs:
            bc.update_physical(phig_values)

    # Iterate until the velocity changes by less than max_error, or by less than
    # max_error_rel relative to its norm, both L2 norms (independent of the mesh); the
    # absolute test decides for a fluid (nearly) at rest.
    growing = 0           # passes in a row whose velocity change grew, above the first's
    if velocity_pressure_solve == "coupled":
        # One system for the step's velocity and pressure; the passes below then do
        # not run (no velocity change left), nor does the velocity update.
        inner_iter = 1
        t0 = OasisTimer("Coupled velocity-pressure solve")
        with tel.block("first_inner_assemble"):
            assemble_first_inner_iter(**vars())
        with tel.block("coupled_solve"):
            coupled_solve(**vars())
        t0.stop()
        udiff[0] = 0.0
    while (udiff[0] > max(max_error, max_error_rel * unorm[0])
           and inner_iter < num_iter):
        inner_iter += 1
        udiff_last = udiff[0]

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            with tel.block("first_inner_assemble"):
                assemble_first_inner_iter(**vars())
        udiff[0] = unorm[0] = 0.0
        for i, ui in enumerate(u_components):
            t1 = OasisTimer('Solving tentative velocity ' + ui, print_solve_info)
            with tel.block("velocity_assemble"):
                velocity_tentative_assemble(**vars())
            velocity_tentative_hook(**vars())
            with tel.block("velocity_solve"):
                velocity_tentative_solve(**vars())
            t1.stop()
        udiff[0], unorm[0] = udiff[0] ** 0.5, unorm[0] ** 0.5
        # Growth counts only above the first pass's change: a change that grows
        # but stays below it is the solves' noise near convergence, not divergence.
        if inner_iter == 1:
            udiff_first = udiff[0]
        growing = (growing + 1 if inner_iter > 1 and udiff_last < udiff[0]
                   and udiff[0] > udiff_first else 0)
        if diverging_passes and growing >= diverging_passes:
            raise RuntimeError(
                "timestep {}: the inner passes diverge (the velocity change grew {} passes "
                "in a row, to {:.2e}); a smaller dt may help. The last checkpoint is kept."
                .format(tstep, growing, udiff[0]))

        t0 = OasisTimer("Pressure solve", print_solve_info)
        with tel.block("pressure_assemble"):
            pressure_assemble(**vars())
        pressure_hook(**vars())
        with tel.block("pressure_solve"):
            pressure_solve(**vars())
        with tel.block("pressure_update"):
            pressure_update(**vars())
            # Mix only when another pass follows: the velocity update after the
            # last pass uses that pass's own pressure change, so the stored
            # pressure must be the one it came from.
            if (anderson is not None and inner_iter < num_iter
                    and udiff[0] > max(max_error, max_error_rel * unorm[0])):
                anderson(x_['p'], inner_iter)
        t0.stop()

        print_velocity_pressure_info(**vars())

    # The passes stopped at their limit without converging: worth knowing, since the
    # step then carries the velocity-pressure splitting error of its last pass.
    if num_iter > 1 and udiff[0] > max(max_error, max_error_rel * unorm[0]):
        info_red("Warning: timestep {}: {} inner passes without converging (velocity change "
                 "{:.2e}, tolerance {:.2e})".format(tstep, inner_iter, udiff[0],
                                                    max(max_error, max_error_rel * unorm[0])))

    # Update velocity
    if velocity_pressure_solve == "passes":
        t0 = OasisTimer("Velocity update")
        with tel.block("velocity_update"):
            velocity_update(**vars())
        t0.stop()
    if tstep % stat_interval == 0:
        with tel.block("hooks"):
            divergence_stats(tstep, t)

    # Solve for scalars
    if len(scalar_components) > 0:
        scalar_assemble(**vars())
        for ci in scalar_components:
            t1 = OasisTimer('Solving scalar {}'.format(ci), print_solve_info)
            scalar_hook(**vars())
            scalar_solve(**vars())
            t1.stop()

    # temporal_hook may return a dict of values to update; None means nothing.
    # A returned stop=True ends the run after this step.
    with tel.block("hooks"):
        vars().update(temporal_hook(**vars()) or {})

        # Save solution if required and check for killoasis file
        stop = save_solution(**vars()) or stop
    tel.end_step(tstep, t, inner_iter)

    # Update to a new timestep
    for ui in u_components:
        x_2[ui].zero()
        x_2[ui].axpy(1.0, x_1[ui])
        x_1[ui].zero()
        x_1[ui].axpy(1.0, x_[ui])
    
    x_2['p'].zero()
    x_2['p'].axpy(1., x_1['p'])
    x_1['p'].zero()
    x_1['p'].axpy(1., x_['p'])

    x_2['phig'].zero()
    x_2['phig'].axpy(1., x_1['phig'])
    x_1['phig'].zero()
    x_1['phig'].axpy(1., x_['phig'])

    for ci in scalar_components:
        x_1[ci].zero()
        x_1[ci].axpy(1., x_[ci])
    collect_garbage()  # what this step dropped

    # Print some information
    if tstep % print_intermediate_info == 0:
        toc = tx.stop()
        info_green( 'Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T))
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(
            print_intermediate_info, toc))
        list_timings(TimingClear.clear, [TimingType.wall])
        tx.start()


total_timer.stop()
list_timings(TimingClear.keep, [TimingType.wall])
info_red('Total computing time = {0:f}'.format(total_timer.elapsed()[0]))
oasis_memory('Final memory use ')
total_initial_dolfin_memory = MPI.sum(MPI.comm_world, initial_memory_use)
info_red('Memory use for importing dolfin = {} MB (RSS)'.format(
    total_initial_dolfin_memory))
info_red('Total memory use of solver = ' +
         str(oasis_memory.memory - total_initial_dolfin_memory) + " MB (RSS)")

for tstepfile in tstepfiles.values():
    tstepfile.close()
tel.close()
if restart_folder is not None:
    merge_visualization_files(**vars())

# Final hook
theend_hook(**vars())