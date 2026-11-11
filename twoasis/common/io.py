__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from os import link, makedirs, listdir, remove, path
import shutil
import pickle
import time
import glob

from dolfin import MPI, Function, HDF5File
from femtools.dolfin_layout import write_dofs, write_function
from femtools.swap import new_folder, recover, swap_in
from femtools.xdmf import continued, merge_series, XDMFFile2
from twoasis.common.utilities import phi_g, physical_pressure, set_subfunction, subfunction
from twoasis.problems import info_red

__all__ = ["create_initial_folders", "save_solution", "save_tstep_solution_h5",
           "save_checkpoint_solution_h5", "check_if_kill",
           "init_from_restart", "merge_visualization_files",
           "load_timeseries"]


def create_initial_folders(folder, restart_folder, sys_comp, tstep, info_red,
                           scalar_components, **NS_namespace):
    """Create necessary folders."""
    info_red("Creating initial folders")
    # To avoid writing over old data create a new folder for each run
    if MPI.rank(MPI.comm_world) == 0:
        try:
            makedirs(folder)
        except OSError:
            pass

    MPI.barrier(MPI.comm_world)
    newfolder = path.join(folder, 'data')
    if restart_folder:
        # The run folder that holds the checkpoint (<folder>/data/<n>/Checkpoint).
        newfolder = path.dirname(path.abspath(restart_folder.rstrip("/")))
    else:
        if not path.exists(newfolder):
            newfolder = path.join(newfolder, '1')
        else:
            previous = [f for f in listdir(newfolder) if f.isnumeric()]
            previous = max(map(int, previous)) if previous else 0
            newfolder = path.join(newfolder, str(previous + 1))

    MPI.barrier(MPI.comm_world)
    tstepfolder = path.join(newfolder, "Timeseries")
    comps = ['p', 'u', 'phi', 'g'] + scalar_components
    # The series of this run start at tstep. A restart from the same step as an
    # earlier one gets a name of its own (_r1, _r2, ...), so that it does not
    # overwrite that run's files.
    stem = None
    if MPI.rank(MPI.comm_world) == 0:
        makedirs(tstepfolder, exist_ok=True)
        stem, k = "_from_tstep_{}".format(tstep), 0
        while any(glob.glob(path.join(tstepfolder, ui + stem + ".*")) for ui in comps):
            k += 1
            stem = "_from_tstep_{}_r{}".format(tstep, k)
    stem = MPI.comm_world.bcast(stem, root=0)

    # XDMFFile2 (femtools) writes what dolfin's XDMFFile writes, but keeps the
    # midside nodes of a P2 field and compresses the data.
    tstepfiles = {}
    for ui in comps:
        tstepfiles[ui] = XDMFFile2(MPI.comm_world, path.join(tstepfolder, ui + stem + ".xdmf"))
        tstepfiles[ui].parameters["rewrite_function_mesh"] = False
        tstepfiles[ui].parameters["flush_output"] = True

    return newfolder, tstepfiles


def save_solution(tstep, t, q_, q_1, folder, newfolder, save_step, checkpoint,
                  NS_parameters, tstepfiles, u_, u_components, scalar_components,
                  constrained_domain,
                  AssignedVectorFunction, killtime, total_timer, **NS_namespace):
    """Called at end of timestep. Check for kill and save solution if required."""
    NS_parameters.update(t=t, tstep=tstep)
    if tstep % save_step == 0:
        save_tstep_solution_h5(t, q_, u_, newfolder, tstepfiles, AssignedVectorFunction,
                               NS_parameters)

    while check_if_pause(folder, newfolder):
        time.sleep(5)

    killoasis = check_if_kill(folder, killtime, total_timer, newfolder=newfolder)
    if tstep % checkpoint == 0 or killoasis:
        save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters)

    return killoasis


def save_tstep_solution_h5(t, q_, u_, newfolder, tstepfiles, AssignedVectorFunction,
                           NS_parameters):
    """Store the velocity (as a vector), the physical pressure p + phi*g, the phase
    field and the chemical potential on the current timestep in the XDMF time
    series. params.dat, written with the first step, records that the pressure is
    the physical one (timeseries_pressure), which load_timeseries reads; a folder
    whose params.dat has no such record goes on holding p. The velocity vector and
    the pressure's Function are built on the first call."""
    if id(tstepfiles) not in _TIMESERIES_OUTPUT:
        physical = True
        if MPI.rank(MPI.comm_world) == 0:
            paramsfilename = path.join(newfolder, "Timeseries", "params.dat")
            if path.exists(paramsfilename):
                with open(paramsfilename, 'rb') as f:
                    physical = pickle.load(f).get("timeseries_pressure") == "physical"
            else:
                with open(paramsfilename, 'wb') as f:
                    pickle.dump(dict(NS_parameters, timeseries_pressure="physical"), f)
        out = dict(u=AssignedVectorFunction(u_, name="u"))
        if MPI.comm_world.bcast(physical, root=0):
            out["p"] = Function(q_['p'].function_space())
            out["p"].rename("p", "p")
        _TIMESERIES_OUTPUT[id(tstepfiles)] = out
    out = dict(_TIMESERIES_OUTPUT[id(tstepfiles)])
    out["u"]()
    if "p" in out:
        physical_pressure(q_, out["p"])
    for i, name in enumerate(["phi", "g"]):
        out[name] = subfunction(q_['phig'], i)
        out[name].rename(name, name)
    for comp, tstepfile in tstepfiles.items():
        tstepfile.write(out[comp] if comp in out else q_[comp], float(t))


# The velocity vector and pressure Function of save_tstep_solution_h5, per set of
# time series files.
_TIMESERIES_OUTPUT = {}


#: the spaces (by id) whose dof tables this process has written to a run's checkpoint
#: folder: from then on each checkpoint there links them instead of writing them again
_tables_written = {}


def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters):
    """Write the parameters and the current and previous levels of every field
    to Checkpoint/.

    The files are written to Checkpoint.new/, which then replaces Checkpoint/ by
    renaming (femtools.swap), so a run stopped while writing leaves the last
    complete checkpoint in place. Each field is stored as dolfin's
    HDF5File.write(Function) stores it (femtools.dolfin_layout), with its cells'
    global ids and dofs, so that a restart reads it on any number of processes. Those dof tables are fixed for a
    run, so each space's are written once, at the run's first checkpoint
    (dofs_<field>.h5, for the first field on the space), hard-linked into every
    later checkpoint (written again where that fails), and linked from the
    fields' files.
    """
    comm = MPI.comm_world
    checkpointfolder = path.join(newfolder, "Checkpoint")
    tmpfolder = new_folder(checkpointfolder)
    NS_parameters["num_processes"] = MPI.size(comm)
    written = _tables_written.setdefault(checkpointfolder, set())
    tables = {}             # space id -> the file of its tables
    unlinked = set()
    for ui in q_:
        tables.setdefault(q_[ui].function_space().id(), "dofs_{}.h5".format(ui))
    if MPI.rank(comm) == 0:
        recover_checkpoint(checkpointfolder)
        # Left over if an earlier run stopped while writing.
        shutil.rmtree(tmpfolder, ignore_errors=True)
        makedirs(tmpfolder)
        with open(path.join(tmpfolder, "params.dat"), "wb") as f:
            pickle.dump(NS_parameters, f)
        for key, name in tables.items():
            if key in written:
                try:
                    link(path.join(checkpointfolder, name), path.join(tmpfolder, name))
                except OSError:     # Checkpoint/ moved away, or no hard links here
                    unlinked.add(key)
    written -= comm.bcast(unlinked, root=0)
    for ui in q_:
        V = q_[ui].function_space()
        name = tables[V.id()]
        if V.id() not in written:
            write_dofs(path.join(tmpfolder, name), V, "dofs")
            written.add(V.id())
        filename = path.join(tmpfolder, ui + ".h5")
        write_function(filename, q_[ui], "current", mode="w", dofs=(name, "dofs"))
        write_function(filename, q_1[ui], "previous", dofs=(name, "dofs"))
    MPI.barrier(comm)
    if MPI.rank(comm) == 0:
        swap_in(tmpfolder, checkpointfolder)
    MPI.barrier(comm)


def recover_checkpoint(checkpointfolder):
    """Put back the last complete checkpoint if a run stopped between the two
    renames of save_checkpoint_solution_h5 (femtools.swap.recover)."""
    recover(checkpointfolder)


def _control_file(key, folder, newfolder):
    """The control file `key` the user has put in the run's folder (newfolder)
    or in folder, or None. Checked on rank 0 only."""
    for where in (newfolder, folder):
        if where is not None and path.exists(path.join(where, key)):
            return path.join(where, key)
    return None


def check_if_kill(folder, killtime, total_timer, key='killtwoasis', newfolder=None):
    """Whether to stop the run cleanly: a file named key in the run's folder or
    in folder (a file in folder stops the first run to see it), or killtime
    reached. Rank 0 decides, and every rank gets its answer in one broadcast."""
    stop = None
    if MPI.rank(MPI.comm_world) == 0:
        found = _control_file(key, folder, newfolder)
        if found:
            remove(found)
            info_red(key + ' found! Stopping simulations cleanly...')
            stop = True
        elif killtime is not None and killtime <= total_timer.elapsed()[0]:
            info_red('Given killtime reached! Stopping simulations cleanly...')
            stop = True
        else:
            stop = False
    return MPI.comm_world.bcast(stop, root=0)


def check_if_pause(folder, newfolder=None, key='pausetwoasis'):
    """Whether a file named key (pausetwoasis) in the run's folder or in folder
    pauses the run. Rank 0 decides; one broadcast."""
    found = None
    if MPI.rank(MPI.comm_world) == 0:
        found = _control_file(key, folder, newfolder)
        if found:
            info_red(key + ' found! Simulations paused. Remove ' + found + ' to resume simulations...')
    return MPI.comm_world.bcast(found, root=0) is not None


def init_from_restart(restart_folder, sys_comp, uc_comp,
                      q_, q_1, q_2, tstep, num_processes=None, **NS_namespace):
    """Initialize solution from checkpoint files.

    A field stored as a Function is read onto this run's partition, whatever the
    number of processes that wrote it. A field stored as a bare vector
    (HDF5File.write of a vector) is in the dof order of the run that wrote it, and
    is read only on the same number of processes.
    """
    if not restart_folder:
        return
    comm = MPI.comm_world
    if MPI.rank(comm) == 0:
        info_red('Restarting from checkpoint at time step {}'.format(tstep))
    for ui in sys_comp:
        filename = path.join(restart_folder, ui + '.h5')
        with HDF5File(comm, filename, "r") as hdf5_file:
            # A Function is a group carrying its element's signature; a bare
            # vector a dataset carrying its partition.
            if "signature" in hdf5_file.attributes("/current").list_attributes():
                read = lambda u, name: hdf5_file.read(u, name)
            else:
                if num_processes is not None and num_processes != MPI.size(comm):
                    raise ValueError(
                        "The checkpoint in {} was written by {} processes as bare vectors; "
                        "restart it with the same number (got {})".format(
                            restart_folder, num_processes, MPI.size(comm)))
                read = lambda u, name: hdf5_file.read(u.vector(), name, False)
            read(q_[ui], "/current")
            q_[ui].vector().apply('insert')
            q_1[ui].vector().zero()
            q_1[ui].vector().axpy(1., q_[ui].vector())
            q_1[ui].vector().apply('insert')
            # A checkpoint may hold only the current level of a field. In that
            # case use it for the previous level too; the BDF solver then takes
            # a first-order first step.
            if hdf5_file.has_dataset("/previous"):
                read(q_2[ui], "/previous")
            else:
                q_2[ui].vector().zero()
                q_2[ui].vector().axpy(1., q_[ui].vector())
            q_2[ui].vector().apply('insert')


def series_files(folder, name):
    """The time series files of field `name` in folder, in the order they were
    written: the first run's, then those of the restarts that continued it."""
    return sorted(glob.glob(path.join(folder, name + "_from_tstep_*.xdmf")),
                  key=lambda f: (path.getmtime(f), f))


def merge_visualization_files(newfolder, **namespace):
    """Merge the time series of each field written by a run and the restarts that
    continued it into the first run's file (femtools.xdmf.merge_series)."""
    if MPI.rank(MPI.comm_world) == 0:
        timefolder = path.join(newfolder, 'Timeseries')
        for name in ["u", "p", "phi", "g"]:
            files = series_files(timefolder, name)
            if len(files) > 1:
                merge_series(files)


def load_timeseries(folder, q_, VV, constrained_domain, time=None):
    """Set q_ (velocity, pressure, phase field and chemical potential) from the
    XDMF time series in folder, the Timeseries/ of an earlier run: at the last
    saved time of the velocity, or at its saved time closest to `time`; the
    other fields must have a step at the same time. Returns the time read.

    The steps of a field are those of the run's files and of the restarts that
    continued it (femtools.xdmf.continued). Each field is read by femtools
    (XDMFFile2.read_checkpoint): from its nodal values, P1 or P2, on any number of
    processes, onto the field's space of any degree. A series whose params.dat marks
    its pressure as the physical one (timeseries_pressure) gives p = P - phi*g;
    a series without that record holds p."""
    from dolfin import FunctionAssigner, VectorFunctionSpace
    from femtools.xdmf import grids
    from twoasis.common.utilities import subspace

    mesh = VV['p'].mesh()
    gdim = mesh.geometry().dim()

    def step(name, t_exact=None):
        """(time, file, counter) of the step of series name at the requested time,
        or, given t_exact, at exactly that time (so that all fields are read at
        the same time)."""
        per_file = []
        for f in series_files(folder, name):
            picked = [g for g in grids(f) if name in g["attributes"]]
            per_file.append([(g["time"], f, k) for k, g in enumerate(picked)])
        steps = continued(per_file, lambda s: s[0])
        if not steps:
            raise FileNotFoundError("no {}_from_tstep_*.xdmf in {}".format(name, folder))
        if t_exact is None:
            t_target = max(s[0] for s in steps) if time is None else time
            return min(reversed(steps), key=lambda s: abs(s[0] - t_target))
        match = [s for s in steps if abs(s[0] - t_exact) <= 1e-9 * max(1., abs(t_exact))]
        if not match:
            raise ValueError("{} has no step at t = {} in {}".format(name, t_exact, folder))
        return match[-1]

    def read(function, name, t_exact=None):
        t, f, counter = step(name, t_exact)
        with XDMFFile2(mesh.mpi_comm(), f) as xdmf:
            xdmf.read_checkpoint(function, name, counter)
        return t

    V = VV['u0']
    us = [q_['u{}'.format(i)] for i in range(gdim)]
    el = V.ufl_element()
    uv = Function(VectorFunctionSpace(mesh, el.family(), el.degree(),
                                      constrained_domain=constrained_domain))
    t = read(uv, "u")
    FunctionAssigner([V] * gdim, uv.function_space()).assign(us, uv)
    read(q_['p'], "p", t)
    W = q_['phig'].function_space()
    for i, name in enumerate(["phi", "g"]):
        f = Function(subspace(W, i))
        read(f, name, t)
        set_subfunction(q_['phig'], i, f)
    for f in q_.values():
        f.vector().update_ghost_values()
    params = path.join(folder, "params.dat")
    if path.exists(params):
        with open(params, "rb") as f:
            physical = pickle.load(f).get("timeseries_pressure") == "physical"
        if physical:
            x = q_['p'].vector()
            x.set_local(x.get_local() - phi_g(q_))
            x.apply('insert')
            x.update_ghost_values()
    return t
