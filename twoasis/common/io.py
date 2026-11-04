__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from os import link, makedirs, listdir, remove, rename, path
import shutil
from xml.etree import ElementTree as ET
import pickle
import time
import glob

import numpy as np
from dolfin import MPI, Function, FunctionSpace, HDF5File
from femtools.dolfin_layout import write_dofs, write_function
from femtools.xdmf import XDMFFile2
from twoasis.common.utilities import set_subfunction, subfunction
from twoasis.problems import info_red

__all__ = ["create_initial_folders", "save_solution", "save_tstep_solution_h5",
           "save_checkpoint_solution_h5", "check_if_kill",
           "init_from_restart", "merge_visualization_files", "merge_xml_files",
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
    """Store the velocity (as a vector), pressure, phase field and chemical
    potential on the current timestep in the XDMF time series. The velocity
    vector is built on the first call."""
    if id(tstepfiles) not in _TIMESERIES_OUTPUT:
        _TIMESERIES_OUTPUT[id(tstepfiles)] = AssignedVectorFunction(u_, name="u")
    out = dict(u=_TIMESERIES_OUTPUT[id(tstepfiles)])
    out["u"]()
    for i, name in enumerate(["phi", "g"]):
        out[name] = subfunction(q_['phig'], i)
        out[name].rename(name, name)
    for comp, tstepfile in tstepfiles.items():
        tstepfile.write(out[comp] if comp in out else q_[comp], float(t))

    if MPI.rank(MPI.comm_world) == 0:
        paramsfilename = path.join(newfolder, "Timeseries", "params.dat")
        if not path.exists(paramsfilename):
            with open(paramsfilename, 'wb') as f:
                pickle.dump(NS_parameters, f)


# The velocity vector of save_tstep_solution_h5, per set of time series files.
_TIMESERIES_OUTPUT = {}


#: the spaces (by id) whose dof tables this process has written to a run's checkpoint
#: folder: from then on each checkpoint there links them instead of writing them again
_tables_written = {}


def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters):
    """Write the parameters and the current and previous levels of every field
    to Checkpoint/.

    The files are written to Checkpoint.new/, which then replaces Checkpoint/ by
    renaming, so a run stopped while writing leaves the last complete checkpoint
    in place. Each field is stored as dolfin's HDF5File.write(Function) stores it
    (femtools.dolfin_layout), with its cells' global ids and dofs, so that a
    restart reads it on any number of processes. Those dof tables are fixed for a
    run, so each space's are written once, at the run's first checkpoint
    (dofs_<field>.h5, for the first field on the space), hard-linked into every
    later checkpoint, and linked from the fields' files.
    """
    comm = MPI.comm_world
    checkpointfolder = path.join(newfolder, "Checkpoint")
    tmpfolder = checkpointfolder + ".new"
    oldfolder = checkpointfolder + ".old"
    NS_parameters["num_processes"] = MPI.size(comm)
    written = _tables_written.setdefault(checkpointfolder, set())
    tables = {}             # space id -> the file of its tables
    for ui in q_:
        tables.setdefault(q_[ui].function_space().id(), "dofs_{}.h5".format(ui))
    if MPI.rank(comm) == 0:
        recover_checkpoint(checkpointfolder)
        # Left over if an earlier run stopped while writing.
        for folder in (tmpfolder, oldfolder):
            shutil.rmtree(folder, ignore_errors=True)
        makedirs(tmpfolder)
        with open(path.join(tmpfolder, "params.dat"), "wb") as f:
            pickle.dump(NS_parameters, f)
        for key, name in tables.items():
            if key in written:
                link(path.join(checkpointfolder, name), path.join(tmpfolder, name))
    MPI.barrier(comm)
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
        if path.exists(checkpointfolder):
            rename(checkpointfolder, oldfolder)
        rename(tmpfolder, checkpointfolder)
        shutil.rmtree(oldfolder, ignore_errors=True)
    MPI.barrier(comm)


def recover_checkpoint(checkpointfolder):
    """Put back the last complete checkpoint if a run stopped between the two
    renames of save_checkpoint_solution_h5: Checkpoint/ is then missing, and
    Checkpoint.new/ (complete, since the renames start once it is written) or
    else Checkpoint.old/ holds it."""
    if path.exists(checkpointfolder) or not path.exists(checkpointfolder + ".old"):
        return
    for folder in (checkpointfolder + ".new", checkpointfolder + ".old"):
        if path.exists(folder):
            rename(folder, checkpointfolder)
            return


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
    number of processes that wrote it. A field stored as a bare vector (the old
    layout) is in the dof order of the run that wrote it, and is read only on the
    same number of processes.
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


def continued(series, time):
    """Merge per-file lists of steps (each starting with its earliest), taken in
    the order they were written: a later list replaces the steps from its first
    time on, as a restart continues a run from a checkpoint and the run's steps
    after it belong to an abandoned history. `time(step)` gives a step's time."""
    steps = []
    for later in (s for s in series if s):
        t0 = time(later[0])
        steps = [s for s in steps if time(s) < t0 - 1e-9 * max(1., abs(t0))] + list(later)
    return steps


def merge_visualization_files(newfolder, **namespace):
    """Merge the time series of each field written by a run and the restarts that
    continued it into the first run's file (merge_xml_files)."""
    if MPI.rank(MPI.comm_world) == 0:
        timefolder = path.join(newfolder, 'Timeseries')
        for name in ["u", "p", "phi", "g"]:
            files = series_files(timefolder, name)
            if len(files) > 1:
                merge_xml_files(files)


def merge_xml_files(files):
    """Merge XDMF time series, given in the order they were written, into the
    first file: each file's steps replace the steps before it from its first time
    on (continued). The other .xdmf files are deleted; the .h5 files they refer
    to, which the merged series refers to now, are kept."""
    ET.register_namespace("xi", "http://www.w3.org/2001/XInclude")
    trees = [ET.parse(f) for f in files]
    collections = [tree.getroot().find("Domain").find("Grid") for tree in trees]
    grids = [[g for g in c if g.tag == "Grid"] for c in collections]
    merged = continued(grids, lambda g: float(g.find("Time").get("Value")))
    for g in list(collections[0]):
        collections[0].remove(g)
    for g in merged:
        collections[0].append(g)
    trees[0].write(files[0], xml_declaration=True)
    for f in files[1:]:
        remove(f)


def xdmf_time_series(xdmf_file):
    """The steps of a dolfin XDMF time series: a list of (time, data, geometry),
    where data and geometry are (HDF5 file, dataset) of the step's values and of
    the mesh coordinates they belong to (`femtools.xdmf.grids`)."""
    from femtools.xdmf import grids
    return [(g["time"], list(g["attributes"].values())[-1] if g["attributes"] else None,
             g["geometry"]) for g in grids(xdmf_file)]


class PointMatcher:
    """Finds, for given points, the saved point within tol of each. The saved
    points are hashed (from their coordinates rounded to multiples of tol) and
    sorted once; a query is then a binary search."""

    def __init__(self, points, tol):
        self.points, self.tol = points, tol
        keys = self._hash(points)
        self.order = np.argsort(keys)
        self.sorted_keys = keys[self.order]

    def _hash(self, x):
        k = np.round(x / self.tol).astype(np.int64).view(np.uint64)
        h = np.zeros(len(x), dtype=np.uint64)
        for j in range(k.shape[1]):
            h = h * np.uint64(0x9E3779B97F4A7C15) + k[:, j]  # wraps around
        return h

    def __call__(self, queries):
        if not len(queries):
            return np.zeros(0, dtype=np.int64)
        keys = self._hash(queries)
        by_key = np.argsort(keys)  # sorted queries make the search cache-friendly
        pos = np.empty(len(keys), dtype=np.int64)
        pos[by_key] = np.searchsorted(self.sorted_keys, keys[by_key])
        index = self.order[np.minimum(pos, len(self.order) - 1)]
        # A coordinate that rounds the other way (or a hash collision) is
        # missed; look those few up directly. Many misses mean another mesh.
        missed = np.flatnonzero(np.linalg.norm(self.points[index] - queries, axis=1) > self.tol)
        if len(missed) > 0.01 * len(queries) + 10:
            missed = missed[:1]
        for i in missed:
            index[i] = np.argmin(np.linalg.norm(self.points - queries[i], axis=1))
        distance = np.linalg.norm(self.points[index] - queries, axis=1)
        if distance.max() > self.tol:
            i = distance.argmax()
            raise ValueError("no saved point within {:g} of {} (nearest at {:g}): the time "
                             "series was written on a different mesh".format(
                                 self.tol, queries[i], distance[i]))
        return index


def match_points(points, queries, tol):
    """For each row of queries, the index of the row of points at distance at
    most tol from it; raise ValueError if there is none."""
    return PointMatcher(points, tol)(queries)


def load_timeseries(folder, q_, VV, constrained_domain, time=None):
    """Set q_ (velocity, pressure, phase field and chemical potential) from the
    XDMF time series in folder, the Timeseries/ of an earlier run: at the last
    saved time of the velocity, or at its saved time closest to `time`; the
    other fields must have a step at the same time. Returns the time read.

    The series hold the values at the mesh vertices, and for a P2 field also at
    the edge midpoints (series written by dolfin's XDMFFile hold only the
    vertices). Each process reads, in chunks, the saved nodes inside its part of
    the mesh and matches the coordinates of its own degrees of freedom to them,
    so the run may use a different number of processes from the one that wrote
    the files. Fields of higher degree than the saved values are interpolated
    from the vertex values."""
    import h5py

    mesh = VV['p'].mesh()
    gdim = mesh.geometry().dim()
    x = mesh.coordinates()
    tol = 1e-8 * MPI.max(mesh.mpi_comm(), float((x.max(axis=0) - x.min(axis=0)).max()))
    # This process's part of the mesh: every dof of a Lagrange space lies in it.
    lo, hi = x.min(axis=0) - tol, x.max(axis=0) + tol
    nodes_of = {}  # geometry dataset -> (rows here, matcher), shared by the fields
    CHUNK = 1 << 20

    def step(name, t_exact=None):
        """The step of the series name at the requested time, or, given t_exact,
        at exactly that time (so that all fields are read at the same time)."""
        steps = continued([xdmf_time_series(f) for f in series_files(folder, name)],
                          lambda s: s[0])
        if not steps:
            raise FileNotFoundError("no {}_from_tstep_*.xdmf in {}".format(name, folder))
        if t_exact is None:
            t_target = max(s[0] for s in steps) if time is None else time
            return min(reversed(steps), key=lambda s: abs(s[0] - t_target))
        match = [s for s in steps if abs(s[0] - t_exact) <= 1e-9 * max(1., abs(t_exact))]
        if not match:
            raise ValueError("{} has no step at t = {} in {}".format(name, t_exact, folder))
        return match[-1]

    def read(name, t_exact=None):
        """The saved values (one column per component), a matcher for the saved
        coordinates, and the time."""
        t, (h5name, dataset), (h5geometry, geometry) = step(name, t_exact)
        key = (h5geometry, geometry)
        if key not in nodes_of:
            rows, nodes = [], []
            with h5py.File(h5geometry, "r") as h5f:
                ds = h5f[geometry]
                n = ds.shape[0]
                for start in range(0, n, CHUNK):
                    block = ds[start:start + CHUNK, :gdim]
                    here = np.flatnonzero(((block >= lo) & (block <= hi)).all(axis=1))
                    rows.append(start + here)
                    nodes.append(block[here])
            matcher = PointMatcher(np.concatenate(nodes), tol)
            # More nodes than vertices: the edge midpoints of a P2 field too.
            matcher.degree = 2 if n > mesh.num_entities_global(0) else 1
            nodes_of[key] = (np.concatenate(rows), matcher)
        rows, matcher = nodes_of[key]
        values = []
        with h5py.File(h5name, "r") as h5f:
            ds = h5f[dataset]
            for start in range(0, ds.shape[0], CHUNK):
                block = ds[start:start + CHUNK]
                values.append(block[rows[(rows >= start) & (rows < start + CHUNK)] - start])
        values = np.concatenate(values)
        return values.reshape(len(values), -1), matcher, t

    def fill(function, space, values, matcher):
        """Set the dofs of space (function's space, or a subspace of it) from
        the saved values."""
        if space.ufl_element().degree() <= matcher.degree:
            first = space.dofmap().ownership_range()[0]
            dofs = np.asarray(space.dofmap().dofs()) - first
            coordinates = function.function_space().tabulate_dof_coordinates().reshape(-1, gdim)
            array = function.vector().get_local()
            array[dofs] = values[matcher(coordinates[dofs])]
            function.vector().set_local(array)
            function.vector().apply("insert")
        else:
            P1 = FunctionSpace(mesh, "CG", 1, constrained_domain=constrained_domain)
            f1 = Function(P1)
            fill(f1, P1, values, matcher)
            f1.vector().update_ghost_values()
            if not len(space.component()):  # not a subspace
                function.interpolate(f1)
            else:
                set_subfunction(function, space.component()[0], f1)

    u, matcher, t = read("u")
    for i in range(gdim):
        fill(q_['u{}'.format(i)], q_['u{}'.format(i)].function_space(), u[:, i], matcher)
    p, matcher, _ = read("p", t)
    fill(q_['p'], q_['p'].function_space(), p[:, 0], matcher)
    W = q_['phig'].function_space()
    for i, name in enumerate(["phi", "g"]):
        values, matcher, _ = read(name, t)
        fill(q_['phig'], W.sub(i), values[:, 0], matcher)
    for f in q_.values():
        f.vector().update_ghost_values()
    return t
