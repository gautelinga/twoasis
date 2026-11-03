__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"
# Modified for two-phase flow by Gaute Linga, 2022-2026.

from os import makedirs, listdir, remove, rename, path
import shutil
from xml.etree import ElementTree as ET
import pickle
import time
import glob

import numpy as np
from dolfin import MPI, Function, FunctionAssigner, FunctionSpace, XDMFFile, HDF5File, assign
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
    if MPI.rank(MPI.comm_world) == 0:
        if not restart_folder:
            makedirs(path.join(newfolder, "Timeseries"))
            makedirs(path.join(newfolder, "Checkpoint"))

    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles = {}
    comps = ['p', 'u', 'phi', 'g'] + scalar_components
    for ui in comps:
        tstepfiles[ui] = XDMFFile(MPI.comm_world, path.join(
            tstepfolder, ui + '_from_tstep_{}.xdmf'.format(tstep)))
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

    pauseoasis = check_if_pause(folder)
    while pauseoasis:
        time.sleep(5)
        pauseoasis = check_if_pause(folder)

    killoasis = check_if_kill(folder, killtime, total_timer)
    if tstep % checkpoint == 0 or killoasis:
        save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters)

    return killoasis


def save_tstep_solution_h5(t, q_, u_, newfolder, tstepfiles, AssignedVectorFunction,
                           NS_parameters):
    """Store the velocity (as a vector), pressure, phase field and chemical
    potential on the current timestep in the XDMF time series. The functions
    written and the assigners filling them are built on the first call."""
    out = _TIMESERIES_OUTPUT.get(id(tstepfiles))
    if out is None:
        W = q_['phig'].function_space()
        out = dict(u=AssignedVectorFunction(u_, name="u"))
        for i, name in enumerate(["phi", "g"]):
            out[name] = Function(W.sub(i).collapse(), name=name)
            out[name + "_assigner"] = FunctionAssigner(out[name].function_space(), W.sub(i))
        _TIMESERIES_OUTPUT[id(tstepfiles)] = out
    out["u"]()
    for name in ["phi", "g"]:
        out[name + "_assigner"].assign(out[name], q_['phig'].sub(["phi", "g"].index(name)))
    for comp, tstepfile in tstepfiles.items():
        tstepfile.write(out[comp] if comp in out else q_[comp], float(t))

    if MPI.rank(MPI.comm_world) == 0:
        paramsfilename = path.join(newfolder, "Timeseries", "params.dat")
        if not path.exists(paramsfilename):
            with open(paramsfilename, 'wb') as f:
                pickle.dump(NS_parameters, f)


# The output functions of save_tstep_solution_h5, per set of time series files.
_TIMESERIES_OUTPUT = {}


def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, NS_parameters):
    """Write the parameters and the current and previous levels of every field
    to Checkpoint/.

    The files are written to Checkpoint.new/, which then replaces Checkpoint/ by
    renaming, so a run stopped while writing leaves the last complete checkpoint
    in place. The fields are stored in the parallel dof order, so a restart must
    use the same number of processes (post_import_problem checks this).
    """
    comm = MPI.comm_world
    checkpointfolder = path.join(newfolder, "Checkpoint")
    tmpfolder = checkpointfolder + ".new"
    oldfolder = checkpointfolder + ".old"
    NS_parameters["num_processes"] = MPI.size(comm)
    if MPI.rank(comm) == 0:
        # Left over if an earlier run stopped while writing.
        for folder in (tmpfolder, oldfolder):
            shutil.rmtree(folder, ignore_errors=True)
        makedirs(tmpfolder)
        with open(path.join(tmpfolder, "params.dat"), "wb") as f:
            pickle.dump(NS_parameters, f)
    MPI.barrier(comm)
    for ui in q_:
        with HDF5File(comm, path.join(tmpfolder, ui + ".h5"), "w") as h5file:
            h5file.write(q_[ui].vector(), "/current")
            h5file.write(q_1[ui].vector(), "/previous")
    MPI.barrier(comm)
    if MPI.rank(comm) == 0:
        if path.exists(checkpointfolder):
            rename(checkpointfolder, oldfolder)
        rename(tmpfolder, checkpointfolder)
        shutil.rmtree(oldfolder)
    MPI.barrier(comm)


def check_if_kill(folder, killtime, total_timer, key='killtwoasis'):
    """Check if user has put a file named {key} in folder or if given killtime has been reached."""
    found = 0
    if key in listdir(folder):
        found = 1
    collective = MPI.sum(MPI.comm_world, found)
    if collective > 0:
        if MPI.rank(MPI.comm_world) == 0:
            remove(path.join(folder, key))
            info_red(key+' found! Stopping simulations cleanly...')
        return True
    else:
        # The slowest rank decides, so that all ranks stop (and checkpoint) together.
        elapsed_time = MPI.max(MPI.comm_world, float(total_timer.elapsed()[0]))
        if killtime is not None and killtime <= elapsed_time:
            if MPI.rank(MPI.comm_world) == 0:
                info_red('Given killtime reached! Stopping simulations cleanly...')
            return True
        else:
            return False


def check_if_pause(folder, key='pausetwoasis'):
    """Check if user has put a file named pausetwoasis in folder."""
    found = 0
    if key in listdir(folder):
        found = 1
    collective = MPI.sum(MPI.comm_world, found)
    if collective > 0:
        if MPI.rank(MPI.comm_world) == 0:
            info_red(key+' found! Simulations paused. Remove ' + path.join(folder, key) + ' to resume simulations...')
        return True
    else:
        return False


def init_from_restart(restart_folder, sys_comp, uc_comp,
                      q_, q_1, q_2, tstep, **NS_namespace):
    """Initialize solution from checkpoint files """
    if restart_folder:
        if MPI.rank(MPI.comm_world) == 0:
            info_red('Restarting from checkpoint at time step {}'.format(tstep))

        for ui in sys_comp:
            filename = path.join(restart_folder, ui + '.h5')
            with HDF5File(MPI.comm_world, filename, "r") as hdf5_file:
                hdf5_file.read(q_[ui].vector(), "/current", False)
                q_[ui].vector().apply('insert')
                q_1[ui].vector().zero()
                q_1[ui].vector().axpy(1., q_[ui].vector())
                q_1[ui].vector().apply('insert')
                # A checkpoint may hold only the current level of a field. In that
                # case use it for the previous level too; the BDF solver then takes
                # a first-order first step.
                if hdf5_file.has_dataset("/previous"):
                    hdf5_file.read(q_2[ui].vector(), "/previous", False)
                else:
                    q_2[ui].vector().zero()
                    q_2[ui].vector().axpy(1., q_[ui].vector())
                q_2[ui].vector().apply('insert')


def merge_visualization_files(newfolder, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        timefolder = path.join(newfolder, 'Timeseries')
        # Gather files
        for name in ["u", "p", "phi", "g"]:
            files = glob.glob(path.join(timefolder, name + "_from_tstep_*.xdmf"))
            if len(files) > 1:
                merge_xml_files(files)


def merge_xml_files(files):
    # Get first timestep and trees
    first_timesteps = []
    trees = []
    for f in files:
        trees.append(ET.parse(f))
        root = trees[-1].getroot()
        first_timesteps.append(float(root[0][0][0][2].attrib["Value"]))

    # Index valued sort (bypass numpy dependency)
    first_timestep_sorted = sorted(first_timesteps)
    indexes = [first_timesteps.index(i) for i in first_timestep_sorted]

    # Get last timestep of first tree
    base_tree = trees[indexes[0]]
    last_node = base_tree.getroot()[0][0][-1]
    ind = 1 if len(list(last_node)) == 3 else 2
    last_timestep = float(last_node[ind].attrib["Value"])

    # Append
    for index in indexes[1:]:
        tree = trees[index]
        for node in list(tree.getroot()[0][0]):
            ind = 1 if len(list(node)) == 3 else 2
            if last_timestep < float(node[ind].attrib["Value"]):
                base_tree.getroot()[0][0].append(node)
                last_timestep = float(node[ind].attrib["Value"])

    # Write the merged series to the first file in time and delete the others.
    base_tree.write(files[indexes[0]], xml_declaration=True)
    for index in indexes[1:]:
        remove(files[index])


def xdmf_time_series(xdmf_file):
    """The steps of a dolfin XDMF time series: a list of (time, data, geometry),
    where data and geometry are (HDF5 file, dataset) of the step's values and of
    the mesh coordinates they belong to."""
    basedir = path.dirname(xdmf_file)

    def address(item):
        h5file, dataset = item.text.strip().split(":")
        return path.join(basedir, h5file), dataset

    steps, times, geometry = [], [], None
    for i, step in enumerate(ET.parse(xdmf_file).getroot()[0][0]):
        if step.tag == "Time":  # all times in one list (older dolfin)
            times = [float(t) for t in step[0].text.split()]
        elif step.tag == "Grid":
            t, data = None, None
            for prop in step:
                if prop.tag == "Time":
                    t = float(prop.attrib["Value"])
                elif prop.tag == "Attribute":
                    data = address(prop[0])
                elif prop.tag == "Geometry" and len(prop) and prop[0].text and ":" in prop[0].text:
                    geometry = address(prop[0])
            steps.append((times[i - 1] if t is None else t, data, geometry))
    return steps


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

    The series hold the values at the mesh vertices. Each process matches the
    coordinates of its own degrees of freedom to the saved ones, so the run may
    use a different number of processes from the one that wrote the files.
    Fields of degree higher than one are interpolated from the vertex values."""
    import h5py

    mesh = VV['p'].mesh()
    gdim = mesh.geometry().dim()
    x = mesh.coordinates()
    tol = 1e-8 * MPI.max(mesh.mpi_comm(), float((x.max(axis=0) - x.min(axis=0)).max()))
    matchers = {}  # each field's file holds the same coordinates; sort them once

    def step(name, t_exact=None):
        """The step of the series name at the requested time, or, given t_exact,
        at exactly that time (so that all fields are read at the same time)."""
        steps = [s for f in sorted(glob.glob(path.join(folder, name + "_from_tstep_*.xdmf")))
                 for s in xdmf_time_series(f)]
        if not steps:
            raise FileNotFoundError("no {}_from_tstep_*.xdmf in {}".format(name, folder))
        if t_exact is None:
            t_target = max(s[0] for s in steps) if time is None else time
            return min(steps, key=lambda s: abs(s[0] - t_target))
        match = [s for s in steps if abs(s[0] - t_exact) <= 1e-9 * max(1., abs(t_exact))]
        if not match:
            raise ValueError("{} has no step at t = {} in {}".format(name, t_exact, folder))
        return match[-1]

    def read(name, t_exact=None):
        """The saved values (one column per component), a matcher for the saved
        coordinates, and the time."""
        t, (h5name, dataset), (h5geometry, geometry) = step(name, t_exact)
        with h5py.File(h5name, "r") as h5f:
            values = h5f[dataset][:]
        with h5py.File(h5geometry, "r") as h5f:
            nodes = h5f[geometry][:, :gdim]
        key = hash(nodes.tobytes())
        if key not in matchers:
            matchers[key] = PointMatcher(nodes, tol)
        return values.reshape(len(values), -1), matchers[key], t

    def fill(function, space, values, matcher):
        """Set the dofs of space (function's space, or a subspace of it) from
        the vertex values."""
        if space.ufl_element().degree() == 1:
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
                f = Function(space.collapse())
                f.interpolate(f1)
                assign(function.sub(space.component()[0]), f)

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
