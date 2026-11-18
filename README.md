twoasis
=======

*twoasis* simulates two-phase flow using the incompressible Cahn–Hilliard–Navier–Stokes model, solved with the finite element method on legacy [FEniCS](https://fenicsproject.org/download/archive/) (dolfin 2019.2) or on [FEniCSx](https://fenicsproject.org) (dolfinx 0.11): one code, the same algorithm and defaults on both (see [Backends](#backends)). 
It is originally a stripped-down fork of [Oasis](https://github.com/mikaem/Oasis) by Mikael Mortensen, and is closely related to [Bernaise](https://github.com/gautelinga/BERNAISE), which simulates two-phase electrohydrodynamic flow (see [Publications](#publications)).

## Installation

### Backend
On Ubuntu, add the FEniCS PPA:
```
sudo add-apt-repository ppa:fenics-packages/fenics
```
1. Then install the legacy dolfin backend: `sudo apt install fenics`
2. or the DOLFINx backend: `sudo apt-get install fenicsx`

To install *twoasis*, install all dependencies (not all are required), clone the repository and install the package:
```
sudo apt-get install python3-mpi4py python3-h5py \
    python3-h5py-mpi python3-cppimport python3-pybind11 libparmetis-dev \
    python3-matplotlib python3-gmsh python3-meshio python3-pytest python3-pytest-xdist \
    python3-nanobind nanobind-dev cmake g++ python3-scipy python3-skimage
python3 -m pip install --user --break-system-packages --no-deps meshpy

git clone https://github.com/gautelinga/twoasis.git
cd twoasis
python3 -m pip install --user --break-system-packages --no-deps -e .
```

*twoasis* also needs 

* [femtools](https://github.com/gautelinga/femtools): the common API of the two backends (`femtools.backend`), periodic boundary conditions, improved output, parallel partition of periodic 3D meshes
* [partractools](https://github.com/gautelinga/partractools): output for [partrac](https://github.com/gautelinga/partrac)
* [meshtools](https://github.com/gautelinga/meshtools): for the mesh scripts in `meshes/`

Install them the same way:
```
for repo in femtools partractools meshtools; do
    git clone https://github.com/gautelinga/$repo.git
done
git -C femtools checkout 696f85b
git -C partractools checkout 2026.2.1
git -C meshtools checkout 2026.2.0
for repo in femtools partractools meshtools; do
    python3 -m pip install --user --break-system-packages --no-deps -e $repo
done
```

Checkpoints and time series are written by all ranks in parallel through MPI-IO.
The `twoasis` command runs one OpenMP and BLAS thread per process (setting
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` unless they are already set).
Do the same when calling twoasis modules from your own script.

## Running a simulation

The command
```
twoasis TPfracStep problem=PeriodicBubble2D T=0.5
```
runs the problem defined in `twoasis/problems/TPfracStep/PeriodicBubble2D.py` (a bubble in a periodic channel) up to time 0.5. 
Any parameter can be set on the command line as `key=value`. 
Values are parsed as JSON, so lists and dicts work:

```
twoasis TPfracStep problem=PeriodicBubble2D solver=IPCS dt=0.001 rho='[1, 0.1]' \
    pressure_krylov_solver='{"relative_tolerance": 1e-8}'
```

Parameter values are set from (in prioritized order): the command line, the problem file (`problem_parameters` in the problem's `.py` file), and the defaults in `twoasis/problems/TPfracStep/__init__.py`.
The most central ones:

| Parameter               | Meaning |
|-------------------------|---------|
| `T`, `dt`               | end time and time step |
| `solver`                | `BDF` (second order by default, `bdf_order=1` for first), `IPCS` (first order), or `Naive` (a slow reference IPCS) |
| `rho`, `mu`             | densities and viscosities of the two fluids, `[phi=1, phi=-1]` |
| `sigma`, `epsilon`, `M` | surface tension, interface width, mobility |
| `velocity_degree`       | polynomial degree of the velocity (1 or 2) |
| `phase_field_degree`    | polynomial degree of the phase field and chemical potential (default: the pressure's, P1) |
| `backend`               | `dolfin`, `dolfinx` or `auto` (see [Backends](#backends)) |
| `phase_refine`          | solve the phase field on the flow mesh refined this many times (0: the same mesh; BDF and IPCS) |
| `folder`                | where results are written |
| `save_step`, `checkpoint`, `stat_interval` | how often to write fields, checkpoints and statistics (in steps) |
| `checkpoint_minutes`    | also checkpoint once this many minutes of wall time have passed since the last checkpoint |
| `max_iter`, `max_error` | inner pressure-velocity passes per step, at most, and the L2 norm of the velocity's change below which they stop |
| `velocity_pressure_solve` | `passes` (default: the pressure-correction passes above) or `coupled`: each BDF step's velocity and pressure solved as one system, to `coupled_krylov_solver`'s tolerance, in a number of iterations meant not to grow as the mesh is refined; needs `velocity_degree` > `pressure_degree` |

Problems that run out of the box:

| Problem | Description |
|---------|-------------|
| `PeriodicBubble2D` | Bubble carried by flow in a periodic channel |
| `Bubble2D`         | The Hysing benchmark case |
| `BubbleOpen`       | Bubble in a through-flow between inflow and outflow boundaries |
| `OscBubble2D`      | Oscillating bubble        |
| `PlaneCouette`     | Plane Couette flow        |
| `RayleighTaylor2D` | Rayleigh-Taylor instability in 2D   |
| `TaylorBubble2D`   | A Taylor bubble in 2D               |
| `TaylorGreen2D`    | Taylor-Green vortex                 |
| `LaplaceBubble2D`  | A resting bubble, measures the pressure jump against the Young–Laplace law and spurious currents. |

The other problems read a mesh from `meshes/`, which you generate first with the scripts there, with Bernaise's mesh utility (see [Publications](#publications)), or with those in `rcp_beads/` of [meshtools](https://github.com/gautelinga/meshtools) for `Porous3D` and `Porous3DFlux`. For example, for `SimplePore2D`:

```
mkdir meshes && cd meshes && python3 /path/to/twoasis/meshes/lattice_mesh.py && cd ..
twoasis TPfracStep problem=SimplePore2D
```

In parallel, start it with MPI: `mpirun -np 4 twoasis TPfracStep problem=...`.

## Backends

The backend is set by the command line parameter `backend` or with the environment variable
`TWOASIS_BACKEND`. 
The backend is printed at start-up and recorded in `params.dat`:

```
twoasis TPfracStep problem=PeriodicBubble2D backend=dolfinx
```

Both backends run every problem and solver, serially or in parallel, agree within solver precision, read and write the same files, which restart on either backend and on any number of processes, and time series in the same XDMF layout. A DOLFINx run does not import dolfin, and vice versa.
The finite-element calls go through `twoasis.backend`, which takes its names from `femtools.backend.dolfin` or `femtools.backend.dolfinx`.

## Output and checkpoints

Each run writes to a new numbered folder, `<folder>/data/1/`, `.../2/`, with the following subfolders:

- `Timeseries/`: the velocity, pressure and phase field as XDMF files (e.g. for ParaView)
- `Stats/`: scalars over time written by the problem (e.g. `tdata.dat`),
- `Checkpoint/`: the latest state, for restarting.

To continue a run, point to its checkpoint and specify a later end time:

```
twoasis TPfracStep problem=PeriodicBubble2D restart_folder=<folder>/data/1/Checkpoint T=1.0
```
The restart may use a different number of processes than the initial run.

To kill a running simulation, add an empty `killtwoasis` file to the folder:
```
touch <folder>/data/1/killtwoasis
```
or to pause, and unpause, it:
```
touch <folder>/data/1/pausetwoasis
rm <folder>/data/1/pausetwoasis
```
A run checks for these files every 10 s or so.

## Your own problem

Copy a problem file to your working directory, e.g. `PeriodicBubble2D.py` as `MyProblem.py`, and run `twoasis TPfracStep problem=MyProblem` from that directory.
Any hook the file does not define falls back to the defaults in `twoasis/problems/TPfracStep/__init__.py` and `twoasis/problems/__init__.py`. 
A problem file defines the mesh (`mesh`), its parameters (`problem_parameters`), boundary markers and conditions (`mark_subdomains`, `create_bcs`), the initial state (`initialize`), and what to compute and write during the run (`pre_solve_hook`, `temporal_hook`). 
`PeriodicBubble2D.py` is a short example of each.

## Tests

```
cd tests
python3 -m pytest           # fast tests
python3 -m pytest -m slow   # convergence studies and a run of every problem
python3 -m pytest -n 8      # in parallel (pytest-xdist)
```

The cross-backend tests run both backends whatever is chosen; the others run on the backend `TWOASIS_BACKEND` names (dolfin by default):

```
TWOASIS_BACKEND=dolfinx python3 -m pytest
```

## Publications

Papers with simulations made with twoasis:

- G. Linga, K. Pierce, M. Moura, J. Mathiesen, F. Renard and T. Le Borgne, *Dynamic multiphase flow triggers chaotic mixing in porous media*, Proc. Natl. Acad. Sci. U.S.A. **123**, e2612884123 (2026), [doi:10.1073/pnas.2612884123](https://doi.org/10.1073/pnas.2612884123). Problems `Porous2D` and `Porous3DFlux`.
- M. Klahn, G. Linga, T. Le Borgne and J. Mathiesen, *Stability of trapped fluid clusters in two-phase porous media flow*, J. Fluid Mech. **1039**, A8 (2026), [doi:10.1017/jfm.2026.11807](https://doi.org/10.1017/jfm.2026.11807).

Bernaise is described in:

- G. Linga, A. Bolet and J. Mathiesen, *Bernaise: A flexible framework for simulating two-phase electrohydrodynamic flows in complex domains*, Front. Phys. **7**, 21 (2019), [doi:10.3389/fphy.2019.00021](https://doi.org/10.3389/fphy.2019.00021).

twoasis has been validated against its two-phase (uncharged) variant in Bernaise. Bernaise's mesh scripts also make the meshes of `Porous2D` and `Porous2DFlux` (`mesh=periodic_porous_2d`) and of `CylArr2D` (`mesh=cyl_arr_2d`): run `python3 generate_mesh.py mesh=... key=value ...` in Bernaise's `utilities/` folder, which writes to its `meshes/`.

## AI declaration
*twoasis* was primarily human-developed from its beginning in March 2022 until September 2026. Since then, AI tools (Claude Opus 5/5.5, Fable 5) have been used for bug fixing, code optimization, building an extensive test suite, and for adapting the code to the dolfinx backend (through femtools). Every commit has been approved and checked by the developers.
The user is responsible for the physical validity of the results.

## License

GNU LGPL v3 or later; see `COPYING` and `COPYING.LESSER`.
