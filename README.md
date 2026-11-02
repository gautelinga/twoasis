TwOasis
=======

TwOasis simulates two-phase flow: incompressible Navier–Stokes coupled to a
Cahn–Hilliard phase field (phi = 1 in one fluid, phi = -1 in the other), solved
with the finite element method in legacy FEniCS (dolfin 2019.2). It is a
stripped-down fork of [Oasis](https://github.com/mikaem/Oasis).

## Installation

On Ubuntu 24.04, install legacy dolfin from the FEniCS PPA, then TwOasis:

```
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get install python3-dolfin python3-mpi4py python3-h5py python3-h5py-mpi \
    python3-matplotlib python3-gmsh python3-meshio python3-pytest
git clone https://github.com/gautelinga/twoasis.git
cd twoasis
python3 -m pip install --user --break-system-packages --no-deps -e .
```

`-e` (editable) means changes you make to the code take effect without
reinstalling. This is the setup the tests run on (`.github/workflows/tests.yml`).

## Running a simulation

```
twoasis TPfracStep problem=PeriodicBubble2D T=0.5
```

runs the problem defined in `twoasis/problems/TPfracStep/PeriodicBubble2D.py`
(a bubble in a periodic channel) up to time 0.5. Any parameter can be set on the
command line as `key=value`; values are read as JSON, so lists and dicts work:

```
twoasis TPfracStep problem=PeriodicBubble2D solver=IPCS dt=0.001 rho='[1, 0.1]' \
    pressure_krylov_solver='{"relative_tolerance": 1e-8}'
```

A parameter's value is taken from, in order: the command line, the problem file
(`problem_parameters` in the problem's `.py` file), and the defaults in
`twoasis/problems/TPfracStep/__init__.py`, which also explain what they do.
The most common ones:

| Parameter | Meaning |
|---|---|
| `T`, `dt` | end time and time step |
| `solver` | `BDF` (second order by default, `bdf_order=1` for first) or `IPCS` (first order) |
| `rho`, `mu` | densities and viscosities of the two fluids, `[phi=1, phi=-1]` |
| `sigma`, `epsilon`, `M` | surface tension, interface width, mobility |
| `velocity_degree` | polynomial degree of the velocity (1 or 2) |
| `folder` | where results are written |
| `save_step`, `checkpoint`, `stat_interval` | how often to write fields, checkpoints and statistics |

Problems that run without further setup: `PeriodicBubble2D`, `Bubble2D`,
`BubbleOpen`, `OscBubble2D`, `PlaneCouette`, `RayleighTaylor2D`,
`TaylorBubble2D` and `TaylorGreen2D` (which compares with an exact solution).
The others read a mesh from `meshes/`, which you generate first with the scripts
there. For example, for `SimplePore2D`:

```
mkdir meshes && cd meshes && python3 /path/to/twoasis/meshes/lattice_mesh.py && cd ..
twoasis TPfracStep problem=SimplePore2D
```

In parallel, start it with MPI: `mpirun -np 4 twoasis TPfracStep problem=...`.

## Output

Each run writes to a new numbered folder, `<folder>/data/1/`, `.../2/`, ...:

- `Timeseries/`: the velocity, pressure and phase field as XDMF, for ParaView,
- `Stats/`: scalars over time written by the problem (e.g. `tdata.dat`),
- `Checkpoint/`: the latest state, for restarting.

To continue a run, point to its checkpoint and give a later end time:

```
twoasis TPfracStep problem=PeriodicBubble2D restart_folder=<folder>/data/1/Checkpoint T=1.0
```

## Your own problem

Copy a problem file to your working directory, e.g. `PeriodicBubble2D.py` as
`MyProblem.py`, change its first import to
`from twoasis.problems.TPfracStep import *`, and run
`twoasis TPfracStep problem=MyProblem` from that directory. A problem file
defines the mesh (`mesh`), its parameters (`problem_parameters`), boundary
markers and conditions (`mark_subdomains`, `create_bcs`), the initial state
(`initialize`), and what to compute and write during the run (`pre_solve_hook`,
`temporal_hook`). `PeriodicBubble2D.py` is a short example of each.

## Tests

```
cd tests
python3 -m pytest           # fast tests, ~2 minutes
python3 -m pytest -m slow   # convergence studies and a run of every problem
```

## License

GNU LGPL v3 or later; see `COPYING` and `COPYING.LESSER`.
