twoasis
=======

*twoasis* simulates two-phase flow using the incompressible Cahn–Hilliard–Navier–Stokes model, solved with the finite element method in legacy [FEniCS](https://fenicsproject.org/download/archive/) (dolfin 2019.2). 
It is a stripped-down fork of [Oasis](https://github.com/mikaem/Oasis) by Mikael Mortensen, and is closely related to [Bernaise](https://github.com/gautelinga/BERNAISE), which simulates two-phase electrohydrodynamic flow (see [Publications](#publications)).

## Installation

On Ubuntu 24.04, install legacy dolfin from the FEniCS PPA, then twoasis:

```
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get install python3-dolfin python3-mpi4py python3-h5py \
    python3-h5py-mpi python3-cppimport python3-pybind11 libparmetis-dev \
    python3-matplotlib python3-gmsh python3-meshio python3-pytest
git clone https://github.com/gautelinga/twoasis.git
cd twoasis
python3 -m pip install --user --break-system-packages --no-deps -e .
```

twoasis also needs 

* [femtools](https://github.com/gautelinga/femtools): periodic boundary conditions, improved output, partition of periodic 3D meshes in parallel
* [partractools](https://github.com/gautelinga/partractools): output for [partrac](https://github.com/gautelinga/partrac)
* [meshtools](https://github.com/gautelinga/meshtools): for the mesh scripts in `meshes/`

Install them the same way, at the releases twoasis is tested with:

```
sudo apt-get install python3-scipy python3-skimage
python3 -m pip install --user --break-system-packages --no-deps meshpy
for repo in femtools partractools meshtools; do
    git clone https://github.com/gautelinga/$repo.git
done
git -C femtools checkout 2026.2.2
git -C partractools checkout 2026.2.1
git -C meshtools checkout 2026.2.0
for repo in femtools partractools meshtools; do
    python3 -m pip install --user --break-system-packages --no-deps -e $repo
done
```

## Running a simulation

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
| `folder`                | where results are written |
| `save_step`, `checkpoint`, `stat_interval` | how often to write fields, checkpoints and statistics |

Problems that run without further setup:

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

The other problems read a mesh from `meshes/`, which you generate first with the scripts there, with Bernaise's for `Porous2D`, `Porous2DFlux` and `CylArr2D` (see [Publications](#publications)), or with those in `rcp_beads/` of [meshtools](https://github.com/gautelinga/meshtools) for `Porous3D` and `Porous3DFlux`. For example, for `SimplePore2D`:

```
mkdir meshes && cd meshes && python3 /path/to/twoasis/meshes/lattice_mesh.py && cd ..
twoasis TPfracStep problem=SimplePore2D
```

In parallel, start it with MPI: `mpirun -np 4 twoasis TPfracStep problem=...`.

## Output

Each run writes to a new numbered folder, `<folder>/data/1/`, `.../2/`, ...:

- `Timeseries/`: the velocity, pressure and phase field as XDMF, for ParaView; the
  pressure there (and in `Timestamps/`) is the physical pressure p + φg, while the
  solver's own p, in `Checkpoint/`, is the physical pressure minus φg,
- `Stats/`: scalars over time written by the problem (e.g. `tdata.dat`),
- `Timestamps/` (the porous-medium and tube problems): a partrac case, with `boundaries.xdmf`, the walls partrac's `wall_upgrade` uses,
- `Checkpoint/`: the latest state, for restarting.

To continue a run, point to its checkpoint and give a later end time:

```
twoasis TPfracStep problem=PeriodicBubble2D restart_folder=<folder>/data/1/Checkpoint T=1.0
```

The restart may use another number of processes. It takes its parameters from the checkpoint, except the Krylov solvers' settings; the command line can still change any of them, while the problem file's values are those the run started with.

## Your own problem

Copy a problem file to your working directory, e.g. `PeriodicBubble2D.py` as `MyProblem.py`, and run `twoasis TPfracStep problem=MyProblem` from that directory. Any hook the file does not define falls back to the defaults in `twoasis/problems/TPfracStep/__init__.py` and `twoasis/problems/__init__.py`. 
A problem file defines the mesh (`mesh`), its parameters (`problem_parameters`), boundary markers and conditions (`mark_subdomains`, `create_bcs`), the initial state (`initialize`), and what to compute and write during the run (`pre_solve_hook`, `temporal_hook`). 
`PeriodicBubble2D.py` is a short example of each.

## Tests

```
cd tests
python3 -m pytest           # fast tests, ~2 minutes
python3 -m pytest -m slow   # convergence studies and a run of every problem
```

## Publications

Papers with simulations made with twoasis:

- G. Linga, K. Pierce, M. Moura, J. Mathiesen, F. Renard and T. Le Borgne, *Dynamic multiphase flow triggers chaotic mixing in porous media*, Proc. Natl. Acad. Sci. U.S.A. **123**, e2612884123 (2026), [doi:10.1073/pnas.2612884123](https://doi.org/10.1073/pnas.2612884123). Problems `Porous2D` and `Porous3DFlux`.
- M. Klahn, G. Linga, T. Le Borgne and J. Mathiesen, *Stability of trapped fluid clusters in two-phase porous media flow*, J. Fluid Mech. **1039**, A8 (2026), [doi:10.1017/jfm.2026.11807](https://doi.org/10.1017/jfm.2026.11807).

Bernaise is described in:

- G. Linga, A. Bolet and J. Mathiesen, *Bernaise: A flexible framework for simulating two-phase electrohydrodynamic flows in complex domains*, Front. Phys. **7**, 21 (2019), [doi:10.3389/fphy.2019.00021](https://doi.org/10.3389/fphy.2019.00021).

twoasis has been validated against its two-phase (uncharged) variant in Bernaise. Bernaise's mesh scripts also make the meshes of `Porous2D` and `Porous2DFlux` (`mesh=periodic_porous_2d`) and of `CylArr2D` (`mesh=cyl_arr_2d`): run `python3 generate_mesh.py mesh=... key=value ...` in Bernaise's `utilities/` folder, which writes to its `meshes/`.

## License

GNU LGPL v3 or later; see `COPYING` and `COPYING.LESSER`.
