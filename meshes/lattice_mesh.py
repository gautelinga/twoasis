"""Mesh of a single pore throat for the SimplePore2D problem.

The domain is [-Lx/2, Lx/2] x [-Ly/2, Ly/2], periodic in y, with a circular
obstacle of radius rad centred on the periodic boundary at x = 0. Across the
periodic boundary the two half-disks form one obstacle, so the pore throat at
x = 0 has width Ly - 2*rad.

Writes lattice_Lx{Lx}_Ly{Ly}_r{rad}_dx{res}.h5 (the mesh) and .dat (one line
"x y r" per obstacle, used to mark the walls) to the current directory. Run it
inside meshes/, where SimplePore2D looks for them.
"""
import argparse
import os

import gmsh
import meshio
import numpy as np
import dolfin as df
from meshtools.io import numpy_to_dolfin


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a periodic 2D pore-throat mesh using gmsh.")
    parser.add_argument("-Lx", type=float, default=4.0, help="Length (x)")
    parser.add_argument("-Ly", type=float, default=2.0, help="Height (y), periodic")
    parser.add_argument("-rad", type=float, default=0.5, help="Obstacle radius")
    parser.add_argument("-res", type=float, default=0.05, help="Resolution")
    return parser.parse_args()


def make_mesh(Lx, Ly, rad, res, mshfile):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("lattice")
    occ = gmsh.model.occ
    box = occ.addRectangle(-Lx/2, -Ly/2, 0, Lx, Ly)
    disks = [occ.addDisk(0, y, 0, rad, rad) for y in (-Ly/2, Ly/2)]
    occ.cut([(2, box)], [(2, disk) for disk in disks])
    occ.synchronize()

    # Make each curve on the top boundary a periodic copy of the matching
    # curve on the bottom boundary.
    eps = 1e-6
    translate_y = [1, 0, 0, 0, 0, 1, 0, Ly, 0, 0, 1, 0, 0, 0, 0, 1]
    bottom = gmsh.model.getEntitiesInBoundingBox(
        -Lx/2-eps, -Ly/2-eps, -eps, Lx/2+eps, -Ly/2+eps, eps, 1)
    for _, curve in bottom:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, curve)
        top = gmsh.model.getEntitiesInBoundingBox(
            xmin-eps, ymin+Ly-eps, -eps, xmax+eps, ymax+Ly+eps, eps, 1)
        gmsh.model.mesh.setPeriodic(1, [top[0][1]], [curve], translate_y)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
    gmsh.model.mesh.generate(2)
    gmsh.write(mshfile)
    gmsh.finalize()


if __name__ == "__main__":
    args = parse_args()
    fname = "lattice_Lx{}_Ly{}_r{}_dx{}".format(args.Lx, args.Ly, args.rad, args.res)

    mshfile = "tmp_lattice.msh"
    make_mesh(args.Lx, args.Ly, args.rad, args.res, mshfile)
    msh = meshio.read(mshfile)
    os.remove(mshfile)

    # Keep only the triangles and the nodes they use (gmsh also stores the
    # geometry points, e.g. disk centres outside the domain).
    elems = [c for c in msh.cells if c.type == "triangle"][0].data
    used = np.unique(elems)
    renumber = -np.ones(len(msh.points), dtype=int)
    renumber[used] = np.arange(len(used))
    nodes = msh.points[used, :2]
    elems = renumber[elems]

    # Check that the nodes on the top and bottom boundaries match.
    eps = 1e-7
    xs_bottom = np.sort(nodes[nodes[:, 1] < -args.Ly/2 + eps, 0])
    xs_top = np.sort(nodes[nodes[:, 1] > args.Ly/2 - eps, 0])
    assert len(xs_bottom) == len(xs_top) and np.abs(xs_bottom - xs_top).max() < eps

    mesh = numpy_to_dolfin(nodes, elems)
    with df.HDF5File(mesh.mpi_comm(), fname + ".h5", "w") as h5f:
        h5f.write(mesh, "mesh")
    np.savetxt(fname + ".dat", [[0., -args.Ly/2, args.rad], [0., args.Ly/2, args.rad]])
    print("Wrote {}.h5 ({} cells) and {}.dat".format(fname, mesh.num_cells(), fname))
