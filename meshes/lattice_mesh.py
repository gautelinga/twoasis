"""Mesh of one pore throat, or of two throats with a pore between them, for the
SimplePore2D and PoreThroat2D problems.

The domain is [-Lx/2, Lx/2] x [-Ly/2, Ly/2], periodic in y, with a circular
obstacle of radius rad centred on the periodic boundary at x = 0. Across the
periodic boundary the two half-disks form one obstacle, so the pore throat at
x = 0 has width Ly - 2*rad. With -rad2 > 0 there are two obstacles instead, of
radii rad and rad2, centred at x = -pore/2 and x = pore/2: two throats, of widths
Ly - 2*rad and Ly - 2*rad2, with a pore between them.

Writes lattice_Lx{Lx}_Ly{Ly}_r{rad}_dx{res}.h5 (the mesh), or with two obstacles
throats_Lx{Lx}_Ly{Ly}_r{rad}_r2{rad2}_pore{pore}_dx{res}.h5, and .dat (one line
"x y r" per obstacle) to the current directory. Run it inside meshes/, where the
problems look for them.
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
    parser.add_argument("-rad2", type=float, default=0.0,
                        help="Radius of a second obstacle (0: one obstacle)")
    parser.add_argument("-pore", type=float, default=2.0,
                        help="Distance between the two obstacles' centres")
    return parser.parse_args()


def obstacles(rad, rad2=0.0, pore=2.0):
    """(x, r) of each obstacle's centre (on the periodic boundary) and radius."""
    if rad2 > 0:
        return [(-pore / 2, rad), (pore / 2, rad2)]
    return [(0.0, rad)]


def mesh_name(Lx, Ly, rad, res, rad2=0.0, pore=2.0):
    if rad2 > 0:
        return "throats_Lx{}_Ly{}_r{}_r2{}_pore{}_dx{}".format(Lx, Ly, rad, rad2, pore, res)
    return "lattice_Lx{}_Ly{}_r{}_dx{}".format(Lx, Ly, rad, res)


def make_mesh(Lx, Ly, grains, res, mshfile):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("lattice")
    occ = gmsh.model.occ
    box = occ.addRectangle(-Lx/2, -Ly/2, 0, Lx, Ly)
    disks = [occ.addDisk(x, y, 0, r, r) for x, r in grains for y in (-Ly/2, Ly/2)]
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
    fname = mesh_name(args.Lx, args.Ly, args.rad, args.res, args.rad2, args.pore)
    grains = obstacles(args.rad, args.rad2, args.pore)

    mshfile = "tmp_lattice.msh"
    make_mesh(args.Lx, args.Ly, grains, args.res, mshfile)
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
    np.savetxt(fname + ".dat", [[x, y, r] for x, r in grains for y in (-args.Ly/2, args.Ly/2)])
    print("Wrote {}.h5 ({} cells) and {}.dat".format(fname, mesh.num_cells(), fname))
