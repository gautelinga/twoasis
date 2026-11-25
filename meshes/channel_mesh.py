"""An unstructured triangle mesh of the channel [-L/2, L/2] x [-W/2, W/2], periodic along x (its
left and right edges carry the same nodes), for the Channel2D problem: meshes/channel_L<L>_W<W>_
res<res>.h5, the name Channel2D.meshfile_name reads."""
import argparse
import gmsh
import numpy as np
import dolfin as df
from meshtools.io import numpy_to_dolfin


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", type=float, default=4.0, help="length (periodic)")
    parser.add_argument("-W", type=float, default=1.0, help="width")
    parser.add_argument("-res", type=float, default=0.04, help="cell size")
    parser.add_argument("-seed", type=int, default=0, help="gmsh's random seed")
    return parser.parse_args()


if __name__ == "__main__":
    a = parse_args()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.RandomSeed", a.seed)
    gmsh.model.add("channel")
    rect = gmsh.model.occ.addRectangle(-a.L / 2, -a.W / 2, 0, a.L, a.W)
    gmsh.model.occ.synchronize()
    eps = 1e-6
    left = gmsh.model.getEntitiesInBoundingBox(-a.L/2 - eps, -a.W/2 - eps, -eps, -a.L/2 + eps, a.W/2 + eps, eps, 1)
    right = gmsh.model.getEntitiesInBoundingBox(a.L/2 - eps, -a.W/2 - eps, -eps, a.L/2 + eps, a.W/2 + eps, eps, 1)
    T = [1, 0, 0, a.L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, [right[0][1]], [left[0][1]], T)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), a.res)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :2]
    index = {t: i for i, t in enumerate(tags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    tri = np.array([index[t] for t in enodes[list(etypes).index(2)]]).reshape(-1, 3)
    gmsh.finalize()
    used = np.unique(tri)                       # drop nodes no triangle uses
    renum = -np.ones(len(nodes), dtype=int); renum[used] = np.arange(len(used))
    nodes, tri = nodes[used], renum[tri]
    xl = np.sort(nodes[np.abs(nodes[:, 0] + a.L/2) < 1e-9, 1])
    xr = np.sort(nodes[np.abs(nodes[:, 0] - a.L/2) < 1e-9, 1])
    assert len(xl) == len(xr) and np.abs(xl - xr).max() < 1e-9, "edges not periodic"
    mesh = numpy_to_dolfin(nodes, tri)
    name = "channel_L{}_W{}_res{}.h5".format(a.L, a.W, a.res)
    with df.HDF5File(mesh.mpi_comm(), name, "w") as h5f:
        h5f.write(mesh, "mesh")
    print(name, mesh.num_vertices(), "vertices", mesh.num_cells(), "cells, hmin/hmax",
          mesh.hmin(), mesh.hmax())
