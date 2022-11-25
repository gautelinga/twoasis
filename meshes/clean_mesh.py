import dolfin as df
import numpy as np
import argparse
import meshio
from meshtools.io import numpy_to_dolfin, dolfin_to_numpy

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Clean mesh")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("-L", type=float, default=1., help="Length")
    return parser.parse_args()

def xysort(xy, indx, indy):
    ind = np.lexsort((xy[:, indy], xy[:, indx]))    
    return xy[ind, :]

def get_key(xy):
    return (round(100000*xy[0]), round(100000*xy[1]))

def make_xyset(xy_0):
    xyset_0 = set()
    for xy in xy_0:
        key = get_key(xy)
        if not key in xyset_0:
            xyset_0.add(key)
        else:
            exit(" df ")
            return False
    return xyset_0

def make_xydict(xy_0):
    xydict_0 = dict()
    for xy in xy_0:
        key = get_key(xy)
        xydict_0[key] = xy
    return xydict_0


if __name__ == "__main__":
    args = parse_args()
    tol = 1e-5
    eps = 1e-5

    #msh = meshio.read(args.infile)
    #tetra_cells = []
    #for cell in msh.cells:
    #    if cell.type == "tetra":
    #        tetra_cells.append(cell.data)
    #tetra_cells = np.vstack(tetra_cells)

    #print(msh.cell_data_dict)

    #for key in msh.cell_data_dict["gmsh:geometrical"].keys():
    #    if key == "tetra":
    #        tetra_data = msh.cell_data_dict["gmsh:geometrical"][key]

    #print(tetra_cells.shape, tetra_data.shape)
    #tetra_cells = tetra_cells[tetra_data[:] == 1, :]

    #nodes_unique = np.unique(tetra_cells)
    #old2new = np.zeros(nodes_unique.max()+1, dtype=int)
    #for new, old in enumerate(nodes_unique):
     #   old2new[old] = new

    #for old_id, new_id in zip(nodes_unique, ra  nge(len(nodes_unique))):
    #    tetra_cells[tetra_cells == old_id] = new_id
    #tetra_cells = old2new[tetra_cells]
    #points = msh.points[nodes_unique, :]

    #tetra_mesh = meshio.Mesh(points=points, cells={"tetra": tetra_cells})
    #meshio.write(f"{args.infile}.tmp.xdmf", tetra_mesh)

    meshin = df.Mesh()

    #with df.XDMFFile(f"{args.infile}.tmp.xdmf") as infile:
    #    infile.read(mesh)
    with df.HDF5File(meshin.mpi_comm(), args.infile, "r") as infile:
        infile.read(meshin, "mesh", False)

    nodes, elems = dolfin_to_numpy(meshin)
    print("before cleaning:", nodes.shape, elems.shape)

    nodes_unique = np.unique(elems)
    old2new = np.zeros(nodes_unique.max()+1, dtype=int)
    for new, old in enumerate(nodes_unique):
        old2new[old] = new

    elems = old2new[elems]
    nodes = nodes[nodes_unique, :]
    print("after cleaning:", nodes.shape, elems.shape)

    mesh = numpy_to_dolfin(nodes, elems)

    #assert len(np.unique(mesh.cells())) == len(mesh.coordinates())
    
    x = mesh.coordinates()
    """
    for i in range(3):
        x[x[:, i] > args.L - tol, i] = args.L
        x[x[:, i] < 0. + tol, i] = 0.

    x = mesh.coordinates()
    for i in range(3):
        ctop = np.sum(x[:, i] == args.L)
        cbtm = np.sum(x[:, i] == 0.)
        print(cbtm, ctop)
    """

    x_max = x[:, :].max(axis=0)
    x_min = x[:, :].min(axis=0)

    print(x_max, x_min)

    # Check periodicity
    for d in range(3):

        print(f"dim={d}")
        xydims = [0, 1, 2]
        xydims.remove(d)
        ids_min = x[:, d] < x_min[d] + eps
        ids_max = x[:, d] > x_max[d] - eps
        xy_0 = x[ids_min, :][:, xydims]
        xy_1 = x[ids_max, :][:, xydims]

        xyset_0 = make_xyset(xy_0)
        xyset_1 = make_xyset(xy_1)
      
        #print("ERROR: ")
        assert(not len(xyset_1 - xyset_0))
        assert(not len(xyset_0 - xyset_1))

        xydict_0 = make_xydict(xy_0)

        x[ids_min, d] = 0.
        x[ids_max, d] = args.L

        xy_1_2 = np.array([xydict_0[get_key(_xy)] for _xy in xy_1])
        for dd in range(2):
            x[ids_max, xydims[dd]] = xy_1_2[:, dd]

        plt.plot(xy_0[:, 0], xy_0[:, 1], ".")
        plt.plot(xy_1[:, 0], xy_1[:, 1], ".")
        plt.show()


    with df.HDF5File(mesh.mpi_comm(), args.outfile + ".h5", "w") as outfile:
        outfile.write(mesh, "mesh")

    with df.XDMFFile(mesh.mpi_comm(), args.outfile + "_show.xdmf") as xdmff:
        xdmff.write(mesh)