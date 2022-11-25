import argparse
import os
import meshio
from meshtools.io import numpy_to_dolfin
import numpy as np
import dolfin as df
#import matplotlib.pyplot as plt

gmsh_code_tube_header = """
L = {L};
res = {res};
x=0; y=0; z=0; R={R};
eps = 1e-3;
"""
gmesh_code_tube_body = """
SetFactory("OpenCASCADE");

Cylinder(11) = {x, y, z, 0, 0, L, R};

Geometry.OCCBoundsUseStl = 1;

MeshSize { PointsOf{ Volume{11}; }} = res;

Sxmin() = Surface In BoundingBox{x-R-eps, y-R-eps, z-eps, x+R+eps, y+R+eps, z+L+eps};
For i In {0:#Sxmin()-1}
    bb() = BoundingBox Surface { Sxmin(i) };
    Szmax() = Surface In BoundingBox { bb(0)-eps, bb(1)-eps, bb(2)-eps+L,
                                       bb(3)+eps, bb(4)+eps, bb(5)+eps+L };
    For j In {0:#Szmax()-1}
    bbZ() = BoundingBox Surface { Szmax(j) };
    bbZ(2) -= L;
    bbZ(5) -= L;
    If(Fabs(bbZ(0)-bb(0)) < eps && Fabs(bbZ(1)-bb(1)) < eps &&
        Fabs(bbZ(2)-bb(2)) < eps && Fabs(bbZ(3)-bb(3)) < eps &&
        Fabs(bbZ(4)-bb(4)) < eps && Fabs(bbZ(5)-bb(5)) < eps)
        Periodic Surface {Szmax(j)} = {Sxmin(i)} Translate {0,0,L};
    EndIf
    EndFor
EndFor
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and check tube mesh using gmsh.")
    parser.add_argument("-L", type=float, default=4, help="Length")
    parser.add_argument("-R", type=float, default=0.5, help="Radius")
    parser.add_argument("-res", type=float, default=0.1, help="Resolution")
    parser.add_argument("--show", action="store_true", help="Show (XDMF)")
    return parser.parse_args()

def xysort(xy):
    ind = np.lexsort((xy[:, 1], xy[:, 0]))    
    return xy[ind, :]

if __name__ == "__main__":
    args = parse_args()

    tmpname = "tmp_tube"
    with open(f"{tmpname}.geo", "w") as ofile:
        code = gmsh_code_tube_header.format(L=args.L, R=args.R, res=args.res) + gmesh_code_tube_body
        ofile.write(code)

    os.system(f"gmsh {tmpname}.geo -3")

    mesh = meshio.read(f"{tmpname}.msh")
    
    os.remove(f"{tmpname}.geo")
    os.remove(f"{tmpname}.msh")
    #os.remove(f"{tmpname}.geo.msh")

    print(mesh)
    nodes = mesh.points
    cells = [c for c in mesh.cells if c.type == "tetra"]
    assert(len(cells)==1)
    elems = cells[0].data

    eps = 1e-7
    z_max = nodes[:, 2].max()
    z_min = nodes[:, 2].min()
    L = z_max-z_min

    # Check periodicity
    xy_0 = xysort(nodes[nodes[:, 2] < z_min + eps, :2])
    xy_1 = xysort(nodes[nodes[:, 2] > z_max - eps, :2])
    dxymax = np.linalg.norm(xy_1-xy_0, axis=1).max()
    print(dxymax)
    assert(dxymax < eps)

    mesh = numpy_to_dolfin(nodes, elems)
    with df.HDF5File(mesh.mpi_comm(), "tube.h5", "w") as h5f:
        h5f.write(mesh, "mesh")

    if args.show:
        with df.XDMFFile(mesh.mpi_comm(), "tube_show.xdmf") as xdmff:
            xdmff.write(mesh)
