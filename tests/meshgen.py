"""Small meshes for the problems that read a mesh file, made with gmsh for the
smoke tests: a box with circular (2D) or spherical (3D) obstacles inside it,
periodic along the given axes (the opposite faces of the box are meshed alike),
written in dolfin's HDF5 layout, and the obstacle list (one line "x y r" or
"x y z r" per obstacle) for the problems that read one.

The meshes stand in for the ones made with Bernaise (Porous2D, CylArr2D) or
meshtools (Porous3D, Porous3DFlux, BeadContact3D): coarse, and with obstacles
that do not cross the periodic boundaries.
"""
import numpy as np

EPS = 1e-6


def obstacle_mesh(path, lower, upper, obstacles, periodic, res):
    """Mesh the box from lower to upper minus the obstacles (rows of centre and
    radius), periodic along the axes where `periodic` is True, with cells of
    size about res; write the mesh to path (dolfin's HDF5 layout)."""
    import gmsh
    from femtools.dolfin_layout import write_mesh_arrays

    dim = len(lower)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("obstacles")
        occ = gmsh.model.occ
        size = upper - lower
        if dim == 2:
            box = occ.addRectangle(*lower, 0, *size)
            holes = [occ.addDisk(x, y, 0, r, r) for x, y, r in obstacles]
        else:
            box = occ.addBox(*lower, *size)
            holes = [occ.addSphere(x, y, z, r) for x, y, z, r in obstacles]
        if holes:
            occ.cut([(dim, box)], [(dim, h) for h in holes])
        occ.synchronize()

        # Each entity on the upper face of a periodic axis is meshed as a
        # translated copy of the matching one on the lower face.
        for axis in (a for a in range(dim) if periodic[a]):
            shift = np.eye(4)
            shift[axis, 3] = size[axis]
            lo3, hi3 = np.r_[lower, [0.] * (3 - dim)], np.r_[upper, [0.] * (3 - dim)]
            face_lo, face_hi = lo3 - EPS, hi3 + EPS
            face_hi[axis] = lo3[axis] + EPS
            for _, tag in gmsh.model.getEntitiesInBoundingBox(*face_lo, *face_hi, dim - 1):
                bb = np.array(gmsh.model.getBoundingBox(dim - 1, tag))
                bb[[axis, axis + 3]] += size[axis]
                match = gmsh.model.getEntitiesInBoundingBox(*(bb[:3] - EPS), *(bb[3:] + EPS),
                                                            dim - 1)
                gmsh.model.mesh.setPeriodic(dim - 1, [match[0][1]], [tag], list(shift.ravel()))

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
        gmsh.model.mesh.generate(dim)
        tags, x, _ = gmsh.model.mesh.getNodes()
        _, cell_nodes = gmsh.model.mesh.getElementsByType(2 if dim == 2 else 4)
    finally:
        gmsh.finalize()

    # Keep the nodes the cells use, numbered from 0.
    x = np.asarray(x).reshape(-1, 3)[:, :dim]
    index = np.full(int(np.max(tags)) + 1, -1)
    index[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
    cells = index[np.asarray(cell_nodes, dtype=np.int64)].reshape(-1, dim + 1)
    used = np.unique(cells)
    renumber = np.full(len(x), -1)
    renumber[used] = np.arange(len(used))
    write_mesh_arrays(str(path), x[used], renumber[cells])


def write_obstacles(path, obstacles):
    np.savetxt(str(path), np.asarray(obstacles, float).reshape(len(obstacles), -1))
