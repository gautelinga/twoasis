#!/usr/bin/env -S yade -x

from yade import pack
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-R", type=float, default=0.1, help="Radius")
    parser.add_argument("-L", type=float, default=1.0, help="Size")
    return parser.parse_args()

def map_back(x, L):
    if x < 0:
        x += L
    elif x >= L:
        x -= L
    return x

if __name__ == "__main__":
    args = parse_args()

    factor = 1.26
    tol = 1e-3

    L_ = np.array([args.L, args.L, args.L])
    L_ext = factor * L_

    size = np.zeros(3)
    for i in range(100):
        rcp = pack.randomPeriPack(args.R, L_ext, rRelFuzz=0)
        size[:] = rcp.cellSize
        factor = args.L/size[0]
        if abs(factor - 1.0) < tol:
            break
        L_ext *= factor
        print(i, L_ext[0], size[0])
    print(i, L_ext[0], size[0])

    pos = np.zeros((len(rcp), 3))
    for i, (x, r) in enumerate(rcp):
        pos[i, :] = x/factor

    for d in range(3):
        print(pos[:, d].max())

    for i in range(len(pos)):
        for d in range(3):
            pos[i, d] = map_back(pos[i, d], L_[d])

    dmat = -np.ones((len(pos), len(pos)))
    for i in range(len(pos)):
        for j in range(i):
            dx = abs(pos[i, :] - pos[j, :])
            dx = np.minimum(dx, np.array(L_)-dx)
            dmat[i, j] = np.linalg.norm(dx)

    #plt.imshow(dmat)
    #plt.show()

    dist = sorted(dmat[dmat >= 0])
    #print(dist)
    plt.hist(dist, bins=256)
    plt.show()

    np.savetxt("rcp_pos.dat", pos)