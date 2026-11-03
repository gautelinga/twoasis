#!/usr/bin/env python

import ast
import os
import sys

import numpy as np

def read_params(paramsfile):
    """key=value lines, the values as Python literals where they are."""
    params = dict()
    with open(paramsfile) as infile:
        for line in infile:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                try:
                    params[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    params[key] = val
    return params


def main():
    """Dimensionless numbers of a porous-medium run, from its Stats/tdata.dat
    and Timestamps/params.dat: those whose parameters the run wrote."""
    assert len(sys.argv) > 1
    tdatafile = os.path.join(sys.argv[1], "Stats", "tdata.dat")
    paramsfile = os.path.join(sys.argv[1], "Timestamps", "params.dat")
    for f in (tdatafile, paramsfile):
        if not os.path.exists(f):
            sys.exit("Could not find {}".format(f))
    data = np.loadtxt(tdatafile, ndmin=2)
    params = read_params(paramsfile)
    print(params)

    d = 2*params["rad"]
    mu = np.mean(params["mu"])
    rho = np.mean(params["rho"])
    nu = mu/rho
    sigma = params.get("sigma")
    fy = abs(params["F0"][1])
    M = params.get("M")
    epsilon = params.get("epsilon")

    t = data[:, 1]
    uy = data[:, 3]
    uy_mean = uy.mean()

    t_adv = d / uy_mean

    numbers = dict(
        fy = fy,
        uy_mean = uy_mean,
        t_adv = t_adv,
        Re = uy_mean * d / nu,
    )
    # The two-phase numbers, for a run that has a surface tension.
    if sigma is not None:
        numbers.update(Ca = mu * uy_mean / sigma, Oh = mu / np.sqrt(rho * sigma * d),
                       Bo = fy * d**2 / sigma)
    if epsilon is not None:
        numbers.update(Cn = epsilon / d)
    if None not in (sigma, epsilon, M):
        numbers.update(PePF = uy_mean * epsilon * d / (M * sigma))

    string = "\n".join([a + " = {" + a + "}" for a in numbers.keys()])
    print(string.format(**numbers))

    if "plot=true" in sys.argv:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.plot(t / t_adv, uy_mean*np.ones_like(t))
        ax.plot(t / t_adv, uy)
        ax.set_xlabel("$t / t_{adv}$")
        ax.set_ylabel("$<u_y>$")
        plt.show()

    if "copyfriendly=true" in sys.argv:
        header = "\t".join([a for a in numbers.keys()])
        print(header)
        string = "\t".join(["{" + a + "}" for a in numbers.keys()])
        print(string.format(**numbers))
    

if __name__ == "__main__":
    main()