#!/usr/bin/env python

import sys, os
import numpy as np

def main():
    assert len(sys.argv) > 1
    statsfolder = os.path.join(sys.argv[1], "Stats")
    timestampsfolder = os.path.join(sys.argv[1], "Timestamps")
    tdatafile = os.path.join(statsfolder, "tdata.dat")
    paramsfile = os.path.join(timestampsfolder, "params.dat")

    if os.path.exists(statsfolder) and os.path.exists(tdatafile) and os.path.exists(timestampsfolder) and os.path.exists(paramsfile):
        data = np.loadtxt(tdatafile)
        params = dict()
        with open(paramsfile, "r") as infile:
            text = infile.read().split("\n")[:-1]
            for line in text:
                key, val = line.split("=")
                params[key] = eval(val)
        print(params)
    else:
        print("Could not find folder: {}".format(statsfolder))
        exit()

    d = 2*params["rad"]
    mu = np.mean(params["mu"])
    rho = np.mean(params["rho"])
    nu = mu/rho
    sigma = params["sigma"]
    fy = params["F0"][1]
    res = params["res"]
    M = params["M"]
    epsilon = params["epsilon"]

    t = data[:, 1]
    uy = data[:, 3]
    uy_mean = uy.mean()

    t_adv = d / uy_mean
    
    numbers = dict(
        Ca = mu * uy_mean / sigma,
        Oh = mu / np.sqrt(rho * sigma * d),
        Re = uy_mean * d / nu,
        Bo = fy * d**2 / sigma,
        Cn = epsilon / d,
        PePF = uy_mean * epsilon * d / (M * sigma) 
    )

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
    

if __name__ == "__main__":
    main()