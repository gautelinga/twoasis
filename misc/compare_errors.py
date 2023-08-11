import numpy as np
import argparse
import os
import pickle
import matplotlib.pyplot as plt

"""
python3 misc/compare_errors.py -f ../twoasis_runs/taylorgreen2d_results/data/84 ../twoasis_runs/taylorgreen2d_results/data/85 ../twoasis_runs/taylorgreen2d_results/data/86 ../twoasis_runs/taylorgreen2d_results/data/87/ --order 1 -dt 0.001
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Compare errors")
    parser.add_argument("-f", "--folders", nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--order", default=1, type=int, help="Temporal order")
    parser.add_argument("-dt", default=None, type=float, help="Timestep")
    parser.add_argument("-N", default=None, type=int, help="Number of points")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    assert(args.N or args.dt)

    err_ = dict()

    for folder in args.folders:
        timeseriesfolder = os.path.join(folder, "Timeseries")
        statsfolder = os.path.join(folder, "Stats")

        paramsfile = os.path.join(timeseriesfolder, "params.dat")
        tdatafile = os.path.join(statsfolder, "tdata.dat")

        if not os.path.exists(paramsfile): 
            continue
        if not os.path.exists(tdatafile):
            continue

        with open(paramsfile, 'rb') as f:
            params = pickle.load(f)
        
        #print(params)
        N = params["N"]
        dt = params["dt"]
        temporal_order = params["bdf_order"]
        
        if temporal_order != args.order:
            continue

        if args.N and N != args.N:
            continue

        if args.dt and dt != args.dt:
            continue

        #print(N, temporal_order)

        tdata_ = np.loadtxt(tdatafile)
        #print(tdata_)
        key = N if args.dt else dt

        err_[key] = dict(
            u0=tdata_[-1, 4],
            u1=tdata_[-1, 5],
            p=tdata_[-1, 6],
            phi=tdata_[-1, 7]
        )

    x_ = np.array(list(sorted(err_.keys())))
    y_ = dict([(field, np.array([err_[x][field] for x in x_])) for field in err_[x_[0]].keys()])
    print(x_, y_)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x_, y_['u0'])
    ax[0].plot(x_, y_['u1'])
    
    ax[1].plot(x_, y_['p'])
    
    if args.dt:
        ax[0].plot(x_, x_.astype(float)**-2)
        ax[1].plot(x_, x_.astype(float)**-2)

    ax[0].loglog()
    ax[1].loglog()

    plt.show()