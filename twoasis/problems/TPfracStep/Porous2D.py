__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path, getcwd
import pickle
# from .Porous3D import ccode_expr

class PBC(SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool(
            (near(x[0], -self.Lx / 2) or near(x[1], -self.Ly / 2))
            and (not (near(x[0], self.Lx / 2) or near(x[1], self.Ly / 2)))
            and on_boundary)

    def map(self, x, y):
        if near(x[0], self.Lx / 2) and near(x[1], self.Ly / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        elif near(x[0], self.Lx / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] - self.Ly

class Walls(SubDomain):
    def __init__(self, pos_, r_):
        self.pos_ = pos_
        self.r_ = r_
        SubDomain.__init__(self)

    def inside(self, x, on_bnd):
        x_ = np.outer(x, np.ones(len(self.pos_))).T
        dr_ = np.linalg.norm(x_ - self.pos_, axis=1)
        return any(dr_ < self.r_ + DOLFIN_EPS_LARGE)

def inlet(x, on_bnd):
    return on_bnd and near(x[0], 0)

def get_fname(Lx, Ly, rad, R, N, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, rad=rad, R=R, N=N, res=res, ext=ext)
    fname = "meshes/periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.{ext}".format(**meshparams)    
    return fname

def ccode_expr(checkerboard, L):
    expr_fhat = "0.5*(tanh(({xhat}+0.25)/(sqrt(2)*{eps})) - tanh(({xhat}-0.25)/(sqrt(2)*{eps})))"
    expr_xhat = "({N}*x[{d}]/{L}-({i}+0.5-{N}/2))"
    ccode = ""
    factors = []
    for d in range(len(checkerboard)):
        if checkerboard[d] > 0:
            expr_xhat_d = []
            expr_eps = "{N} / {L} * epsilon".format(N=checkerboard[d], L=L[d])
            for i in range(checkerboard[d]):
                expr_xhat_di = expr_xhat.format(N=checkerboard[d], d=d, i=i, L=L[d])
                expr_xhat_d.append(expr_xhat_di)
            factors.append("(2*(" + "+".join([expr_fhat.format(xhat=expr_xhat_di, eps=expr_eps) for expr_xhat_di in expr_xhat_d]) + ")-1.0)")
        else:
            factors.append("1")
    ccode = "*".join(factors)
    return ccode

def initialize_checkerboard_sat(phi_init, phi_target, checkerboard, L, epsilon, tol=1e-8):
    if abs(phi_target) > 1 - tol:
        phi_init.vector()[:] = np.sign(phi_target)
        return
    
    L = np.array(L)
    x0 = -L/2 # np.array([-Lx/2, -Ly/2])
    dims = range(len(L))
    Dx = L / np.array(checkerboard) / 2

    x_ = [interpolate(Expression(f"x[{dim}]", degree=1), phi_init.function_space()).vector()[:] for dim in dims]
    i_ = [np.floor((x_[dim]-x0[dim])/Dx[dim]) for dim in dims]
    xi_ = [x0[dim] + (i_[dim] + 0.5) * Dx[dim] for dim in dims]
    dxi_ = [abs(x_[dim] - xi_[dim]) for dim in dims]
    ir_ = sum([np.round((x_[dim]-x0[dim])/Dx[dim]) for dim in dims])
    psi_ = np.minimum.reduce(dxi_) * (-1)**ir_

    c = 0.
    vol = assemble(Constant(1.) * dx(domain=phi_init.function_space().mesh()))
    phi_mean = 0.
    it = 0
    # Newton iterations
    while abs(phi_mean - phi_target) > tol:
        info_blue(f"it={it}, c={c}, psi_mean={phi_mean}")
        phi_init.vector()[:] = np.tanh( (psi_ + c ) / (np.sqrt(2) * epsilon))
        phi_mean = assemble(phi_init * dx) / vol
        phi2_mean = assemble(phi_init**2 * dx) / vol
        c += np.sqrt(2) * epsilon * (phi_target - phi_mean) / (1 - phi2_mean)
        it += 1

    phi_init.vector()[:] = np.tanh((psi_ + c)/(np.sqrt(2) * epsilon))

# Create a mesh
def mesh(Lx, Ly, rad, R, N, res, **params):
    mesh = Mesh()
    #fname = "meshes/periodic_porous_Lx20_Ly10_rad0.25_N300_dx0.05.h5"
    fname = get_fname(Lx, Ly, rad, R, N, res, "h5")
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(mesh, "mesh", False)
    return mesh

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, commandline_kwargs, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
         restart_folder = commandline_kwargs["restart_folder"]
         restart_folder = path.join(getcwd(), restart_folder)
         f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder, 'params.dat'), 'rb')
         NS_parameters.update(pickle.load(f))
         NS_parameters['restart_folder'] = restart_folder
         globals().update(NS_parameters)
    else:
        NS_parameters.update(
            T=100.0,
            Lx=4,
            Ly=8,
            rad=0.5,
            N=19,
            res=0.05,
            R=0.55,
            dt=0.05,
            rho=[1, 1],
            mu=[1, 1],
            theta=np.pi/3,
            epsilon=0.05,
            sigma=5.0,
            M=0.0001,
            F0=[0., 10.],
            g0=[0., 0.],
            init_state="",
            checkerboard=[12, 18],
            phi_target=0.0,
            velocity_degree=1,
            folder="porous2d_results",
            plot_interval=10,
            stat_interval=10,
            timestamps_interval=10,
            save_step=10,
            checkpoint=10,
            print_intermediate_info=10,
            use_krylov_solvers=True,
            solver="IPCS")

        # Need to override these here for PBC compliance
        for key in ["Lx", "Ly"]:
            if key in commandline_kwargs:
                NS_parameters[key] = commandline_kwargs[key]

    #scalar_components += ["alfa", "beta"]
    #Schmidt["alfa"] = 1.
    #Schmidt["beta"] = 10.

    #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}
    NS_expressions.update(dict(
        constrained_domain=PBC(NS_parameters["Lx"], NS_parameters["Ly"])
    ))

def mark_subdomains(subdomains, dim, Lx, Ly, rad, R, N, res, **NS_namespace):
    fname = get_fname(Lx, Ly, rad, R, N, res, "dat")
    obst = np.loadtxt(fname)
    x = obst[:, :dim]
    r = obst[:, dim]
    wall = Walls(x, r)
    wall.mark(subdomains, 1)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, subdomains, **NS_namespace):
    bc_ux_wall = DirichletBC(V, 0, subdomains, 1)
    bc_uy_wall = DirichletBC(V, 0, subdomains, 1)
    return dict(u0=[bc_ux_wall, bc_ux_wall],
                u1=[bc_uy_wall, bc_uy_wall],
                p=[],
                phig=[])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Lx, Ly, init_state, checkerboard, mesh, V, Q, restart_folder,
               phi_target, **NS_namespace):
    if restart_folder is None:
        if init_state == "":
            if checkerboard is None:
                phi_init = interpolate(Expression(
                    #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
                    "tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
                    epsilon=epsilon, Ly=Ly, degree=2), VV['phig'].sub(0).collapse())
            elif False:
                ccode = ccode_expr(checkerboard, [Lx, Ly])
                phi_init = interpolate(Expression(ccode, epsilon=epsilon, degree=2), VV['phig'].sub(0).collapse())
            else:
                phi_init = Function(VV['phig'].sub(0).collapse(), name="phi")
                initialize_checkerboard_sat(phi_init, phi_target, checkerboard, [Lx, Ly], epsilon)
            assign(q_['phig'].sub(0), phi_init)
            q_1['phig'].vector()[:] = q_['phig'].vector()
        else:
            if not path.exists(init_state):
                info_red(f"ERROR: Initial state {init_state} does not exist.")
            phi_init = Function(VV['phig'].sub(0).collapse())
            Vv = VectorFunctionSpace(mesh, V.ufl_element().family(), V.ufl_element().degree(),
                                    constrained_domain=V.dofmap().constrained_domain)
            u_init = Function(Vv)
            with HDF5File(mesh.mpi_comm(), init_state, "r") as h5f:
                h5f.read(phi_init, "phi")
                h5f.read(u_init, "u")
                h5f.read(q_['p'], "p")
            assign(q_['phig'].sub(0), phi_init)
            assign(q_['u0'], u_init.sub(0))
            assign(q_['u1'], u_init.sub(1))
            for ui in q_:
                q_1[ui].vector()[:] = q_[ui].vector()[:]
                q_2[ui].vector()[:] = q_[ui].vector()[:]

        for ui in x_1:
            [bc.apply(x_1[ui]) for bc in bcs[ui]]
        for ui in x_2:
            [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, rad, res, dt, **NS_namespace):
    volume = assemble(Constant(1.) * dx(domain=mesh))
    statsfolder = path.join(newfolder, "Stats")
    timestampsfolder = path.join(newfolder, "Timestamps")

    # Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = AssignedVectorFunction(u_, name="u")

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(statsfolder):
        makedirs(statsfolder)

    if MPI.rank(MPI.comm_world) == 0 and not path.exists(timestampsfolder):
        makedirs(timestampsfolder)

    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "params.dat"), "a+") as ofile:
            keys = ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt"]
            for key in keys:
                ofile.write("{}={}\n".format(key, eval(key)))
        with open(path.join(timestampsfolder, "dolfin_params.dat"), "a+") as ofile:
            ofile.write("velocity_space=P{}\n".format(velocity_degree))
            ofile.write("pressure_space=P{}\n".format(pressure_degree))
            ofile.write("phase_field_space=P{}\n".format(velocity_degree))
            ofile.write("timestamps=timestamps.dat\n")
            ofile.write("mesh=mesh.h5\n")
            ofile.write("periodic_x=true\n")
            ofile.write("periodic_y=true\n")
            ofile.write("periodic_z=true\n")
            ofile.write("rho=1.0\n")
    with HDF5File(mesh.mpi_comm(),
                  path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume)

def write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder):
    uv()

    h5fname = "up_{}.h5".format(tstep)

    phi__, g__ = q_['phig'].split(deepcopy=True)
    phi__.rename("phi", "phi")

    with HDF5File(mesh.mpi_comm(), path.join(timestampsfolder, h5fname), "w") as h5f:
        h5f.write(uv, "u")
        h5f.write(p_, "p")
        h5f.write(phi__, "phi")

    if MPI.rank(MPI.comm_world) == 0:
        with open(path.join(timestampsfolder, "timestamps.dat"), "a+") as ofile:
            ofile.write("{:.6f} {}\n".format(t, h5fname))

def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, rho_,
                  sigma, epsilon, volume, statsfolder, timestampsfolder,
                  plot_interval, stat_interval, timestamps_interval,
                  uv, mesh,
                  **NS_namespace):
    info_red("tstep = {}".format(tstep))
    if tstep % plot_interval == 0 and False:
        plot(u_, title='Velocity')
        plt.show()
        plot(phi_, title='Phase field')
        plt.show()
        #plot(p_, title='Pressure')
        #plot(q_['alfa'], title='alfa')
        #plot(q_['beta'], title='beta')
    if tstep % stat_interval == 0:
        sigma_bar = sigma * 3./(2*np.sqrt(2))
        u0m = assemble(q_['u0'] * dx) / volume
        u1m = assemble(q_['u1'] * dx) / volume
        phim = assemble(phi_ * dx) / volume
        E_kin = 0.5*assemble(rho_ * (u_[0]**2 + u_[1]**2) * dx) / volume
        E_int = 0.5 * sigma_bar * epsilon * assemble((phi_.dx(0)**2 + phi_.dx(1)**2) * dx) / volume
        E_pot = 0.25 * sigma_bar / epsilon * assemble((1-phi_**2)**2 * dx) / volume
        # Do not forget boundary term in E_int !
        if MPI.rank(MPI.comm_world) == 0:
            with open(statsfolder + "/tdata.dat", "a") as tfile:
                tfile.write("{:d} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                    tstep, t, u0m, u1m, phim, E_kin, E_int, E_pot))
    if tstep % timestamps_interval == 0:
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    return dict()


def theend_hook(u_, p_, testing, **NS_namespace):
    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
