__author__ = "Gaute Linga <gaute.linga@mn.uio.no>"
__date__ = "2023-06-5"
__copyright__ = "Copyright (C) 2023 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..TPfracStep import *
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, getcwd
from .Porous2D import Walls
import pickle
import h5py
from xml.etree import cElementTree as ET

#from addictif.common.utils import helpers

helper_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>

class Triangle {
public:
  Triangle(const dolfin::Cell cell){
    for (dolfin::VertexIterator v(cell); !v.end(); ++v)
    {
        const std::size_t pos = v.pos();
        xx_[pos] = v->x(0);
        yy_[pos] = v->x(1);
    }

    double j11 = xx_[1]-xx_[0];
    double j12 = yy_[1]-yy_[0];
    
    double j21 = xx_[2]-xx_[0];
    double j22 = yy_[2]-yy_[0];
    
    double det = j11*j22-j12*j21;
    double d = 1.0/det;

    g2x_ = j22*d;   g3x_ = -j12*d;
    g2y_ = -j21*d;  g3y_ = j11*d;
    g1x_ = -g2x_-g3x_;  g1y_ = -g2y_-g3y_;
  }
  void linearbasis(double r,
                   double s,
                   double t,
                   std::vector<double> &N) const
  {
    N[0] = r;
    N[1] = s;
    N[2] = t;
  }
  
  void linearderiv(std::vector<double> &Nx,
                   std::vector<double> &Ny) const {
    Nx[0] = g1x_;
    Nx[1] = g2x_;
    Nx[2] = g3x_;

    Ny[0] = g1y_;
    Ny[1] = g2y_;
    Ny[2] = g3y_;
  }
private:
  std::array<double, 4> xx_, yy_;
  double g1x_, g1y_;
  double g2x_, g2y_;
  double g3x_, g3y_;
};

class AbsGrad : public dolfin::Expression
{
public:

  // Create expression with 1 component
  AbsGrad() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& ufc_cell) const override
  {
    const uint cell_index = ufc_cell.index;
    const dolfin::Cell dolfin_cell(*a->function_space()->mesh(), cell_index);
    //dolfin_cell.get_cell_data(ufc_cell);
    const dolfin::FiniteElement element = *a->function_space()->element();

    std::vector<double> coordinate_dofs;
    dolfin_cell.get_coordinate_dofs(coordinate_dofs);
    // const size_t dim = 3; // a->function_space()->mesh()->geometry().dim();
    const size_t ncoeff = 3;
 
    std::vector<double> coefficients_(ncoeff);

    a->restrict(coefficients_.data(), element, dolfin_cell,
                coordinate_dofs.data(), ufc_cell);

    std::vector<double> Nx_(ncoeff);
    std::vector<double> Ny_(ncoeff);

    Triangle tri(dolfin_cell);
    tri.linearderiv(Nx_, Ny_);

    double dadx = std::inner_product(Nx_.begin(), Nx_.end(), coefficients_.begin(), 0.0);
    double dady = std::inner_product(Ny_.begin(), Ny_.end(), coefficients_.begin(), 0.0);

    values[0] = sqrt(dadx * dadx + dady * dady);
  }
  std::shared_ptr<dolfin::Function> a;
};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<AbsGrad, std::shared_ptr<AbsGrad>, dolfin::Expression>
    (m, "AbsGrad")
    .def(py::init<>())
    .def_readwrite("a", &AbsGrad::a);
}
"""

helpers = compile_cpp_code(helper_code)

def parse_xdmf(xml_file, get_mesh_address=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    basedir = path.dirname(xml_file)

    dsets = []
    timestamps = []

    geometry_found = not get_mesh_address
    topology_found = not get_mesh_address

    for i, step in enumerate(root[0][0]):
        if step.tag == "Time":
            # Support for earlier dolfin formats
            timestamps = [float(time) for time in
                          step[0].text.strip().split(" ")]
        elif step.tag == "Grid":
            timestamp = None
            dset_address = None
            for prop in step:
                if prop.tag == "Time":
                    timestamp = float(prop.attrib["Value"])
                elif prop.tag == "Attribute":
                    dset_address = prop[0].text.split(":") # [1]
                    dset_address[0] = path.join(basedir, dset_address[0])
                elif not topology_found and prop.tag == "Topology":
                    topology_address = prop[0].text.split(":")
                    topology_address[0] = path.join(basedir, topology_address[0])
                    topology_found = True
                elif not geometry_found and prop.tag == "Geometry":
                    geometry_address = prop[0].text.split(":")
                    geometry_address[0] = path.join(basedir, geometry_address[0])
                    geometry_found = True
            if timestamp is None:
                timestamp = timestamps[i-1]
            dsets.append((timestamp, dset_address))
    if get_mesh_address and topology_found and geometry_found:
        return (dsets, topology_address, geometry_address)
    return dsets

def prep(x_list):
    """ Prepare a tuple representing coordinates to be used as key in a dict. """
    return tuple(x_list)

def load_scalar(phi_next, dset_phi, glob2loc, index=0):
    with h5py.File(dset_phi[0], "r") as h5f:
        phi_next.vector()[:] = h5f[dset_phi[1]][:, index][glob2loc]

def mpi_max(a):
    return MPI.max(MPI.comm_world, np.max(a))

class PBC2(SubDomain):
    def __init__(self, Lx):
        self.Lx = Lx
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        # return True if on left boundary AND NOT on the two slave edge
        return bool(near(x[0], -self.Lx / 2) and not near(x[0], self.Lx / 2) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.Lx / 2):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        else:  # near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1] #- self.Ly

class Bottom(SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -self.Ly/2)

class Top(SubDomain):
    def __init__(self, Ly):
        self.Ly = Ly
        super().__init__()

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self.Ly/2)

def get_fname(Lx, Ly, rad, R, N, res, ext):
    meshparams = dict(Lx=Lx, Ly=Ly, rad=rad, R=R, N=N, res=res, ext=ext)
    fname = "meshes/periodic_porous_Lx{Lx}_Ly{Ly}_r{rad}_R{R}_N{N}_dx{res}.{ext}".format(**meshparams)    
    return fname

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
            Lx=10,
            Ly=20,
            rad=0.5,
            N=88,
            res=0.05,
            R=0.6,
            dt=0.1,
            rho=[1, 1],
            mu=[10, 1],
            u0=0.001,
            y0=-8,
            theta=np.pi/3,
            epsilon=0.05,
            sigma=5.0,
            M=0.0001,
            F0=[0., 0.],
            g0=[0., 0.],
            velocity_degree=1,
            folder="porous2d_flux_results",
            plot_interval=10,
            stat_interval=10,
            timestamps_interval=10,
            save_step=100,
            checkpoint=10000,
            print_intermediate_info=10,
            use_krylov_solvers=True,
            solver="BDF",
            bdf_order=1,
            AB_projection_pressure=False,
            max_iter=10,                  # Number of inner pressure velocity iterations on timestep
            max_error=1e-3,               # Tolerance for inner iterations (pressure velocity iterations)
            iters_on_first_timestep=20,  # Number of iterations on first timestep
            initial_state=None,
            injected_phase=-1,
        )
        # Need to force this for proper PBCs
        if "Lx" in commandline_kwargs:
            NS_parameters["Lx"] = commandline_kwargs["Lx"]

    #scalar_components += ["alfa", "beta"]
    #Schmidt["alfa"] = 1.
    #Schmidt["beta"] = 10.

    #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}
    NS_expressions.update(dict(
        constrained_domain=PBC2(NS_parameters["Lx"])
    ))

def mark_subdomains(subdomains, dim, Lx, Ly, rad, R, N, res, **NS_namespace):
    fname = get_fname(Lx, Ly, rad, R, N, res, "dat")
    obst = np.loadtxt(fname)
    x = obst[:, :dim]
    r = obst[:, dim]
    wall = Walls(x, r)
    wall.mark(subdomains, 1)
    top = Top(Ly)
    top.mark(subdomains, 2)
    btm = Bottom(Ly)
    btm.mark(subdomains, 3)
    return dict()

def contact_angles(theta, **NS_namespace):
    return [(theta, 1)]

# Specify boundary conditions
def create_bcs(V, Q, u0, W, subdomains, injected_phase, **NS_namespace):
    bc_ux_wall = DirichletBC(V, 0, subdomains, 1)
    bc_uy_wall = DirichletBC(V, 0, subdomains, 1)
    bc_ux_top = DirichletBC(V, 0, subdomains, 2)
    bc_uy_top = DirichletBC(V, u0, subdomains, 2)
    bc_ux_btm = DirichletBC(V, 0, subdomains, 3)
    bc_uy_btm = DirichletBC(V, u0, subdomains, 3)
    #bc_p_top = DirichletBC(Q, 0, subdomains, 2)
    #bc_p_btm = DirichletBC(Q, 0, subdomains, 3)
    #bc_phig_top = DirichletBC(W.sub(0), -1 * injected_phase, subdomains, 2)
    bc_phig_btm = DirichletBC(W.sub(0), 1 * injected_phase, subdomains, 3)
    return dict(u0=[bc_ux_wall, bc_ux_btm, bc_ux_top],
                u1=[bc_uy_wall, bc_uy_btm, bc_uy_top],
                p=[],
                phig=[bc_phig_btm])


def average_pressure_gradient(F0, **NS_namespace):
    # average pressure gradient
    return Constant(tuple(F0))


def acceleration(g0, **NS_namespace):
    # (gravitational) acceleration
    return Constant(tuple(g0))


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, VV, Ly, y0, initial_state, restart_folder, mesh, V, **NS_namespace):
    if restart_folder is None:
        if initial_state is None:
            phi_init = interpolate(Expression(
                #"tanh((sqrt(pow(x[0], 2)+pow(x[1], 2))-0.45)/(sqrt(2)*epsilon))",
                #"tanh((x[1]-0.25*Ly)/(sqrt(2)*epsilon))-tanh((x[1]+0.25*Ly)/(sqrt(2)*epsilon))+1",
                "tanh((x[1]-y0)/(sqrt(2)*epsilon))",
                epsilon=epsilon, Ly=Ly, y0=y0, degree=2), VV['phig'].sub(0).collapse())
            assign(q_['phig'].sub(0), phi_init)
        elif initial_state.replace("/", "").endswith("Timeseries"):
            info_blue("initializing from: " + initial_state)
            u_file = path.join(initial_state, "u_from_tstep_0.xdmf")
            p_file = path.join(initial_state, "p_from_tstep_0.xdmf")
            phi_file = path.join(initial_state, "phi_from_tstep_0.xdmf")
            g_file = path.join(initial_state, "g_from_tstep_0.xdmf")

            dsets_u, topology_address, geometry_address = parse_xdmf(u_file, get_mesh_address=True)
            dsets_u = dict(dsets_u)
            dsets_p = dict(parse_xdmf(p_file, get_mesh_address=False))
            dsets_phi = dict(parse_xdmf(phi_file, get_mesh_address=False))
            dsets_g = dict(parse_xdmf(g_file, get_mesh_address=False))

            with h5py.File(geometry_address[0], "r") as h5f:
                nodes = h5f[geometry_address[1]][:]

            if MPI.rank(MPI.comm_world) == 0:
                xdict = dict([(prep(xloc), i) for i, xloc in enumerate(nodes)])
            else:
                xdict = None
            xdict = MPI.comm_world.bcast(xdict, root=0)
            glob2loc = [xdict[prep(xloc)] for xloc in V.tabulate_dof_coordinates()]

            t_ = sorted(dsets_u.keys())

            tkey = t_[-1]

            phi_init = Function(V)
            g_init = Function(V)
            load_scalar(phi_init, dsets_phi[tkey], glob2loc)
            load_scalar(g_init, dsets_g[tkey], glob2loc)
            assign(q_['phig'].sub(0), phi_init)
            assign(q_['phig'].sub(1), g_init)
            load_scalar(q_['u0'], dsets_u[tkey], glob2loc, index=0)
            load_scalar(q_['u1'], dsets_u[tkey], glob2loc, index=1)
            load_scalar(q_['p'], dsets_p[tkey], glob2loc)
            for key in q_.keys():
                q_1[key].vector()[:] = q_[key].vector()
                q_2[key].vector()[:] = q_[key].vector()
        else:
            info_blue("initializing from: " + initial_state)
            phi_init = read_phase_distribition(initial_state, mesh, q_)
            assign(q_['phig'].sub(0), phi_init)

        q_1['phig'].vector()[:] = q_['phig'].vector()
        q_2['phig'].vector()[:] = q_['phig'].vector()
        for ui in x_1:
            [bc.apply(x_1[ui]) for bc in bcs[ui]]
        for ui in x_2:
            [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(tstep, t, q_, p_, mesh, u_, newfolder, velocity_degree, pressure_degree, AssignedVectorFunction, 
                   F0, g0, mu, rho, sigma, M, theta, epsilon, rad, res, dt, Lx, Ly, S_DG0, **NS_namespace):
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
            keys = ["F0", "g0", "mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "res", "dt", "Lx", "Ly"]
            for key in keys:
                ofile.write("{}={}\n".format(key, eval(key)))
        with open(path.join(timestampsfolder, "dolfin_params.dat"), "a+") as ofile:
            ofile.write("velocity_space=P{}\n".format(velocity_degree))
            ofile.write("pressure_space=P{}\n".format(pressure_degree))
            ofile.write("phase_field_space=P{}\n".format(velocity_degree))
            ofile.write("timestamps=timestamps.dat\n")
            ofile.write("mesh=mesh.h5\n")
            ofile.write("periodic_x=true\n")
            ofile.write("periodic_y=false\n")
            ofile.write("periodic_z=false\n")
            ofile.write("rho=1.0\n") # ? 
    with HDF5File(mesh.mpi_comm(),
                  path.join(timestampsfolder, "mesh.h5"), "w") as h5f:
        h5f.write(mesh, "mesh")
    write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)

    phi__, g__ = q_['phig'].split(deepcopy=True)
    absgrad_g_ = Function(S_DG0, name="absgrad_g")

    return dict(uv=uv, statsfolder=statsfolder, timestampsfolder=timestampsfolder, volume=volume, absgrad_g_=absgrad_g_, g__=g__)

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

def read_phase_distribition(fname, mesh, q_):
    phi__, g__ = q_['phig'].split(deepcopy=True)
    with HDF5File(mesh.mpi_comm(), fname, "r") as h5f:
        h5f.read(phi__, "phi")
    return phi__


def temporal_hook(q_, tstep, t, dx, u_, p_, phi_, rho_, g__, absgrad_g_,
                  sigma, epsilon, volume, statsfolder, timestampsfolder,
                  stat_interval, timestamps_interval, M, res,
                  uv, mesh, dt,
                  **NS_namespace):
    info_red("tstep = {}".format(tstep))
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

    if False:
        with Timer("Computing time step"):
            assign(g__, q_["phig"].sub(1))
            absgrad_g_.interpolate(CompiledExpression(helpers.AbsGrad(), a=g__, degree=0))
            absgrad_g_max = mpi_max(absgrad_g_.vector()[:])
            umax = np.sqrt(mpi_max(q_['u0'].vector()[:]**2 + q_['u1'].vector()[:]**2))

            dt = 0.5 * res / (umax + M*absgrad_g_max) 

            info_blue(f"dt={dt}")

    #with XDMFFile(mesh.mpi_comm(), "absgrad_0.xdmf") as xdmff:
    #    xdmff.write(absgrad_g_, 0.)

    return dict(dt = dt)

def theend_hook(u_, p_, testing, **NS_namespace):
    u_norm = norm(u_[0].vector())
    if MPI.rank(MPI.comm_world) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))
