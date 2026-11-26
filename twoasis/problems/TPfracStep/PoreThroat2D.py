"""Drainage through a single pore throat: a non-wetting phase (oil, phi = -1) pushed by
a pressure difference into a throat filled with the wetting phase (water, phi = +1), up
to the entry pressure or through it (a Haines jump).

The domain is [-Lx/2, Lx/2] x [-Ly/2, Ly/2], periodic in y, with a grain of radius rad
centred on the periodic boundary at x = 0 (mesh from meshes/lattice_mesh.py), so the
throat at x = 0 is Ly - 2 rad wide. The left end is held at physical pressure Dp, with
oil, and the right end at 0. The ends are close to the throat (Lx = 2 by default), so
that a jump moves the fluid near the throat, not a long column: a fixed pressure
difference across a long channel has to accelerate all of it, which with realistic
densities takes hundreds of time units. The phase field has no condition at either end
by default (end_phase "free"), so oil enters with the flow and leaves once through. Held
there instead ("held": oil at the inlet, water at the outlet), the pinned value differs
from the bulk's equilibrium value (shifted by the chemical potential a curved meniscus
sets), and the Cahn-Hilliard flux into the end then drives a steady flow that grows with
M. theta is the contact angle on the grain, measured through water (phi = +1).

With rad2 > 0 there is a second grain, of radius rad2, downstream: the grains sit at
x = -pore/2 and x = pore/2 (mesh throats_..., from meshes/lattice_mesh.py -rad2 -pore;
Lx = pore + 2 keeps one unit before the first throat and after the second). With
rad2 > rad the second throat is the narrower, so its entry pressure is the higher: past
the first throat's, the meniscus jumps across the pore and stops at the second (a Haines
jump), as long as the inlet stays below the second throat's entry pressure. Dp_e and
the pressures below are the first throat's.

In 2D a static meniscus is a circular arc of radius sigma/Dp. Where it meets the grain at
polar angle a (0 at the throat, the wall's slope there), the half-aperture is
h(a) = Ly/2 - rad cos(a) and the contact angle fixes the arc, so

    Dp(a) = sigma cos(theta - a) / h(a),

whose maximum over a is the entry pressure Dp_e. The run starts from the static meniscus
at dp_start * Dp_e in the first throat (on the stable branch, upstream of the maximum).
The inlet is then held at dp * Dp_e from t = 0, or, with dp_rate > 0, moved from
dp_start to dp at dp_rate (in units of Dp_e per unit time), starting at t = dp_t0, and
held there: dp = dp_start checks the static state, dp > 1 drives a jump, and a slow ramp
through 1 approaches it quasi-statically.

Stats/throat.dat, each stat_interval steps: tstep, t, the inlet's Dp, the oil volume,
the front position (the x at which the domain to its left holds the oil volume), the
largest velocity, the CFL number max|u| dt / res, the oil's share of the outlet, the
apex (the most downstream phi = 0 crossing of the centre line y = 0, oil upstream) and
the contact point's x on the (first) grain (the phi = 0 crossings a half and a whole
cell off the wall, extrapolated linearly to it), and the same on the second grain (nan
with one grain, or before the meniscus reaches it). The volume,
the front and the share count the bulk phases' shift from +-1 under a curved meniscus
(g = -Dp/2 at rest, a shift of -Dp epsilon/(4 sigma_bar) in both phases) over the whole
domain; the apex and the contact point, read off the phi = 0 contour, do not. A missing
crossing is written as nan. The run stops when the share passes `stop_share` (oil
breaks through).
"""
import math

import numpy as np
from scipy.optimize import brentq

from twoasis.backend import (
    assemble, dx, Expression, DirichletBC, Measure, mesh_comm, Probes)
from twoasis.common.utilities import set_subfunction
from twoasis.problems import info_blue
from twoasis.problems.TPfracStep.shared import (
    append_row, centred, Faces, load_mesh, mark_walls, PeriodicBox,
    PhysicalPressureBC, standard_stats, start_timestamps, write_tdata, write_timestamp)


def get_fname(Lx, Ly, rad, res, ext, rad2=0.0, pore=2.0):
    if rad2 > 0:
        return "meshes/throats_Lx{}_Ly{}_r{}_r2{}_pore{}_dx{}.{}".format(
            Lx, Ly, rad, rad2, pore, res, ext)
    return "meshes/lattice_Lx{}_Ly{}_r{}_dx{}.{}".format(Lx, Ly, rad, res, ext)


def grains(rad, rad2=0.0, pore=2.0):
    """(x, r) of each grain: its centre's x (on the periodic boundary) and radius."""
    if rad2 > 0:
        return [(-pore / 2, rad), (pore / 2, rad2)]
    return [(0.0, rad)]


def mesh(Lx, Ly, rad, res, rad2, pore, **params):
    return load_mesh(get_fname(Lx, Ly, rad, res, "h5", rad2, pore))


def static_pressure(a, sigma, theta, rad, Ly):
    """Dp of the static meniscus meeting the grain at polar angle a."""
    return sigma * np.cos(theta - a) / (Ly / 2 - rad * np.cos(a))


def entry_pressure(sigma, theta, rad, Ly):
    """(Dp_e, a_e): the largest static Dp, and where it is reached."""
    a = np.linspace(-(np.pi / 2 - theta) + 1e-6, np.pi / 2, 20001)
    p = static_pressure(a, sigma, theta, rad, Ly)
    i = int(np.argmax(p))
    return float(p[i]), float(a[i])


def static_meniscus(Dp, sigma, theta, rad, Ly):
    """The static meniscus at Dp on the stable branch: (contact x, half-aperture h, arc
    centre xc, radius R)."""
    Dp_e, a_e = entry_pressure(sigma, theta, rad, Ly)
    if not 0 < Dp < Dp_e:
        raise ValueError("no static meniscus at Dp = {} (entry pressure {})".format(Dp, Dp_e))
    a0 = -(np.pi / 2 - theta) + 1e-9
    a = brentq(lambda s: static_pressure(s, sigma, theta, rad, Ly) - Dp, a0, a_e)
    Px, h = rad * math.sin(a), Ly / 2 - rad * math.cos(a)
    R = sigma / Dp
    return Px, h, Px - math.sqrt(R**2 - h**2), R


def area_left(x, Lx, Ly, grain_list):
    """The domain's area left of x (each grain's two halves make one disk)."""
    area = (x + Lx / 2) * Ly
    for xg, r in grain_list:
        xc = min(max(x - xg, -r), r)
        area -= r**2 * (math.pi - math.acos(xc / r)) + xc * math.sqrt(r**2 - xc**2)
    return area


def front_position(volume, Lx, Ly, grain_list):
    return brentq(lambda x: area_left(x, Lx, Ly, grain_list) - volume, -Lx / 2, Lx / 2)


def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        T=200.0,
        Lx=2.0,
        Ly=2.0,
        rad=0.5,
        rad2=0.0,               # > 0: a second grain downstream, of this radius
        pore=2.0,               # with rad2 > 0: the distance between the grains' centres
        res=0.02,
        dt=0.05,
        # water (phi = +1) and decane (phi = -1) in units of the throat scale, 1 mPa s and
        # sigma: rho = rho_phys sigma L / mu_ref^2 for L = 50 um, sigma = 52 mN/m
        rho=[2595, 1898],
        mu=[1.0, 0.92],
        theta=math.radians(30),
        epsilon=0.04,
        sigma=1.0,
        M=4e-4,                 # sqrt(M mu) = epsilon/2
        dp_start=0.8,           # the initial static meniscus, in units of Dp_e
        dp=1.1,                 # the inlet pressure from t = 0, in units of Dp_e
        dp_rate=0.0,            # > 0: the inlet moves from dp_start to dp at this rate
        dp_t0=0.0,              # the time the ramp starts (the inlet is at dp_start before)
        stop_share=0.5,
        timestamps_dt=None,     # > 0: Timestamps each time t passes a multiple of it
                                # (for a varying dt), instead of each timestamps_interval
        end_phase="free",       # "free" or "held"
        F0=[0., 0.],
        g0=[0., 0.],
        velocity_degree=2,
        bdf_order=2,
        velocity_pressure_solve="coupled",
        folder="porethroat2d_results",
        plot_interval=1000000,
        stat_interval=10,
        timestamps_interval=1000000,
        save_step=1000000,
        checkpoint=1000000,
        print_intermediate_info=100,
        solver="BDF",
    )


def constrained_domain(Ly, **params):
    return PeriodicBox({1: centred(Ly)})


def mark_subdomains(subdomains, mesh, Lx, **NS_namespace):
    """Mark the grain 1, the inlet 2 and the outlet 3."""
    mark_walls(subdomains, mesh)
    Faces(0, -Lx/2).mark(subdomains, 2)
    Faces(0, Lx/2).mark(subdomains, 3)
    return dict()


def inlet_dp(t, dp_start, dp, dp_rate, dp_t0=0.0):
    """The inlet pressure at time t, in units of Dp_e: dp, or with dp_rate > 0 the ramp
    from dp_start towards dp, held at dp once there."""
    if dp_rate <= 0:
        return dp
    step = dp_rate * max(t - dp_t0, 0.0)
    return min(dp, dp_start + step) if dp >= dp_start else max(dp, dp_start - step)


def create_bcs(V, Q, W, subdomains, sigma, theta, rad, Ly, dp, dp_start, dp_rate, dp_t0,
               end_phase, **NS_namespace):
    if end_phase not in ("free", "held"):
        raise ValueError("end_phase: 'free' or 'held', not {!r}".format(end_phase))
    Dp_e, _ = entry_pressure(sigma, theta, rad, Ly)
    phig = []
    if end_phase == "held":
        phig = [DirichletBC(W.sub(0), -1, subdomains, 2),
                DirichletBC(W.sub(0), 1, subdomains, 3)]
    return dict(u0=[DirichletBC(V, 0, subdomains, 1)],
                u1=[DirichletBC(V, 0, subdomains, 1)],
                p=[PhysicalPressureBC(Q, inlet_dp(0.0, dp_start, dp, dp_rate, dp_t0) * Dp_e,
                                      subdomains, 2),
                   PhysicalPressureBC(Q, 0, subdomains, 3)],
                phig=phig)


def initialize(q_, q_1, q_2, x_1, x_2, bcs, epsilon, sigma, theta, rad, rad2, pore, Ly,
               dp_start, **NS_namespace):
    Dp_e, _ = entry_pressure(sigma, theta, rad, Ly)
    x1 = grains(rad, rad2, pore)[0][0]
    Px, h, xc, R = static_meniscus(dp_start * Dp_e, sigma, theta, rad, Ly)
    Px, xc = Px + x1, xc + x1
    second = ""
    if rad2 > 0:
        Dp_e2, _ = entry_pressure(sigma, theta, rad2, Ly)
        second = "; second throat {:.4f} ({:.3f} Dp_e)".format(Dp_e2, Dp_e2 / Dp_e)
    info_blue("PoreThroat2D: Dp_e = {:.4f}{}; start at {:.2f} Dp_e, contact x = {:.4f}, "
              "apex x = {:.4f}".format(Dp_e, second, dp_start, Px, xc + R))
    # signed distance to the arc where it spans the throat, horizontal elsewhere;
    # oil (-1) on the inlet side
    set_subfunction(q_['phig'], 0, Expression(
        "tanh((fabs(x[1]) <= h ? (x[0] > xc ? sqrt(pow(x[0] - xc, 2) + pow(x[1], 2)) - R"
        " : x[0] - xc - R) : x[0] - Px) / (sqrt(2) * epsilon))",
        h=h, xc=xc, R=R, Px=Px, epsilon=epsilon, degree=2))
    q_1['phig'].vector()[:] = q_['phig'].vector()
    q_2['phig'].vector()[:] = q_['phig'].vector()
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def contour_points(Lx, Ly, grain_list, res):
    """Probe points for the phi = 0 contour: the centre line y = 0 every res/4, and for
    each grain two rings round its upper half's lower side, a half and a whole cell off
    the wall, by polar angle a (0 at the throat, x = xg + r sin a), every 0.25 degrees.
    Returns (points, the centre line's x, the angles)."""
    x = np.arange(-Lx / 2 + res, Lx / 2 - res, res / 4)
    a = np.radians(np.arange(-89.0, 89.0, 0.25))
    line = np.c_[x, np.zeros_like(x)]
    rings = [np.c_[xg + r * np.sin(a), Ly / 2 - r * np.cos(a)]
             for xg, rad in grain_list for r in (rad + res / 2, rad + res)]
    return np.r_[(line, *rings)], x, a


def upstream_crossing(s, phi):
    """The largest s at which phi goes from oil (< 0) to water (>= 0) along s, linearly
    interpolated; nan if there is none."""
    i = np.nonzero((phi[:-1] < 0) & (phi[1:] >= 0))[0]
    if not len(i):
        return float("nan")
    i = i[-1]
    return float(s[i] + (s[i + 1] - s[i]) * phi[i] / (phi[i] - phi[i + 1]))


def pre_solve_hook(newfolder, VV, Lx, Ly, rad, rad2, pore, res, sigma, theta, **NS_namespace):
    d = start_timestamps(
        ["mu", "rho", "sigma", "M", "theta", "epsilon", "rad", "rad2", "pore", "res", "dt",
         "Lx", "Ly", "dp", "dp_start", "dp_rate", "dp_t0"], newfolder=newfolder, VV=VV, Lx=Lx,
        Ly=Ly, rad=rad, rad2=rad2, pore=pore, res=res, sigma=sigma, theta=theta,
        **NS_namespace)
    points, line_x, wall_a = contour_points(Lx, Ly, grains(rad, rad2, pore), res)
    d.update(contour=(Probes(points.flatten(), VV['phig']), line_x, wall_a),
             Dp_e=entry_pressure(sigma, theta, rad, Ly)[0])
    return d


def start_timestep_hook(t, bcs, Dp_e, dp, dp_start, dp_rate, dp_t0, **NS_namespace):
    """The inlet's physical pressure for the step ending at t."""
    bcs['p'][0].physical = inlet_dp(t, dp_start, dp, dp_rate, dp_t0) * Dp_e


def temporal_hook(q_, x_, tstep, t, dt, u_, p_, phi_, rho_, sigma, epsilon, volume,
                  statsfolder, timestampsfolder, stat_interval, timestamps_interval, uv,
                  mesh, subdomains, Lx, Ly, rad, rad2, pore, res, dp, dp_start, dp_rate, dp_t0,
                  stop_share, contour, timestamps_dt, Dp_e, **NS_namespace):
    """Timestamps at each multiple of timestamps_dt (or each timestamps_interval steps);
    Stats/tdata.dat (standard_stats) and Stats/throat.dat each stat_interval steps; stop
    when oil breaks through at the outlet."""
    if (math.floor(t / timestamps_dt + 1e-9) > math.floor((t - dt) / timestamps_dt + 1e-9)
            if timestamps_dt else tstep % timestamps_interval == 0):
        write_timestamp(tstep, t, mesh, uv, q_, p_, timestampsfolder)
    if tstep % stat_interval:
        return None
    write_tdata(statsfolder, tstep, t, standard_stats(q_, u_, phi_, rho_, sigma, epsilon,
                                                      volume, mesh))
    comm = mesh_comm(mesh)
    v_oil = assemble(0.5 * (1 - phi_) * dx(domain=mesh))
    grain_list = grains(rad, rad2, pore)
    front = front_position(v_oil, Lx, Ly, grain_list)
    local = [x_[ui].get_local() for ui in ("u0", "u1")]
    umax = comm.allreduce(max(float(np.abs(a).max()) if a.size else 0.0 for a in local),
                          op=_MAX())
    ds = Measure("ds", domain=mesh, subdomain_data=subdomains)
    share = assemble(0.5 * (1 - phi_) * ds(3)) / Ly
    probes, line_x, wall_a = contour
    probes(q_['phig'])
    values = probes.array(N=0)
    probes.clear()
    if comm.rank == 0:
        phi = values[:, 0]
        apex = upstream_crossing(line_x, phi[:len(line_x)])
        n, m = len(line_x), len(wall_a)
        contact = []
        for k, (xg, r) in enumerate(grain_list):
            a_half = upstream_crossing(wall_a, phi[n + 2 * k * m:n + (2 * k + 1) * m])
            a_whole = upstream_crossing(wall_a, phi[n + (2 * k + 1) * m:n + (2 * k + 2) * m])
            contact.append(xg + r * math.sin(2 * a_half - a_whole))
        contact2 = contact[1] if len(contact) > 1 else float("nan")
        append_row(statsfolder + "/throat.dat", tstep,
                   "{} {} {} {} {} {} {} {} {} {} {}\n".format(
                       tstep, t, inlet_dp(t, dp_start, dp, dp_rate, dp_t0) * Dp_e, v_oil, front,
                       umax, umax * dt / res, share, apex, contact[0], contact2))
    if share > stop_share:
        info_blue("PoreThroat2D: oil is {:.0%} of the outlet, stopping".format(share))
        return dict(stop=True)
    return None


def _MAX():
    from mpi4py import MPI
    return MPI.MAX
