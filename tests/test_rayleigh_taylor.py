"""Linear growth of the Rayleigh-Taylor instability (RayleighTaylor2D).

A cosine perturbation of amplitude a0 = 0.005 (k*a0 = 0.03) grows as e^(gamma t)
once the initial transient has passed. The problem's defaults (Atwood number
1/2, g = 1, k = 2*pi, mu = 0.01 in both fluids, no surface tension) give
gamma = 1.4446 from the viscous dispersion relation for two semi-infinite
fluids (the walls are a wavelength away, a correction of order e^(-2*k*H) ~ 1e-6).
Measured growth rates (BDF and IPCS agree within 1%, and halving dt changes
nothing):

    epsilon   h      gamma    1 - gamma/theory
    0.04      1/32   1.335    7.6%
    0.02      1/64   1.392    3.7%

The diffuse interface slows the growth by an amount first order in epsilon;
extrapolated to epsilon = 0 the rate is 1.449, within 0.3% of theory.
"""
from pathlib import Path

import numpy as np
import pytest

from conftest import COLS, QUIET, golden_compare, load_tdata, run_twoasis

# Coarse level, on half the default height. Linear growth for t > 1.5.
RT = dict(QUIET, problem="RayleighTaylor2D", solver="BDF", Ly=2, A=0.005,
          res=0.03125, epsilon=0.04, dt=0.005, T=2.0, stat_interval=5)
FIT_START = 1.5  # start of the window where the amplitude grows exponentially


def dispersion_matrix(n, k, g, rho_l, mu_l, rho_h, mu_h):
    """Interface conditions for a normal mode e^(n t + i k x), light fluid
    below, heavy above (Chandrasekhar 1961, ch. X). The vertical velocity is
    A e^(kz) + B e^(q_l z) below and C e^(-kz) + D e^(-q_h z) above, with
    q^2 = k^2 + n rho/mu; the rows are the continuity of w, Dw and the
    tangential stress, and the jump of the normal stress by the weight of the
    displaced interface."""
    q_l, q_h = np.sqrt(k**2 + n * rho_l / mu_l), np.sqrt(k**2 + n * rho_h / mu_h)
    M = np.zeros((4, 4))
    for j, (d, s, rho, mu) in enumerate([(k, 1, rho_l, mu_l), (q_l, 1, rho_l, mu_l),
                                         (-k, -1, rho_h, mu_h), (-q_h, -1, rho_h, mu_h)]):
        p = (mu * (d**3 - k**2 * d) - rho * n * d) / k**2  # perturbation pressure
        M[:, j] = s, s * d, s * mu * (d**2 + k**2), -s * (p - 2 * mu * d)
        if s < 0:
            M[3, j] -= (rho_h - rho_l) * g / n
    return M


def growth_rate(k, g, rho_l, mu_l, rho_h, mu_h):
    """The unstable root of the viscous dispersion relation, by bisection
    below the inviscid rate."""
    det = lambda n: np.linalg.det(dispersion_matrix(n, k, g, rho_l, mu_l, rho_h, mu_h))
    lo, hi = 1e-3, np.sqrt((rho_h - rho_l) / (rho_h + rho_l) * g * k)
    assert np.sign(det(lo)) != np.sign(det(hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sign(det(mid)) == np.sign(det(lo)) else (lo, mid)
    return 0.5 * (lo + hi)


# The problem's defaults: rho = [3, 1], mu = [0.01, 0.01], g0 = [0, -1], Lx = 1.
GAMMA = growth_rate(2 * np.pi, 1., 1., 0.01, 3., 0.01)


def run_rt(workdir, **params):
    """Run RayleighTaylor2D; return (tdata, amplitude data)."""
    run_twoasis(workdir, **params)
    amp = Path(workdir) / "results" / "data" / "1" / "Stats" / "amplitude.dat"
    return load_tdata(workdir), np.loadtxt(str(amp))


def measured_rate(amplitude):
    """Slope of log(a) over t >= FIT_START."""
    t, a = amplitude[:, 1], amplitude[:, 2]
    window = t >= FIT_START - 1e-9
    return np.polyfit(t[window], np.log(a[window]), 1)[0]


@pytest.fixture(scope="session")
def rt(tmp_path_factory):
    return run_rt(tmp_path_factory.mktemp("rayleigh_taylor"), **RT)


def test_theory():
    """The dispersion relation reduces to sqrt(A g k) without viscosity, and
    viscosity slows the growth."""
    k = 2 * np.pi
    assert growth_rate(k, 1., 1., 1e-7, 3., 1e-7) == pytest.approx(np.sqrt(0.5 * k), rel=1e-3)
    assert GAMMA == pytest.approx(1.4446, abs=1e-4)


def test_exponential_growth(rt):
    """The amplitude grows exponentially, at a constant rate, while it is small."""
    _, amp = rt
    t, a = amp[:, 1], amp[:, 2]
    assert np.isfinite(amp).all()
    assert a[0] == pytest.approx(RT["A"], rel=0.01)
    assert np.all(np.diff(a) > 0)
    assert 2 * np.pi * a[-1] < 0.3, "no longer in the linear regime"
    rates = np.gradient(np.log(a), t)[t >= FIT_START - 1e-9]
    assert np.ptp(rates) < 0.02 * GAMMA


def test_growth_rate(rt):
    """The growth rate is within 10% of linear theory, and below it: the
    diffuse interface slows the growth."""
    _, amp = rt
    rate = measured_rate(amp)
    assert 0.9 * GAMMA < rate < GAMMA, \
        "growth rate {:.4f}, linear theory {:.4f}".format(rate, GAMMA)


def test_mass_conservation(rt):
    td, _ = rt
    phim = td[:, COLS["phim"]]
    assert np.max(np.abs(phim - phim[0])) < 1e-6


def test_golden(request, rt):
    _, amp = rt
    golden_compare(request, "rayleigh_taylor", {"rate": float(measured_rate(amp)),
                                                "a_end": float(amp[-1, 2])})


@pytest.mark.slow
def test_refinement(tmp_path, rt):
    """Halving epsilon and h halves the deficit 1 - gamma/theory (first order),
    and the extrapolation to epsilon = 0 matches theory within 1%."""
    _, coarse = rt
    _, fine = run_rt(tmp_path, **dict(RT, res=0.015625, epsilon=0.02, dt=0.0025, stat_interval=10))
    r_coarse, r_fine = measured_rate(coarse), measured_rate(fine)
    ratio = (GAMMA - r_coarse) / (GAMMA - r_fine)
    assert 1.7 < ratio < 2.3, "deficit ratio {:.2f}, expected ~2 (first order)".format(ratio)
    assert 2 * r_fine - r_coarse == pytest.approx(GAMMA, rel=0.01)
