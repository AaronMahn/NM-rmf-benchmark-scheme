"""eNJL reference implementation.

Model: Pais, Menezes, Providencia, Phys. Rev. C 93, 065805 (2016), arXiv:1603.01239v3, Sec. II.A, Eqs. (1)-(8).

    omega = eps_kin + m rho_s - G_s rho_s^2 + G_v rho^2 + G_sv rho_s^2 rho^2 + G_rho rho_3^2
            + G_vrho rho^2 rho_3^2 + G_srho rho_s^2 rho_3^2 - mu_p rho_p - mu_n rho_n
    eps_kin = <psibar (gamma.p) psi> = sum_i [F1(M,kF_i) - F1(M,Lambda)] - M rho_s
    rho_s   = M sum_i [F0(M,kF_i) - F0(M,Lambda)]            (Dirac sea cut at |p| < Lambda)
    M       = m - 2 G_s rho_s + 2 G_sv rho_s rho^2 + 2 G_srho rho_s rho_3^2          (gap, Eq. 6)
    P       = -omega + eps_0  (vacuum subtracted),  eps = -P + mu_p rho_p + mu_n rho_n

The "- M rho_s" in eps_kin is implied by the paper's definition eps_kin = <psibar gamma.p psi>
(F1 contains sqrt(M^2+p^2)); with it, d omega / dM = 0 reproduces Eq. (6) exactly and the
chemical potentials Eqs. (7)-(8) are d eps / d rho_i. Without it the gap equation would differ.

Units inside: masses and momenta in fm^-1, densities in fm^-3, energy densities in fm^-4;
couplings as in Table I (G_s, G_v, G_rho in fm^2; G_sv, G_vrho, G_srho in fm^8);
Lambda and m given in MeV and divided by hbar c.

Mean-field branch. The paper fixes M by the gap equation (d omega / dM = 0). In vacuum and up to
roughly k_F ~ Lambda the chosen root is also the minimum of the Hartree energy E(M). Once a Fermi
momentum exceeds Lambda, E(M) has no minimum at physical M: it falls monotonically towards
M > M0 (and, for m != 0, towards M < 0), so a strict Ritz minimisation would run to the edge of any
search window. Following the paper we take the lowest-energy gap root with 0 <= M <= 1.05 M0
(M = 0 is always a root for m = 0) and record, at every density, how far below it the trial-mass
energy E(M) goes on the window [-1.2 M0, 1.2 M0] (`E_deficit`, MeV fm^-3; 0 = the root is the
Hartree minimum on that window).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

HBARC = 197.326968  # MeV fm, same constant as the benchmark toolkit
PI2 = math.pi ** 2

# Table I of the paper (verbatim from the arXiv v3 source, lines 151-163).
TABLE1 = {
    'eNJL1':          dict(Gs=4.855,  Gv=4.65,  Gsv=-6.583, Grho=0.5876,  Gvrho=0.0,  Gsrho=0.0, Lambda=388.189, m=0.0),
    'eNJL1wr1':       dict(Gs=4.855,  Gv=4.65,  Gsv=-6.583, Grho=0.5976,  Gvrho=-1.0, Gsrho=0.0, Lambda=388.189, m=0.0),
    'eNJL1wr2':       dict(Gs=4.855,  Gv=4.65,  Gsv=-6.583, Grho=0.6476,  Gvrho=-6.0, Gsrho=0.0, Lambda=388.189, m=0.0),
    'eNJL2':          dict(Gs=3.8,    Gv=3.8,   Gsv=-4.228, Grho=0.6313,  Gvrho=0.0,  Gsrho=0.0, Lambda=422.384, m=0.0),
    'eNJL2wr1':       dict(Gs=3.8,    Gv=3.8,   Gsv=-4.228, Grho=0.6413,  Gvrho=-1.0, Gsrho=0.0, Lambda=422.384, m=0.0),
    'eNJL3':          dict(Gs=1.93,   Gv=3.0,   Gsv=-1.8,   Grho=0.65,    Gvrho=0.0,  Gsrho=0.0, Lambda=534.815, m=0.0),
    'eNJL3sr1':       dict(Gs=1.93,   Gv=3.0,   Gsv=-1.8,   Grho=0.0269,  Gvrho=0.0,  Gsrho=0.5, Lambda=534.815, m=0.0),
    'eNJL1m':         dict(Gs=1.3833, Gv=1.781, Gsv=-2.943, Grho=0.7,     Gvrho=0.0,  Gsrho=0.0, Lambda=478.248, m=450.0),
    'eNJL1msr1':      dict(Gs=1.3833, Gv=1.781, Gsv=-2.943, Grho=0.0739,  Gvrho=0.0,  Gsrho=1.0, Lambda=478.248, m=450.0),
    'eNJL2m':         dict(Gs=1.078,  Gv=1.955, Gsv=-2.74,  Grho=0.75,    Gvrho=0.0,  Gsrho=0.0, Lambda=502.466, m=500.0),
    'eNJL2msr1':      dict(Gs=1.078,  Gv=1.955, Gsv=-2.74,  Grho=-0.1114, Gvrho=0.0,  Gsrho=1.0, Lambda=502.466, m=500.0),
}

# Table II of the paper (symmetric matter at saturation; rho0 in fm^-3, rest in MeV).
TABLE2 = {
    'eNJL1':     dict(rho0=0.148, BA=-16.34, K=267.26, Esym=33.0,  L=99.90),
    'eNJL1wr1':  dict(rho0=0.148, BA=-16.34, K=267.26, Esym=32.65, L=95.02),
    'eNJL1wr2':  dict(rho0=0.148, BA=-16.34, K=267.26, Esym=30.91, L=70.61),
    'eNJL2':     dict(rho0=0.148, BA=-15.56, K=231.13, Esym=33.0,  L=95.03),
    'eNJL2wr1':  dict(rho0=0.148, BA=-15.56, K=231.13, Esym=32.65, L=90.15),
    'eNJL3':     dict(rho0=0.148, BA=-15.69, K=239.70, Esym=31.65, L=85.26),
    'eNJL3sr1':  dict(rho0=0.148, BA=-15.69, K=239.70, Esym=29.91, L=64.45),
    'eNJL1m':    dict(rho0=0.148, BA=-16.05, K=233.75, Esym=32.46, L=86.20),
    'eNJL1msr1': dict(rho0=0.148, BA=-16.05, K=233.75, Esym=30.28, L=60.32),
    'eNJL2m':    dict(rho0=0.148, BA=-16.22, K=286.63, Esym=33.66, L=89.20),
    'eNJL2msr1': dict(rho0=0.148, BA=-16.22, K=286.63, Esym=31.13, L=59.04),
}

# Table III, "without pasta" rows (BPS outer crust + homogeneous matter above neutron drip),
# and M_max of the "with pasta" rows (crust-independent per the paper). Masses in Msun, R in km, rho in fm^-3.
TABLE3 = {
    'eNJL1':     dict(Mmax=2.607, R_Mmax=12.297, rhoc_Mmax=0.776, rho_chi=0.35,  R14=13.971, rhoc14=0.324, crust='no pasta'),
    'eNJL1wr1':  dict(Mmax=2.520, R_Mmax=12.326, rhoc_Mmax=0.796, rho_chi=0.35,  R14=13.773, rhoc14=0.343, crust='no pasta'),
    'eNJL2':     dict(Mmax=2.365, R_Mmax=11.477, rhoc_Mmax=0.916, rho_chi=0.45,  R14=13.708, rhoc14=0.350, crust='no pasta'),
    'eNJL2wr1':  dict(Mmax=2.195, R_Mmax=11.826, rhoc_Mmax=0.891, rho_chi=0.45,  R14=13.456, rhoc14=0.383, crust='no pasta'),
    'eNJL3':     dict(Mmax=2.289, R_Mmax=12.057, rhoc_Mmax=0.875, rho_chi=0.998, R14=13.931, rhoc14=0.339, crust='CP pasta'),
    'eNJL3sr1':  dict(Mmax=2.192, R_Mmax=11.445, rhoc_Mmax=0.966, rho_chi=0.998, R14=13.212, rhoc14=0.391, crust='CP pasta'),
    'eNJL1m':    dict(Mmax=2.072, R_Mmax=12.398, rhoc_Mmax=0.866, rho_chi=None,  R14=13.839, rhoc14=0.356, crust='CP pasta'),
    'eNJL1msr1': dict(Mmax=1.884, R_Mmax=11.521, rhoc_Mmax=1.013, rho_chi=None,  R14=12.809, rhoc14=0.449, crust='CP pasta'),
    'eNJL2m':    dict(Mmax=2.275, R_Mmax=12.380, rhoc_Mmax=0.846, rho_chi=None,  R14=14.168, rhoc14=0.326, crust='CP pasta'),
    'eNJL2msr1': dict(Mmax=2.030, R_Mmax=11.549, rhoc_Mmax=0.981, rho_chi=None,  R14=13.084, rhoc14=0.414, crust='CP pasta'),
}

PAPER_NAMES = {
    'eNJL1': 'eNJL1', 'eNJL1wr1': 'eNJL1ωρ1', 'eNJL1wr2': 'eNJL1ωρ2', 'eNJL2': 'eNJL2',
    'eNJL2wr1': 'eNJL2ωρ1', 'eNJL3': 'eNJL3', 'eNJL3sr1': 'eNJL3σρ1', 'eNJL1m': 'eNJL1m',
    'eNJL1msr1': 'eNJL1mσρ1', 'eNJL2m': 'eNJL2m', 'eNJL2msr1': 'eNJL2mσρ1',
}


def kf(rho):
    """Fermi momentum (fm^-1) of one nucleon species of density rho (fm^-3), spin degeneracy 2."""
    return (3.0 * PI2 * max(rho, 0.0)) ** (1.0 / 3.0)


def _ash(M, x):
    """M^2 asinh(x/|M|), -> 0 for M -> 0."""
    a = np.maximum(np.abs(M), 1e-150)
    return M * M * np.arcsinh(x / a)


def F0(M, x):
    """int_0^x dp p^2 / sqrt(M^2+p^2) / pi^2."""
    E = np.sqrt(x * x + M * M)
    return (x * E - _ash(M, x)) / (2.0 * PI2)


def F1(M, x):
    """int_0^x dp p^2 sqrt(M^2+p^2) / pi^2."""
    E = np.sqrt(x * x + M * M)
    return (x * (2.0 * x * x + M * M) * E - M * M * _ash(M, x)) / (8.0 * PI2)


@dataclass
class MeanField:
    M: float                 # fm^-1
    rho_s: float             # fm^-3
    E: float                 # fm^-4, NOT vacuum subtracted
    stationary: list = field(default_factory=list)   # all gap roots in the root window (fm^-1)
    E_deficit: float = 0.0   # MeV fm^-3: E(chosen root) - min E(M) on [-1.2 M0, 1.2 M0] (0 if none lower)
    M_lowest: float = 0.0    # MeV: trial mass at that lower E(M)


class ENJL:
    def __init__(self, name: str, p: dict | None = None, *, n_grid: int = 3001):
        self.name = name
        p = dict(TABLE1[name] if p is None else p)
        self.p = p
        self.Gs, self.Gv, self.Gsv = p['Gs'], p['Gv'], p['Gsv']
        self.Grho, self.Gvrho, self.Gsrho = p['Grho'], p['Gvrho'], p['Gsrho']
        self.L = p['Lambda'] / HBARC
        self.m = p['m'] / HBARC
        self.n_grid = n_grid
        self.Mhi = 3.0 * 939.0 / HBARC
        vac = self._vacuum()
        self.M0 = vac.M
        self.E0 = vac.E
        self.M0_MeV = self.M0 * HBARC
        self.Mroot = 1.05 * self.M0
        self.Mwin = 1.2 * self.M0

    # --- Hartree energy at fixed densities as a function of the trial mass M -----------------
    def _parts(self, M, kp, kn, rho, rho3):
        S = F0(M, kp) + F0(M, kn) - 2.0 * F0(M, self.L)
        rho_s = M * S
        Ghat = self.Gs - self.Gsv * rho * rho - self.Gsrho * rho3 * rho3
        return rho_s, Ghat

    def energy_of_M(self, M, rho_p, rho_n):
        kp, kn = kf(rho_p), kf(rho_n)
        rho, rho3 = rho_p + rho_n, rho_p - rho_n
        rho_s, Ghat = self._parts(M, kp, kn, rho, rho3)
        return (F1(M, kp) + F1(M, kn) - 2.0 * F1(M, self.L)
                + (self.m - M) * rho_s - Ghat * rho_s * rho_s
                + self.Gv * rho * rho + self.Grho * rho3 * rho3 + self.Gvrho * rho * rho * rho3 * rho3)

    def gap(self, M, rho_p, rho_n):
        kp, kn = kf(rho_p), kf(rho_n)
        rho_s, Ghat = self._parts(M, kp, kn, rho_p + rho_n, rho_p - rho_n)
        return M - self.m + 2.0 * Ghat * rho_s

    def _roots(self, lo, hi, n, rho_p, rho_n):
        grid = np.linspace(lo, hi, n)
        g = self.gap(grid, rho_p, rho_n)
        idx = np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
        return [brentq(self.gap, grid[i], grid[i + 1], args=(rho_p, rho_n), xtol=1e-15, rtol=1e-15, maxiter=200)
                for i in idx]

    def _vacuum(self) -> MeanField:
        """Vacuum: global minimum of E(M) over |M| < 3 x 939 MeV (E(M) is bounded there)."""
        lo = 1e-9 if self.m == 0.0 else -self.Mhi
        cands = ([0.0] if self.m == 0.0 else []) + self._roots(lo, self.Mhi, 2 * self.n_grid, 0.0, 0.0)
        Es = [float(self.energy_of_M(c, 0.0, 0.0)) for c in cands]
        j = int(np.argmin(Es))
        Egrid = self.energy_of_M(np.linspace(lo, self.Mhi, 2 * self.n_grid), 0.0, 0.0)
        if Egrid.min() < Es[j] - 1e-10 * max(1.0, abs(Es[j])) or abs(cands[j]) > 0.98 * self.Mhi:
            raise RuntimeError(f'{self.name}: vacuum gap root is not the Hartree minimum')
        return MeanField(M=float(cands[j]), rho_s=float(self._parts(cands[j], 0.0, 0.0, 0.0, 0.0)[0]),
                         E=Es[j], stationary=[float(c) for c in cands])

    def mean_field(self, rho_p, rho_n) -> MeanField:
        """Lowest-energy gap root with 0 <= M <= 1.05 M0 (paper's prescription, see module doc)."""
        cands = ([0.0] if self.m == 0.0 else []) + self._roots(1e-9, self.Mroot, self.n_grid, rho_p, rho_n)
        if not cands:
            raise RuntimeError(f'{self.name}: no gap root in [0, 1.05 M0] at rho_p={rho_p}, rho_n={rho_n}')
        Es = [float(self.energy_of_M(c, rho_p, rho_n)) for c in cands]
        j = int(np.argmin(Es))
        Mbest = float(cands[j])
        win = np.linspace(-self.Mwin, self.Mwin, 2 * self.n_grid)
        Ewin = self.energy_of_M(win, rho_p, rho_n)
        k = int(np.argmin(Ewin))
        deficit = max(0.0, (Es[j] - float(Ewin[k])) * HBARC)
        kp, kn = kf(rho_p), kf(rho_n)
        rho_s, _ = self._parts(Mbest, kp, kn, rho_p + rho_n, rho_p - rho_n)
        return MeanField(M=Mbest, rho_s=float(rho_s), E=Es[j], stationary=[float(c) for c in cands],
                         E_deficit=deficit if deficit > 1e-9 else 0.0, M_lowest=float(win[k]) * HBARC)

    # --- thermodynamics (MeV, fm^-3) -------------------------------------------------------
    def state(self, rho_p, rho_n):
        mf = self.mean_field(rho_p, rho_n)
        M, rs = mf.M, mf.rho_s
        kp, kn = kf(rho_p), kf(rho_n)
        rho, r3 = rho_p + rho_n, rho_p - rho_n
        common = 2 * self.Gv * rho + 2 * self.Gsv * rho * rs * rs + 2 * self.Gvrho * r3 * r3 * rho
        iso = 2 * self.Grho * r3 + 2 * self.Gvrho * rho * rho * r3 + 2 * self.Gsrho * r3 * rs * rs
        mu_p = math.sqrt(kp * kp + M * M) + common + iso      # Eq. (7)
        mu_n = math.sqrt(kn * kn + M * M) + common - iso      # Eq. (8)
        eps = mf.E - self.E0                                  # vacuum subtracted, fm^-4
        P = mu_p * rho_p + mu_n * rho_n - eps
        return dict(M_MeV=M * HBARC, rho_s=rs, eps=eps * HBARC, P=P * HBARC,
                    mu_p=mu_p * HBARC, mu_n=mu_n * HBARC, n_stationary=len(mf.stationary),
                    stationary_MeV=[x * HBARC for x in mf.stationary],
                    E_deficit=mf.E_deficit, M_lowest_MeV=mf.M_lowest)

    def eps(self, rho_n, rho_p):
        """Energy density in MeV/fm^3 with the benchmark's (rho_n, rho_p) argument order."""
        return self.state(rho_p, rho_n)['eps']

    # --- symmetric-matter observables ------------------------------------------------------
    def EA(self, rho, delta=0.0):
        rp, rn = rho * (1 - delta) / 2, rho * (1 + delta) / 2
        return self.state(rp, rn)['eps'] / rho - self.M0_MeV

    def P_sym(self, rho):
        return self.state(rho / 2, rho / 2)['P']

    def K(self, rho, h=1e-4):
        """General K = 9 dP/drho (reduces to 9 rho^2 d^2(E/A)/drho^2 at P = 0)."""
        return 9.0 * (self.P_sym(rho + h) - self.P_sym(rho - h)) / (2 * h)

    def Esym_fd(self, rho, d=1e-3):
        return (self.EA(rho, d) - 2 * self.EA(rho, 0.0) + self.EA(rho, -d)) / (2 * d * d)

    def Esym_analytic(self, rho):
        """k_F^2/(6 E_F) + rho (G_rho + G_vrho rho^2 + G_srho rho_s^2); dM/d delta = 0 at delta = 0."""
        st = self.state(rho / 2, rho / 2)
        k = kf(rho / 2)
        M = st['M_MeV'] / HBARC
        val = k * k / (6 * math.sqrt(k * k + M * M)) + rho * (self.Grho + self.Gvrho * rho ** 2 + self.Gsrho * st['rho_s'] ** 2)
        return val * HBARC

    def L_slope(self, rho, h=1e-4):
        return 3 * rho * (self.Esym_analytic(rho + h) - self.Esym_analytic(rho - h)) / (2 * h)

    def saturation(self):
        rho0 = brentq(self.P_sym, 0.10, 0.20, xtol=1e-12)
        return dict(rho0=rho0, BA=self.EA(rho0), K=self.K(rho0), Esym=self.Esym_analytic(rho0),
                    Esym_fd=self.Esym_fd(rho0), L=self.L_slope(rho0), M_star_over_M0=self.state(rho0 / 2, rho0 / 2)['M_MeV'] / self.M0_MeV)
