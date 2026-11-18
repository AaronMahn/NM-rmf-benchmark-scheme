from pathlib import Path
import sys
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'skill/rmf-toolkit/scripts/agentV1-tools'
sys.path.insert(0, str(TOOLS))
from NMtoolsV1 import solver_checks as checks
from TOVtools.beta_eos import solve_beta_equilibrium_point


def test_false_success_with_large_residual_is_retried(monkeypatch):
    original = checks.fsolve
    calls = []
    def first_bad(fn, seed, **kwargs):
        calls.append(np.asarray(seed).copy())
        if len(calls) == 1:
            return np.array([100.0]), {'nfev': 1}, 1, 'reported success'
        return original(fn, seed, **kwargs)
    monkeypatch.setattr(checks, 'fsolve', first_bad)
    x, diagnostic = checks.checked_field_solve(lambda x: x-2.0, [0.0], ['s'], 0,
                                               lambda x: [939.0])
    assert len(calls) > 1
    assert x[0] == pytest.approx(2.0)
    assert diagnostic['relative_residual'] <= checks.FIELD_RESIDUAL_TOL


def test_unconverged_equations_do_not_return_an_iterate():
    with pytest.raises(checks.FieldConvergenceError, match='field root failed'):
        checks.checked_field_solve(lambda x: np.ones_like(x), [0.0], ['s'], 0,
                                   lambda x: [939.0], maxfev=20)


def test_negative_mass_root_is_rejected_even_with_zero_residual():
    with pytest.raises(checks.FieldConvergenceError):
        checks.checked_field_solve(lambda x: x+1, [0.0], ['s'], 0,
                                   lambda x: x, maxfev=20)


def test_normalized_residual_uses_documented_density_scale():
    x, d = checks.checked_field_solve(lambda x: (x-20)*1e6, [19.0], ['s'], .16,
                                     lambda x: [939-x[0]])
    assert x[0] == pytest.approx(20.0)
    assert d['relative_residual'] <= 1e-8


def test_generic_beta_solver_checks_final_chemical_residual():
    energy = lambda rn, rp: 939*(rn+rp) + 100*(rn-rp)**2
    pressure = lambda rn, rp: 100*(rn-rp)**2
    p = solve_beta_equilibrium_point(energy, pressure, .16)
    assert abs(p.delta_beta) <= 1e-6
    assert p.charge_neutrality == 0
    with pytest.raises(RuntimeError):
        solve_beta_equilibrium_point(energy, pressure, .16, max_iter=1)


def test_symbolic_chemical_potentials_do_not_assume_parameter_names():
    from NMtoolsV1.analytic_layer import build_analytic_model
    from TOVtools.tool_beta_eff import compile_beta_effective_model
    terms = ["sp.Integer(1)*psibar_i_dslash_psi-P['m_N']*psibar_psi",
             "-P['scalar_coupling']*s*psibar_psi",
             "-P['vector_coupling']*o*psibar_gamma0_psi",
             "-sp.Rational(1,2)*P['m_sigma']**2*s**2",
             "+sp.Rational(1,2)*P['m_omega']**2*o**2"]
    params = dict(m_N=939., scalar_coupling=-10., vector_coupling=12., m_sigma=500., m_omega=783.)
    analytic = build_analytic_model([f"P['{k}']" for k in params], ['s','o'],
        ['psibar_i_dslash_psi','psibar_psi','psibar_gamma0_psi',
         'psibar_gamma0_tau3_psi','psibar_tau3_psi'], terms)
    model = compile_beta_effective_model(analytic, params)
    x, rn, rp, h = [20.,15.], .12, .04, 1e-6
    mup, mun = model.chemical_potentials_numeric(x, rn, rp)
    ep = (model.energy_numeric(x,rn,rp+h)-model.energy_numeric(x,rn,rp-h))/(2*h)
    en = (model.energy_numeric(x,rn+h,rp)-model.energy_numeric(x,rn-h,rp))/(2*h)
    assert mup == pytest.approx(ep, abs=1e-4)
    assert mun == pytest.approx(en, abs=1e-4)


def test_numeric_beta_search_tries_past_discontinuous_brackets():
    from types import SimpleNamespace
    from NMtoolsV1.numeric_layer import kf_from_rho_fm3
    from TOVtools.beta_eos import solve_beta_equilibrium_point_numeric, electron_chemical_potential
    density = .16
    def mass_n(x):
        y = float(np.clip(x[0],1e-8,.5-1e-8))
        f = -1. if y < .2 else 1. if y < .3 else -1. if y < .4 else 10*(y-.4)
        target = np.sqrt(kf_from_rho_fm3(density*y)**2+1000**2) + electron_chemical_potential(density*y) + f
        return np.sqrt(target**2-kf_from_rho_fm3(density*(1-y))**2)
    model = SimpleNamespace(field_names=['r'],
        eom_numeric=lambda x,rn,rp: np.asarray(x)-rp/(rn+rp),
        mass_p_numeric=lambda x:1000., mass_n_numeric=mass_n,
        energy_numeric=lambda x,rn,rp,p:939*(rn+rp))
    point,_ = solve_beta_equilibrium_point_numeric(model,{},density,field_guess=[.1])
    assert point.rho_p/density == pytest.approx(.4,abs=1e-8)
    assert abs(point.delta_beta) <= 1e-6
