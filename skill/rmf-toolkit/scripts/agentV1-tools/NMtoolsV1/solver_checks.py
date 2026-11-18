"""Checked mean-field roots shared by nuclear-matter and beta-EOS solvers."""
from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import fsolve, least_squares

FIELD_RESIDUAL_TOL = 1e-8
BETA_RESIDUAL_TOL_MEV = 1e-6
HBARC = 197.326968


class FieldConvergenceError(RuntimeError):
    """No finite, positive-effective-mass root met the residual tolerance."""


def field_seeds(initial_guess, field_names):
    first = np.asarray(initial_guess, dtype=float)
    seeds = [first]
    baseline = np.zeros(len(field_names), dtype=float)
    for aliases, value in ((('s', 'sigma'), 20.0), (('o', 'omega'), 15.0)):
        for name in aliases:
            if name in field_names:
                baseline[field_names.index(name)] = value
                break
    alternate = baseline.copy()
    for aliases, value in ((('s', 'sigma'), 45.0), (('o', 'omega'), 30.0)):
        for name in aliases:
            if name in field_names:
                alternate[field_names.index(name)] = value
                break
    for candidate in (baseline, alternate, first * 0.8, first * 1.2):
        if not any(np.allclose(candidate, old, rtol=0, atol=1e-12) for old in seeds):
            seeds.append(candidate)
    return seeds


def checked_field_solve(residual, initial_guess, field_names, rho_b, effective_masses,
                        *, maxfev=20000, residual_tol=FIELD_RESIDUAL_TOL):
    """Return a verified root and diagnostics, or raise explicitly.

    EOMs are in MeV^3. We require max(abs(EOM))/max(hbarc^3*rho_b,1) <= 1e-8,
    finite fields and positive finite effective masses. The first seed is kept
    first for continuity with the released solver. Alternate seeds and scaled
    least-squares are retries, never substitutes for the residual check.
    """
    scale = max(abs(float(rho_b)) * HBARC**3, 1.0)
    attempts = []

    def check(solution):
        solution = np.asarray(solution, dtype=float)
        values = np.asarray(residual(solution), dtype=float)
        masses = np.asarray(effective_masses(solution), dtype=float)
        norm = float(np.max(np.abs(values)) / scale)
        valid = (np.all(np.isfinite(solution)) and np.all(np.isfinite(values))
                 and np.all(np.isfinite(masses)) and np.all(masses > 0)
                 and np.isfinite(norm) and norm <= residual_tol)
        return bool(valid), norm

    seeds = field_seeds(initial_guess, list(field_names))
    for seed in seeds:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                solution, info, ier, message = fsolve(
                    residual, seed, xtol=1e-10, maxfev=maxfev, full_output=True)
                valid, norm = check(solution)
            attempts.append(f'fsolve ier={ier}, residual={norm:.3e}, nfev={info.get("nfev")}')
            if ier == 1 and valid:
                return np.asarray(solution), {'method': 'fsolve', 'ier': int(ier),
                    'relative_residual': norm, 'nfev': int(info.get('nfev', 0))}
        except (ValueError, TypeError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            attempts.append('fsolve ' + type(exc).__name__)
    for seed in seeds[:3]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                fit = least_squares(lambda x: np.asarray(residual(x))/scale,
                                    seed, x_scale='jac', xtol=1e-12, ftol=1e-12,
                                    gtol=1e-12, max_nfev=1000)
                valid, norm = check(fit.x)
            attempts.append(f'least_squares success={fit.success}, residual={norm:.3e}')
            if fit.success and valid:
                return np.asarray(fit.x), {'method': 'least_squares', 'ier': int(fit.status),
                    'relative_residual': norm, 'nfev': int(fit.nfev)}
        except (ValueError, TypeError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            attempts.append('least_squares ' + type(exc).__name__)
    raise FieldConvergenceError(f'field root failed at rho_b={rho_b:.8g}: ' + ' | '.join(attempts))
