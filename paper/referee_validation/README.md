# Validation package for the revised manuscript

Reusable code and minimal checked references behind the revised NNStar manuscript (Table 3, Appendix D) and the accompanying eNJL check:
(1) joint NICER likelihood and sigma6-TM1 fits (Gaussian vs joint), (2) the omega-rho (`c4`) extension,
(3) an independent eNJL reference implementation (Pais, Menezes & Providencia, PRC 93, 065805, arXiv:1603.01239v3).

Location: `paper/referee_validation/` of the release, beside `skill/rmf-toolkit/` and `paper/sigma6_tm1/`.
Nothing here modifies the released toolkit.

## Requirements
Python 3.9+; `pip install -r requirements.txt` (numpy, scipy, numba, sympy, emcee, requests). Reference values were
produced and smoke-checked with the pinned versions. Network access is needed only for `fetch_data.py`.

## Backend discovery (no sibling study folders)
`vpaths.py` resolves, relative to this folder: `../../skill/rmf-toolkit/scripts/agentV1-tools` (BPS crust,
`NMtoolsV1`, `TOVtools`) and `../sigma6_tm1` (`model.json` for backend validation and the TM1 plot).

| Variable | Meaning |
|---|---|
| `NNSTAR_TOOLS_DIR` | released `agentV1-tools` directory |
| `NNSTAR_SIGMA6_DIR` | Directory containing the TM1 `model.json` |
| `NNSTAR_VALIDATION_WORK` | scratch/output dir (default `./work.nosync`, git-ignored). Downloads -> `data/`, samples -> `samples/`, chains -> `mcmc/`, regenerated results -> `results/{A1,A2,A4}/` |

Regenerated results never overwrite `reference/`. Scripts that need a prior result read `work/results` first,
then `reference/`.

## Commands (run from this folder; `PY` = your Python)
Entry scripts have main guards. Imports do not download data or launch long calculations.
The fits/MCMC are long; they were *not* re-run when staging this package.

```sh
# NICER data: downloads Zenodo archives, extracts equal-weight (M,R) samples, checks md5 of the J0740/J0437 archives
$PY nicer_fit/fetch_data.py
# Likelihood grids (KDE, bandwidth multiplier --bw)
$PY nicer_fit/nicer.py [--bw 0.7|1.4]
# Fits: Gaussian, joint, +c4 extension (A2), bandwidth and radio-prior sensitivities
$PY nicer_fit/fit.py ; $PY nicer_fit/fit.py --joint
$PY nicer_fit/fit.py --lambda-v ; $PY nicer_fit/fit.py --joint --lambda-v
$PY nicer_fit/fit.py --joint --bw 0.7 ; $PY nicer_fit/fit.py --joint --bw 1.4 ; $PY nicer_fit/fit.py --joint --radio
# Posterior (emcee), as used for the quoted intervals
$PY nicer_fit/mcmc.py --steps 12000 --workers 6 --seed 1234
$PY nicer_fit/mcmc.py --steps 12000 --workers 4 --seed 5678     # add --resume --steps 8000 to extend
$PY nicer_fit/posterior_summary.py ; $PY nicer_fit/posterior_checks.py ; $PY nicer_fit/final_refine.py
# Validation: KDE FFT vs direct sum; explicit backend vs released symbolic model
$PY nicer_fit/validate.py ; $PY nicer_fit/validate_backend.py
# eNJL reference (all 11 Table I sets by default, or name models)
$PY enjl/run_a4.py [eNJL1 eNJL3sr1 ...]
$PY enjl/independent_check.py ; $PY enjl/convergence_check.py
```
Original seeds/step counts: seeds 1234 and 5678, 20,000 steps each after resumes, 4,000 discarded.
`fit.py` uses DE seed 20261112. Its TM1-based optimiser start vector also initializes the MCMC ensemble
with seed 5678. The ensemble with seed 1234 starts at the nominal joint fit.

For revised Table 3, `../sigma6_tm1/fit_run.py --evaluate-only` recomputes the
Gaussian fits from `reference/A1/gaussian.json` and `reference/A2/gaussian_lambda.json`.
For Appendix D, the joint fits, bandwidth and radio-prior sensitivities, posterior summary
and checks are in `reference/A1/joint*.json`, `reference/A1/posterior_*.json` and
`reference/A2/joint_lambda.json`.

## Data (not redistributed; fetched by `fetch_data.py`)
- J0030 (Vinciguerra et al. updated analyses): doi:10.5281/zenodo.8239000, PDT-U NICER x XMM run1 headline member.
  The 6.9 GB archive is streamed only to that member; the full-archive checksum is not verified.
- J0740: doi:10.5281/zenodo.10519473 (`mr_samples_and_contours.tar.gz`, md5 ca8e70eeca81feb71321f1a88a24f2f9).
- J0437: doi:10.5281/zenodo.13766753 (`headline_result_samples_and_contours.tar.gz`, md5 d015dc7c5782b770b2d08972f9e7692c).
Extracted-sample sha256 values are in `reference/A1/data_manifest.json`.

## Conventions
- Units MeV, fm; `H=197.326968` MeV fm. Couplings in original conventions (g2 in MeV, g_s6 in MeV^-2, others dimensionless).
- c4 term: +c4 omega^2 rho^2 / 2 in the Lagrangian. Lambda_v (g_omega^2 g_rho^2 convention) = c4 / (2 g_omega^2 g_rho^2); c4 is not Lambda_v.
- Likelihood: joint (M,R) KDE (full Scott covariance, FFT binned) divided by the original mass prior; score = -2 ln. Gaussian and joint
  absolute scores use different data/constants and must not be subtracted as a chi-square improvement.
- eNJL: E/A = eps/rho - M0; npe matter, BPS crust below 0.08 fm^-3, RK4 TOV dr=0.005 km, p_surface=1e-12 MeV fm^-3, rho_c=4.0 fm^-3.

## Caveats
- Optima are best-found values (DE+Powell+Nelder-Mead), not claimed unique/global; posterior intervals are conditional finite-chain
  estimates; walker Rhat is a diagnostic only. Priors/boundaries: see `reference/A1/PRIOR_AND_SOURCES.md`.
- Fit and MCMC results depend on the BPS table and TOV settings of this backend, validated against the released symbolic model only at the
  points in `reference/A1/backend_validation.json`.
- eNJL six-vectors: for the three omega-rho sets M,R at rho_c=4.0 are invalid (NaN); for the other eight they are formal (beyond M_max, outside model validity).
  See the `status` fields in `reference/A4/six_vectors.json`.
- Checked numbers in `reference/` come from the original validation runs; this package's staging only re-checked syntax/imports/CLI help and two spot evaluations (see staging RESULT).


Integration check: a complete `enjl/run_a4.py eNJL1` run reproduced the stored
formal six-vector, including M=1.936227625 Msun and R=9.750001 km. Its high-density
domain limitations remain; this check does not validate that continuation as a physical star.
