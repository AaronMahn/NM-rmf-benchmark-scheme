# Revised manuscript reproduction checks

`paper/validation_walecka/run.sh` runs the three Table 2 models. The commands below
recompute the Gaussian-fit results for revised Table 3. The reproduction checks use
Python 3.9.

## Table 2: Walecka-model validation

The table gives the paper's cited-reference values, the paper's printed NNStar row,
and the released pipeline output. The reference column has no reported stellar M-R
values. All differences between the released output and the printed NNStar values
are within the tolerances n0 +/- 0.001 fm^-3, E and Esym +/- 0.1 MeV, L +/- 0.5 MeV, K +/- 1 MeV, Mmax +/- 0.01 Msun, and R1.4 +/- 0.05 km.

| Model | Source | n0 (fm^-3) | E (MeV) | Esym (MeV) | L (MeV) | K (MeV) | Mmax (Msun) | R1.4 (km) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TM1 | Cited reference | 0.142 | -16.2 | 36.0 | 108 | 257 | — | — |
| TM1 | Paper NNStar | 0.145 | -15.9 | 36.6 | 109.9 | 275 | 2.21 | 14.60 |
| TM1 | Released pipeline | 0.14455 | -15.8709 | 36.6472 | 109.9287 | 275.0383 | 2.21113 | 14.6043 |
| NL3 | Cited reference | 0.148 | -16.3 | 38.2 | 121 | 246 | — | — |
| NL3 | Paper NNStar | 0.151 | -16.6 | 38.7 | 122.9 | 264 | 2.78 | 14.85 |
| NL3 | Released pipeline | 0.15138 | -16.5823 | 38.6576 | 122.8887 | 264.4930 | 2.77708 | 14.8467 |
| FSU-delta6.7 | Cited reference | 0.148 | -16.3 | 32.7 | 53.5 | 229 | — | — |
| FSU-delta6.7 | Paper NNStar | 0.149 | -16.4 | 32.2 | 52.5 | 231 | 2.09 | 12.98 |
| FSU-delta6.7 | Released pipeline | 0.14916 | -16.4322 | 32.2450 | 52.4856 | 231.0149 | 2.09122 | 12.9780 |

## Table 3 — TM1 and the Gaussian fits

The TM1 column uses the toolkit's paper-constraint scorer. The Gaussian columns
use `fine_observables` and `fine_score` from the A1 and A2 reference files. The
manuscript rounds the values shown here.

| Observable | Empirical input | TM1 | sigma6 Gaussian | sigma6+c4 Gaussian |
|---|---:|---:|---:|---:|
| n0 (fm^-3) | 0.16 +/- 0.05 | 0.144546 | 0.163537 | 0.165104 |
| E (MeV) | -16.0 +/- 1.0 | -15.87086 | -15.97656 | -15.96926 |
| K (MeV) | 230 +/- 30 | 275.03831 | 229.53021 | 231.37314 |
| Esym (MeV) | 30.9 +/- 1.9 | 36.64721 | 29.58864 | 30.89686 |
| L (MeV) | 52.5 +/- 17.5 | 109.92869 | 87.62185 | 52.58780 |
| m_sigma (MeV) | 600 +/- 200 | 511.00000 | 501.24051 | 534.70135 |
| Mmax (Msun) | >= 2.08 | 2.21115 | 2.07624 | 2.27329 |
| R1.4 (km) | 12.7 +/- 1.1 | 14.60789 | 13.08208 | 12.71645 |
| chi2 | — | 25.49122 | 4.877431 | 0.120310 |

From the repository root, install the numerical and plotting dependencies and run
the scorers with the same Python environment.

```sh
python -m pip install -r requirements.txt -r paper/referee_validation/requirements.txt
python skill/rmf-toolkit/scripts/bja_score.py --model tm1_paper --constraints paper_sigma6_tm1
python paper/sigma6_tm1/fit_run.py --evaluate-only --out /tmp/table3-gaussian-fits.json
python paper/sigma6_tm1/plot_sigma6.py --out /tmp/table3-mr.pdf
```

The evaluate-only script recomputes observables, individual chi2 terms and M-R
curves at the best-fit couplings using the fine settings of the referee-validation
backend. The Mmax term applies a one-sided penalty of width 0.07 Msun below
2.08 Msun. The two input files are [A1/gaussian.json](../paper/referee_validation/reference/A1/gaussian.json)
and [A2/gaussian_lambda.json](../paper/referee_validation/reference/A2/gaussian_lambda.json).

To rerun the optimisation and reproduce the joint likelihood and posterior results
of Appendix D, follow the commands in the [referee-validation README](../paper/referee_validation/README.md).
The joint results and posterior diagnostics are in `reference/A1/joint*.json`,
`reference/A1/posterior_*.json` and `reference/A2/joint_lambda.json` in that package.

## Checked ground-truth audit

The checked field solver requires both successful termination and a normalized equation
residual at most 1e-8, finite fields and positive finite effective masses. Beta equilibrium
also requires a final chemical-potential residual at most 1e-6 MeV. Failed densities are
reported, not skipped. Chemical potentials follow the symbolic model energy.

The full audit (`harness/gt/audit_targets.py`, published density grid, 120 s per task)
completed 92 six-vectors. Tasks 15, 53 and 76 failed beta-root validation; 22, 66 and 83
timed out; 37 and 42 have nonmonotonic cores and are excluded from the working GT.
The other 89 completed rows agree with stored M/R to 3.6e-10 Msun and exactly in radius.
Three completed rows change materially:

| Task | Stored M (Msun) | Checked M | Stored R (km) | Checked R |
|---|---:|---:|---:|---:|
| 14 | 1.529239810 | 1.396664538 | 11.440001 | 10.690001 |
| 31 | 1.913969025 | 1.691610399 | 11.340001 | 10.975001 |
| 78 | 1.454508506 | 1.307277905 | 11.905001 | 10.705001 |

For task 31 the historical EOS contains 37 points whose normalized field residual exceeds
1e-8 (maximum 5.140873); a small beta residual alone concealed these failed field roots.
The checked trace has maximum field residual 2.68e-11. The three forward benchmark tasks
8, 32 and 46 reproduce the stored M/R to 7.6e-11 Msun and exactly in radius on this grid.
On a uniform 0.02 fm^-3 grid, their M differences stay below 0.000180 Msun and R below
0.0051 km. This numerical agreement does not establish branch stability or a global minimum.

The four matter columns are now regenerated from explicit task definitions; among the 92
completed rows the maximum absolute differences are 1.82e-12 MeV (E), 2.16e-10 MeV fm^-3
(P), 8.39e-5 MeV (K), and 1.82e-6 MeV (Esym). The original random sampler/seed remains
unavailable. The shipped matrix is preserved; checked outputs and statuses are separate.
The Table 2 validation rerun with these checks retains the released values above at displayed precision.

## Historical ground-truth reconstruction

`harness/gt/recalibrate_mr.py` regenerated 98 ordinary rows; samples 22 and 76
need skip-and-continue beta-EOS solving and are completed by `harness/gt/fix2276.py`.
Across all 100 samples, the maximum relative difference in the M and R columns
against the shipped `data/gt/targets_100x6.npy` is `3.154e-9`, below the `1e-6`
requirement. The four matter columns are bit-identical to
`harness/gt/targets_100x6.original.npy`. Sample 0 reproduces the shipped values
M = 2.082166 Msun and R = 15.065001 km (relative errors below 1e-6).

This reconstruction preserves the original matter columns and reproduces historical M/R;
it does not check convergence. Use the checked audit above for verified numerical statuses.
Sampling provenance remains limited as documented in `harness/gt/README.md`.
