# Data, prior, integration measure, and interpretation

Public sources selected on 2026-11-12. `data_manifest.json` records exact members, sample counts and hashes.
`fetch_data.py` checks the two compact archive MD5s against the primary Zenodo API; the J0030 archive is streamed to
its selected member without fetching/checksumming the complete 6.9GB archive. Extracted two-column samples are in
`samples/{J0030,J0740,J0437}.npz` (M/M_sun, R/km); they have equal weights, not nested-sampling weights.
`verified_sources.bib` was obtained through the required `inspire_bib.py`, with titles/authors checked.

| Pulsar | Primary dataset DOI | Selected run and reason | Analysis marginal prior in M,R |
|---|---|---|---|
| J0030+0451 | [10.5281/zenodo.8239000](https://zenodo.org/records/8239000) | Vinciguerra et al. 2024, joint NICER+XMM **PDT-U**, `PDTU_LR_1klp_et0p1_se0p8`, `run1`, `pdtu_NICERxXMM_preliminary_run1_nlive1k_eff0p8_et0p1_MMoff_noIS_tol-1post_equal_weights.dat`. Despite `preliminary` in the prefix this reproduces Table 7's headline quantiles (M=1.702, R=14.435) and ln Z=−42677.114. The highest-evidence hot-spot geometry in the paper, chosen before fitting, not because it fits our EOS. | Flat M in [1,3], flat R in [3r_g(1),16], with hard physical cuts. |
| J0740+6620 | [10.5281/zenodo.10519473](https://zenodo.org/records/10519473) | Salmi et al. 2024 joint NICER+XMM headline HR, 40k live points, SE .01, `J0740_gamma_NxX_lp40k_se001_mrsamples_post_equal_weights.dat`. | Marginal M~N(2.082, .0703²); R flat [3r_g(1),16], hard physical cuts. The joint (M,cos i) Gaussian has mean (2.082,.0427), covariance [[.0703²,.0131²],[.0131²,.00304²]]. |
| J0437−4715 | [10.5281/zenodo.13766753](https://zenodo.org/records/13766753) | Choudhury et al. 2024 **CST+PDT**, 3C50 background+AGN lower/upper bounds, high-MultiNest/low-XPSI headline; `J0437_3C50_BKG_AGN_hiMN_lowXPSI_mrsamples_post_equal_weights.dat`. | M~N(1.418, **.044²**), not .037² (the latter is the posterior width); R flat [3r_g(1),16], physical cuts. The mass and inclination priors are treated as independent by the original analysis. |

Sources: [Vinciguerra et al.](https://arxiv.org/abs/2308.09469), §2.2.2 and Table 7;
[Salmi et al.](https://arxiv.org/abs/2406.14466), Table 1;
[Choudhury et al.](https://arxiv.org/abs/2407.06789), §III.1 and Appendix A.
The X-PSI physical support uses polar radius / r_g >3 and log10 g_eff in [13.7,15.0] over the surface.
`nicer.support` implements the oblate polar-radius relation and the complete AlGendy–Morsink gravity polynomial,
including its interior extremum, following [the primary X-PSI source](https://github.com/xpsi-group/xpsi/blob/main/xpsi/surface_radiation_field/effective_gravity_universal.pyx).
Flat prior normalisations and original evidences are parameter-independent and are omitted.

## Baseline requested in the task brief

For each source we evaluate

    L_k(theta) ∝ integral_[1,min(3,Mmax)] dM  p_hat_k(M,R_theta(M)) / pi_k(M,R_theta(M)).

This is **conditional on a fixed flat EOS-side stellar-mass measure** on [1,3] M_sun, restricted to the stable
branch and original analysis support. The normalisation 1/(3−1) is parameter-independent and omitted. Thus the
original radio mass priors in J0740/J0437 are divided out: this is an X-ray effective likelihood conditional on the
original nuisance priors, **not a combined X-ray+radio posterior**. The task's original one-sided Mmax penalty,
[min(0,Mmax−2.08)/.07]², is retained. It is a soft penalty, not a hard Mmax>=2.08 cutoff.

J0740's M–inclination correlation remains in the conditional nuisance prior pi(cos i | M). Dividing the marginal
pi(M,R) is mathematically valid for this nuisance-marginalised effective likelihood. The 2D posterior alone cannot
remove/reanalyse that conditional nuisance prior or isolate a pure X-ray likelihood with different geometry priors.

A different EOS-side source mass measure changes the model. In particular, retaining the original timing
information requires multiplying the integrand by the original Gaussian mass likelihood once. The optional
`fit.py --joint --radio` and `--lambda-v` sensitivity fits do this **and omit the extra J0740 Mmax penalty**, avoiding
reuse of the same timing measurement. These are labelled `_radio`, not mixed with the requested flat-measure chains.
The maximum-mass truncation of the integral follows from the explicitly chosen source mass measure; it is not a
numerical failure.

## KDE and score constants

Use all equal-weight samples, a full 2D covariance Gaussian KDE, Scott factor N^(−1/6), covariance N^(−1/3) Cov(M,R).
`nicer.py` bins at dM=.0015 M_sun, dR=.015 km and convolves with the full correlated Gaussian kernel by FFT.
`validate.py` compares curve integrals against direct evaluation using all samples (no FFT, no sample thinning).
Gaussian tails are truncated at 8 kernel mass standard deviations for direct evaluation; the FFT grid omits values
below 1e−13 times the maximum density to suppress roundoff before dividing by a small prior.
The bandwidth is refitted at .7 and 1.4 times Scott. Raw absolute scores across bandwidths or Gaussian-vs-joint data
sets have arbitrary different additive constants and **must not be interpreted as chi-square improvements**.
A fixed source-specific scale max(p_hat/pi) is divided from each integral at each bandwidth. Within one bandwidth
and one mass-measure convention, score differences between EOS parameters are meaningful.

The joint score is the six NM/m_sigma Gaussian terms + one-sided Mmax term (baseline only) −2 sum ln L_k.
The MCMC posterior is proportional to exp(−score/2) times the stated flat coupling priors. No evidence calculation.

## Limits that must accompany the numerical result

- Results are conditional on PDT-U for J0030. Its high-radius mode is in tension with J0437. ST+PDT gives a different
  radius. This is hot-spot model uncertainty, not three independent measurements to multiply together. The J0030
  paper's sampling-resolution discussion around its **ST-U** tests does not invalidate the chosen PDT-U headline,
  but the low-live-point run and model dependence warrant caution.
- Inverse-prior reconstruction amplifies sparsely sampled mass tails. The .7–1.4 bandwidth refits measure a real
  sensitivity. The KDE-to-exact-KDE check only validates numerical convolution, not the original sampling or
  hot-spot model. There is no justified precision claim below that systematic.
- TOV radii are nonrotating; NICER radii are equatorial for rotating stars. No rotating-star correction was computed.
- The posterior has flat raw coupling priors **conditional on the stated numerical/branch domain**: positive
  effective mass, the physical scalar/vector branch, resolved stable stellar branch, successful core solve through
  1.8 fm^-3. Proposals failing those checks get zero probability. `failure_audit.json` audits 300 uniform raw prior
  draws: 8 core failures; two fail beyond an already resolved maximum, but their NM scores exceed 1000. This supports
  a negligible effect for those audited examples, not a proof that failures have no effect anywhere in the prior.
- No additional causality, potential-boundedness or phase-transition prior was imposed. Negative sextic couplings
  occur, including a boundary optimum. The credible intervals describe this inference setup, not a unique physical
  EOS posterior.

## Origin of 12.7 +/- 1.1 km

The manuscript cites Vinciguerra 2024 as though this were R_1.4. Its introduction quotes
[Riley et al. 2019](https://arxiv.org/abs/1912.05702): M=1.34^(+.15)_(-.16), R_eq=12.71^(+1.14)_(-1.19) km
(16th–84th posterior quantiles). The manuscript's number is consistent with rounding that older result, using only
its upper error and symmetrising. This is an inference about provenance; no original fitting-session record proves
which rounding was used. **It is not a measurement of the radius at exactly 1.4 M_sun**, and not the new Vinciguerra
2024 joint NICER+XMM headline.
