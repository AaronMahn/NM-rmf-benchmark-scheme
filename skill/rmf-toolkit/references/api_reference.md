# RMF toolkit API reference

All commands below are run from the repository root. Python dependencies are listed in
`requirements.txt`; the paper extractor has additional optional dependencies in
`requirements-extract.txt`.

## Build a model from JSON

```python
from model_io import build_model_from_definition, load_model_definition

definition = load_model_definition("skill/rmf-toolkit/models/tm1_paper.json")
model = build_model_from_definition(definition)
```

`model` provides `analytic`, `params`, `compiled`, `epsilon_func`, `pressure_func`,
`initial_field_guess`, and `rho_grid`. See `model_format.md` for the JSON schema and the
warning that Lagrangian expression strings are executable input.

## Run the standard pipeline

```sh
python skill/rmf-toolkit/scripts/run_model.py \
  --model skill/rmf-toolkit/models/tm1_paper.json \
  --out results/tm1
```

The output directory contains nuclear-property, beta-EOS, hybrid-EOS, M-R curve, and summary
JSON files. `--mr-samples` controls the number of central-density samples.

## Score a constraint set

```sh
python skill/rmf-toolkit/scripts/bja_score.py --list-constraints
python skill/rmf-toolkit/scripts/bja_score.py \
  --model skill/rmf-toolkit/models/tm1_paper.json \
  --constraints standard_nm --out results/tm1_score.json
```

Each catalogue entry defines a quantity, central value, uncertainty, type, unit, and source
reference. `one_sided_lower_bound` contributes zero penalty above the bound and a squared
Gaussian penalty below it. The Mmax >= 2.08 Msun penalty uses sigma 0.07 Msun. Entries of
type `parameter_prior` score a model parameter; the `m_sigma` entry is 600 +/- 200 MeV and
is not an observable. `joint_mr_likelihood(observables, constraint)` is the reserved
extension point for a future non-Gaussian joint M-R likelihood; it intentionally raises
`NotImplementedError` until that work is specified.

## Fit selected parameters

```sh
printf '%s\n' '{"g_sigma":[-12,-8],"g_omega":[10,15]}' > /tmp/rmf-fit-bounds.json
python skill/rmf-toolkit/scripts/fit_model.py \
  --model skill/rmf-toolkit/models/tm1_paper.json \
  --constraints standard_nm \
  --free-params g_sigma,g_omega \
  --bounds /tmp/rmf-fit-bounds.json \
  --out results/tm1_fit.json
```

Only listed parameters are optimized. Bounds are mandatory (in the model definition or via
`--bounds`); fitting is separate from scoring and may be computationally expensive when a
constraint set includes neutron-star observables.

## Extract a paper model

Install the opt-in dependencies with `pip install -r requirements-extract.txt`, then set
`ANTHROPIC_API_KEY` in the environment. Never pass a key on a command line or store one in
the model definition.

```sh
python skill/rmf-toolkit/scripts/run_paper_model.py --pdf /path/to/paper.pdf --extract-only
python skill/rmf-toolkit/scripts/run_paper_model.py --arxiv 2303.12345 --mr-sample 80
```

The extractor can also read a previously reviewed definition with `--definition`. Extracted
JSON contains executable Lagrangian expressions: inspect and validate it before running the
physics pipeline. No API call occurs during tests; the extractor client is mocked.

## Ground-truth regeneration

```sh
python harness/gt/recalibrate_mr.py
python harness/gt/fix2276.py
python harness/gt/verify_star.py
```

This writes only beneath `harness/gt/`, leaves the shipped ground-truth file unchanged, and
checks that regenerated M-R values agree with it to 1e-6 relative precision. See
`harness/gt/README.md` for provenance and the limitation that the original matter-target
sampler has not yet been released.
