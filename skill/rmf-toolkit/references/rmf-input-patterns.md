# RMF Toolkit Input Patterns

## 1. What to request when a user gives a new or modified Lagrangian

Request these items before trying to run or encode the model:

1. Lagrangian in LaTeX or typed formula
2. List of active fields
3. Definitions of all couplings and self-interaction terms
4. Which terms are inherited unchanged from the baseline
5. Numerical parameter values
6. Mean-field / zero-temperature / uniform-matter assumptions
7. Whether couplings are constants or density-dependent
8. Whether the pressure/energy-density expressions or field equations follow a standard reference

If items 4-8 are missing, direct execution is risky even if the symbolic formula looks familiar.

## 2. Minimal information package for a baseline-derived variant

A practical handoff should look like this:

### A. Lagrangian

```tex
\mathcal{L} = ...
```

### B. Baseline note

- baseline model: optional
- unchanged terms: ...
- modified terms: ...
- newly added terms: ...
- removed terms: ...

### C. Assumptions

- zero temperature: yes/no
- uniform matter: yes/no
- mean-field approximation: yes/no
- beta equilibrium: yes/no
- charge neutrality: yes/no

### D. Parameters

- masses
- couplings
- self-interaction constants
- cross-coupling constants

## 3. When a simple parameter table is still useful

A compact table is useful only when all listed models share one common Lagrangian structure and differ only in numeric values.

Typical use case:

- one baseline model
- several nearby variants

with the same field content and the same active interaction terms.

In that case, a one-row-per-model table is a convenient implementation detail, not the conceptual definition of the model family.

## 4. Output recommendations

For each model instance, save at least:

- `eos.npy` or equivalent EOS table
- `summary.json` with key metrics
- optional MR outputs such as `mr.json` or `mr.npy`

Recommended summary fields:

```json
{
  "model_id": "TMV1",
  "base_model": "TM1",
  "status": "ok",
  "r14_km": 11.4,
  "r20_km": 12.1,
  "mmax_msun": 2.08,
  "notes": "TM1 plus modified rho-sector coupling"
}
```

## 5. Common mistakes

- Giving only the symbolic Lagrangian without parameter values
- Forgetting to say which terms are inherited from the baseline
- Leaving units implicit
- Treating two structurally different Lagrangians as if they were the same family
- Writing one-off code paths for one model instead of extending the shared parser
- Treating filtering MR curves and presentation MR curves as the same object without documenting the distinction
