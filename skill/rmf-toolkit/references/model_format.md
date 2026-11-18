# RMF model definition format

The loader in `scripts/agentV1-tools/model_io.py` reads a UTF-8 JSON object and
`build_model_from_definition()` creates the symbolic model, energy-density and pressure
functions, and beta-equilibrium compiler.

## Fields

| Key | Required | Meaning |
|---|---:|---|
| `model_name` | no | Display and output name. Defaults to `unnamed_model`. |
| `description` | no | Human-readable model description. |
| `param_keys` | yes | Parameter symbols passed to the analytic model, conventionally written `P['name']`. |
| `field_names` | yes | Mean-field symbols in field-vector order, e.g. `["s", "o", "r"]` for σ, ω, ρ. |
| `extra_symbol_names` | no | Bilinear symbols used by the Lagrangian. Defaults to the five standard nucleon bilinears. |
| `lagrangian_terms` | yes | List of SymPy-compatible expression strings, one term per entry. |
| `params` | yes | Numeric values keyed by bare parameter name, without `P[...]`. |
| `initial_field_guess` | no | Numeric initial values, one per field; defaults to 20 for each field. |
| `rho_beta_min` | no | First beta-EOS baryon density in fm⁻³; defaults to 0.08. |
| `rho_beta_max` | no | Last beta-EOS baryon density in fm⁻³; defaults to 1.2. |
| `rho_beta_step` | no | Uniform beta-EOS density increment in fm⁻³; defaults to 0.001. |

`param_keys` entries may use `P['name']` or `P["name"]`. A parameter present in
`params` and referenced by a Lagrangian term is added automatically if omitted from
`param_keys`.

## Expression symbols

Lagrangian terms are evaluated by the released analytic toolkit with `P` (the parameter
mapping), `sp` (SymPy), the configured mean fields, and the symbols listed in
`extra_symbol_names`. The common nucleon bilinears are:

- `psibar_i_dslash_psi`
- `psibar_psi`
- `psibar_gamma0_psi`
- `psibar_gamma0_tau3_psi`
- `psibar_tau3_psi`

Expression strings are executable model input. Load definitions only from sources you
trust; do not treat arbitrary JSON as inert data.

## Example

```json
{
  "model_name": "example",
  "param_keys": ["P['m_N']", "P['m_sigma']", "P['g_sigma']"],
  "field_names": ["s"],
  "extra_symbol_names": ["psibar_i_dslash_psi", "psibar_psi"],
  "lagrangian_terms": [
    "psibar_i_dslash_psi - P['m_N'] * psibar_psi",
    "-sp.Rational(1, 2) * P['m_sigma']**2 * s**2",
    "-P['g_sigma'] * s * psibar_psi"
  ],
  "params": {"m_N": 939.0, "m_sigma": 500.0, "g_sigma": -10.0},
  "initial_field_guess": [20.0],
  "rho_beta_min": 0.08,
  "rho_beta_max": 1.2,
  "rho_beta_step": 0.001
}
```

Load and build in one call after importing from the toolkit:

```python
from model_io import build_model_from_definition, load_model_definition

definition = load_model_definition("skill/rmf-toolkit/models/tm1_paper.json")
model = build_model_from_definition(definition)
```

The resulting object exposes `analytic`, `params`, `compiled`, `epsilon_func`,
`pressure_func`, `initial_field_guess`, and `rho_grid`.
