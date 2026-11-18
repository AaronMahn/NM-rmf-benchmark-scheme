"""
paper_extractor.py
==================
Extract RMF model definitions and validation targets from physics papers.

Supports:
  - Local PDF files (pdfplumber-based text extraction)
  - arXiv IDs (HTML from ar5iv.labs.arxiv.org, or PDF fallback)

Core function
-------------
    result = extract_model_from_paper(
        pdf_path="mymodel.pdf",  # or arxiv_id="2303.12345"
    )
    # result.model_definition   — ready for model_io.build_model_from_definition()
    # result.paper_reported     — {rho0, BE, K, Esym, L, Mmax, R14, ...} from paper
    # result.raw_extracted      — full LLM response for debugging
    # result.source_text_chars  — number of chars fed to the LLM

The LLM used for extraction is claude-haiku-4-5-20251001 by default (fast, cheap).
Override with the `extraction_model` argument.
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text_pdfplumber(pdf_bytes: bytes) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def _extract_pdf_text_pypdf(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes. Tries pdfplumber first, then pypdf."""
    try:
        return _extract_pdf_text_pdfplumber(pdf_bytes)
    except Exception as pdfplumber_error:
        pdfplumber_message = str(pdfplumber_error)
    try:
        return _extract_pdf_text_pypdf(pdf_bytes)
    except Exception as pypdf_error:
        raise RuntimeError(
            "PDF text extraction failed with pdfplumber "
            f"({pdfplumber_message}) and pypdf ({pypdf_error})"
        ) from pypdf_error


def read_pdf_file(path: str | Path) -> str:
    """Read a local PDF and return its text content."""
    data = Path(path).read_bytes()
    return extract_pdf_text(data)


# ---------------------------------------------------------------------------
# arXiv fetching
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 rmf-agent/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _arxiv_id_clean(arxiv_id: str) -> str:
    """Normalize an arXiv ID: strip leading 'arxiv:' prefix and version suffix."""
    aid = arxiv_id.strip().lower()
    if aid.startswith('arxiv:'):
        aid = aid[6:]
    # Strip version like v1, v2
    aid = re.sub(r'v\d+$', '', aid)
    return aid.strip()


def fetch_arxiv_text(arxiv_id: str) -> str:
    """
    Fetch paper text for an arXiv ID.
    Primary: ar5iv HTML (much better text quality than PDF).
    Fallback: arxiv PDF download.
    """
    aid = _arxiv_id_clean(arxiv_id)

    # Try ar5iv HTML first
    html_error: Exception | None = None
    try:
        html_url = f"https://ar5iv.labs.arxiv.org/html/{aid}"
        html_bytes = _fetch_url(html_url)
        text = _parse_html_text(html_bytes.decode('utf-8', errors='replace'))
        if len(text.strip()) > 500:
            return text
    except Exception as exc:
        html_error = exc

    # Fallback: PDF from arxiv
    pdf_url = f"https://arxiv.org/pdf/{aid}"
    try:
        pdf_bytes = _fetch_url(pdf_url)
        return extract_pdf_text(pdf_bytes)
    except Exception as pdf_error:
        detail = f"ar5iv HTML failed ({html_error}); " if html_error else ""
        raise RuntimeError(
            f"Could not fetch arXiv paper {aid}: {detail}arXiv PDF failed ({pdf_error})"
        ) from pdf_error


def _parse_html_text(html: str) -> str:
    """Very lightweight HTML → plain text (no external deps required)."""
    # Remove scripts, styles, LaTeX macros blocks
    for tag in ('script', 'style', 'head', 'math'):
        html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Decode common HTML entities
    for ent, ch in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                    ('&nbsp;', ' '), ('&minus;', '-'), ('&#8722;', '-')]:
        text = text.replace(ent, ch)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are an expert in Relativistic Mean Field (RMF) nuclear physics. \
Your job is to read sections of a physics paper and extract the RMF model \
definition and reported results in a precise JSON format.

Key conventions used by our code:
- Field names: "s" = sigma (σ), "o" = omega (ω), "r" = rho (ρ), "a" = a0/delta (a₀/δ)
- Parameter keys: P['m_N'], P['m_sigma'], P['m_omega'], P['m_rho'], etc.
- Lagrangian terms are Python/SymPy strings (use sp.Rational, sp.Integer, ** for powers)
- Coupling sign convention (CRITICAL — read carefully):
  Our Lagrangian terms are written as:
      -P['g_sigma'] * s * psibar_psi           ← scalar (attractive)
      -P['g_omega'] * o * psibar_gamma0_psi    ← vector (repulsive)
      -P['g_rho']   * r * psibar_gamma0_tau3_psi
      -P['g_a0']    * a * psibar_tau3_psi
  The effective nucleon mass is: M* = m_N + g_sigma * sigma
  For M* to decrease with density (physical), g_sigma MUST BE NEGATIVE.
  Convention mapping:
    - If paper lists g_sigma = +10.1 (positive), set "g_sigma": -10.1 in params
    - If paper lists g_sigma = -10.1 (already negative), set "g_sigma": -10.1
    - g_omega, g_rho, g_a0 are POSITIVE in both paper and our params
  Always double-check: the scalar coupling should make M* < m_N at finite density.
- Meson self-interaction coupling names:
    g2, g3  = cubic and quartic sigma self-couplings (κ/B, λ/C in many papers)
    c3, c4  = quartic omega self-coupling, omega^2*rho^2 cross-coupling (ξ, Λ_v)
    d3      = quartic rho self-coupling
    g4,g5,g6,g7 = sigma^2*omega^2, sigma^2*rho^2, sigma*rho^2, sigma*omega^2 cross-couplings
- Units: masses in MeV, densities in fm^-3, energies in MeV
"""

_EXTRACTION_USER_TEMPLATE = """\
Below is the text of a nuclear physics paper. Extract the RMF model and return ONLY a \
JSON object with exactly this structure (no markdown, no prose — pure JSON):

{{
  "model_name": "<string: short identifier for this model, e.g. 'NL3', 'FSUGold', 'TM1'>",
  "description": "<string: one-line description from paper title/abstract>",
  "param_keys": [...],
  "field_names": [...],
  "extra_symbol_names": [
    "psibar_i_dslash_psi", "psibar_psi",
    "psibar_gamma0_psi", "psibar_gamma0_tau3_psi", "psibar_tau3_psi"
  ],
  "lagrangian_terms": [...],
  "params": {{
    "m_N": <float>,
    "m_sigma": <float>,
    "m_omega": <float>,
    "m_rho": <float>,
    <other parameters...>
  }},
  "initial_field_guess": [20.0, 15.0, 0.0],
  "rho_beta_min": 0.08,
  "rho_beta_max": 1.2,
  "rho_beta_step": 0.001,

  "paper_reported": {{
    "rho0_fm3":      <float or null>,
    "BE_rho0_MeV":   <float or null>,
    "K_rho0_MeV":    <float or null>,
    "Esym_rho0_MeV": <float or null>,
    "L_rho0_MeV":    <float or null>,
    "Mmax_Msun":     <float or null>,
    "R14_km":        <float or null>,
    "R13_km":        <float or null>
  }},

  "extraction_notes": "<string: any ambiguities, assumptions, or missing values>"
}}

Rules:
1. `lagrangian_terms` must be valid Python/SymPy expressions. The standard free-nucleon kinetic term is always:
   "sp.Integer(1) * psibar_i_dslash_psi - P['m_N'] * psibar_psi"
2. Include ALL meson self-interaction and cross-coupling terms present in the paper.
3. For parameters not found in the paper, omit them from `params` (do not guess).
4. For `paper_reported`, use null for values not given in the paper.
5. If the paper uses notation like κ for the cubic sigma coupling (g2), convert accordingly.
6. `field_names` list order determines field indexing; put sigma first, then omega, rho, a0 if present.
7. If the paper only lists g_sigma^2 / m_sigma^2 (dimensionless ratios), convert to individual values
   using the given masses.

Paper text:
---
{paper_text}
---
"""


def build_extraction_prompt(paper_text: str, max_chars: int = 24000) -> str:
    """Trim paper text to fit context and inject into the prompt template."""
    # Prefer the first ~2/3 of the paper (abstract, intro, model section, tables)
    # and last 1/3 (results tables) — skip the middle bibliography
    if len(paper_text) > max_chars:
        head = paper_text[:int(max_chars * 0.7)]
        tail = paper_text[-int(max_chars * 0.3):]
        trimmed = head + "\n[... middle section omitted for brevity ...]\n" + tail
    else:
        trimmed = paper_text
    return _EXTRACTION_USER_TEMPLATE.format(paper_text=trimmed)


# ---------------------------------------------------------------------------
# LLM API call
# ---------------------------------------------------------------------------

def _call_claude_api(
    system: str,
    user: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 4096,
) -> str:
    """Call Anthropic Claude API and return the text response."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("Set ANTHROPIC_API_KEY in the environment to use paper extraction.")
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not found. Install the optional requirements-extract.txt dependencies."
        ) from None
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _parse_json_from_response(text: str) -> dict:
    """Extract and parse the JSON object from an LLM response."""
    # Strip markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Find the outermost JSON object
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    # Balance braces
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        raise ValueError("Unbalanced JSON object in LLM response")

    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaperExtractionResult:
    model_definition: dict          # ready for model_io.build_model_from_definition()
    paper_reported: dict            # NMP + NS observables from paper
    extraction_notes: str           # LLM's own notes on ambiguities
    raw_response: str               # full LLM response (for debugging)
    source: str                     # 'pdf:<path>' or 'arxiv:<id>'
    source_text_chars: int          # chars of paper text fed to LLM


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_model_from_paper(
    *,
    pdf_path: str | Path | None = None,
    arxiv_id: str | None = None,
    extraction_model: str = "claude-haiku-4-5-20251001",
    max_paper_chars: int = 24000,
    verbose: bool = True,
) -> PaperExtractionResult:
    """
    Extract an RMF model definition and validation targets from a paper.

    Parameters
    ----------
    pdf_path : path to local PDF (mutually exclusive with arxiv_id)
    arxiv_id : arXiv identifier like '2303.12345' (mutually exclusive with pdf_path)
    extraction_model : Claude model to use for extraction
    max_paper_chars  : max paper text characters fed to LLM
    verbose  : print progress messages

    Returns
    -------
    PaperExtractionResult
    """
    if pdf_path is None and arxiv_id is None:
        raise ValueError("Provide either pdf_path or arxiv_id")
    if pdf_path is not None and arxiv_id is not None:
        raise ValueError("Provide either pdf_path or arxiv_id, not both")

    # --- Step 1: Get paper text ---
    if pdf_path is not None:
        source = f"pdf:{pdf_path}"
        if verbose:
            print(f"Reading PDF: {pdf_path}", flush=True)
        paper_text = read_pdf_file(pdf_path)
    else:
        source = f"arxiv:{arxiv_id}"
        if verbose:
            print(f"Fetching arXiv paper: {arxiv_id}", flush=True)
        paper_text = fetch_arxiv_text(arxiv_id)

    if verbose:
        print(f"  Paper text: {len(paper_text):,} chars", flush=True)

    # --- Step 2: Build prompt and call LLM ---
    prompt = build_extraction_prompt(paper_text, max_chars=max_paper_chars)
    n_chars = len(prompt)
    if verbose:
        print(f"  Prompt fed to LLM: ~{n_chars:,} chars", flush=True)
        print(f"  Calling {extraction_model} for extraction …", flush=True)

    raw_response = _call_claude_api(
        system=_EXTRACTION_SYSTEM,
        user=prompt,
        model=extraction_model,
    )

    if verbose:
        print("  Extraction complete.", flush=True)

    # --- Step 3: Parse response ---
    extracted = _parse_json_from_response(raw_response)

    # Split into model_definition and paper_reported
    paper_reported = extracted.pop("paper_reported", {})
    extraction_notes = extracted.pop("extraction_notes", "")

    # Ensure required fields are present with defaults
    extracted.setdefault("extra_symbol_names", [
        "psibar_i_dslash_psi", "psibar_psi",
        "psibar_gamma0_psi", "psibar_gamma0_tau3_psi", "psibar_tau3_psi",
    ])
    extracted.setdefault("rho_beta_min", 0.08)
    extracted.setdefault("rho_beta_max", 1.2)
    extracted.setdefault("rho_beta_step", 0.001)
    n_fields = len(extracted.get("field_names", []))
    extracted.setdefault("initial_field_guess", [20.0, 15.0, 0.0] + [0.0] * max(0, n_fields - 3))

    return PaperExtractionResult(
        model_definition=extracted,
        paper_reported=paper_reported,
        extraction_notes=extraction_notes,
        raw_response=raw_response,
        source=source,
        source_text_chars=len(paper_text),
    )
