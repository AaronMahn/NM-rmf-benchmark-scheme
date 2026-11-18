"""Path discovery for the referee-validation package (no sibling study folders).

Layout assumed: <release>/paper/referee_validation/ next to <release>/skill/rmf-toolkit/ and
<release>/paper/sigma6_tm1/. Override any location with an environment variable:
  NNSTAR_TOOLS_DIR   .../skill/rmf-toolkit/scripts/agentV1-tools  (TOVtools/, NMtoolsV1/, model_io.py)
  NNSTAR_SIGMA6_DIR  .../paper/sigma6_tm1  (model.json for backend validation and plotting)
  NNSTAR_VALIDATION_WORK  scratch/output directory (downloads, samples, chains, regenerated results);
                          default <this folder>/work.nosync
"""
import os
from pathlib import Path
PKG = Path(__file__).resolve().parent
RELEASE = PKG.parents[1]
TOOLS = Path(os.environ.get('NNSTAR_TOOLS_DIR', RELEASE / 'skill/rmf-toolkit/scripts/agentV1-tools'))
SIGMA6 = Path(os.environ.get('NNSTAR_SIGMA6_DIR', RELEASE / 'paper/sigma6_tm1'))
WORK = Path(os.environ.get('NNSTAR_VALIDATION_WORK', PKG / 'work.nosync'))
REF = PKG / 'reference'
RESULTS = WORK / 'results'
BPS = TOOLS / 'TOVtools/bps.dat'


def check_toolkit():
    for p in (TOOLS / 'TOVtools', TOOLS / 'NMtoolsV1', BPS):
        if not p.exists():
            raise FileNotFoundError(f'{p} not found; set NNSTAR_TOOLS_DIR to the released agentV1-tools directory')


def result_dir(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def reference_or_result(name, fname):
    """Freshly regenerated file if present, else the checked reference copy."""
    for p in (RESULTS / name / fname, REF / name / fname):
        if p.exists():
            return p
    raise FileNotFoundError(f'{name}/{fname}')
