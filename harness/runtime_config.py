import os
from pathlib import Path

_DEFAULT_OPENCLAW_HOME = Path.home() / ".openclaw"
OPENCLAW_HOME = Path(
    os.environ.get("OPENCLAW_HOME") or _DEFAULT_OPENCLAW_HOME
).expanduser()
OPENCLAW_WORKSPACE_DIR = _DEFAULT_OPENCLAW_HOME.name
