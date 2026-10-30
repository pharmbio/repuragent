from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# `app.config` creates its data directories at import time, so they are pointed at a
# throwaway location before any app module is imported. Tests never call a real model;
# the key only has to exist for the OpenAI client to be constructed.
_TMP = Path(tempfile.mkdtemp(prefix="repuragent-tests-"))
for _name in ("PERSISTENCE_ROOT", "DATA_ROOT", "RESULTS_ROOT", "MEMORY_ROOT"):
    os.environ[_name] = str(_TMP / _name.lower())
os.environ.setdefault("OPENAI_API_KEY", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
