'''Tests. Run with `python -m tests.run_all`.

`app.config` creates its data directories at import time, so they are pointed at a
throwaway location here, before any app module is imported. Tests never call a real
model; the key only has to exist for the OpenAI client to be constructed.
'''

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="repuragent-tests-"))
for _name in ("PERSISTENCE_ROOT", "DATA_ROOT", "RESULTS_ROOT", "MEMORY_ROOT"):
    os.environ[_name] = str(_TMP / _name.lower())
os.environ.setdefault("OPENAI_API_KEY", "dummy")
