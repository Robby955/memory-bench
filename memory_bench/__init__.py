"""memory-bench: Controlled comparison of memory mechanisms for small-scale LLMs."""

import sys
import os

# Add nanochat to path so we can import from it
_nanochat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nanochat")
if _nanochat_path not in sys.path:
    sys.path.insert(0, _nanochat_path)
