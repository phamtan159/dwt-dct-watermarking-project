"""
Compatibility wrapper for the speaker-aware compare step.

The active implementation is tools/06_compare_transcript_phonemes.py.
This file is kept so older command notes that call step 05 still run the
speaker-aware code path instead of writing flat annotation outputs.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("06_compare_transcript_phonemes.py")
    runpy.run_path(str(target), run_name="__main__")
