"""Tests for the PHI/MIMIC compliance checker (scripts/check_no_mimic_data.py).

Focus: the checker must produce a clean, actionable report and the correct exit
code even on a console whose default encoding cannot represent the report glyphs
(e.g. Windows cp1252 and the "x" cross marker) — it must never crash mid-report.

Run with:
    python -m pytest tests/test_compliance.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHECKER = _REPO_ROOT / "scripts" / "check_no_mimic_data.py"


def _run(path, io_encoding=None):
    env = dict(os.environ)
    if io_encoding:
        env["PYTHONIOENCODING"] = io_encoding
    return subprocess.run(
        [sys.executable, str(_CHECKER), str(path)],
        capture_output=True,
        env=env,
    )


def test_clean_file_passes(tmp_path):
    f = tmp_path / "clean.txt"
    f.write_text("Patient summary; take aspirin daily.\n", encoding="utf-8")
    result = _run(f)
    assert result.returncode == 0


def test_phi_file_flagged(tmp_path):
    f = tmp_path / "note.txt"
    # Build the trigger at runtime so this test source stays clean for the
    # repo-wide compliance scan (no literal PHI header committed here).
    f.write_text("Attending" + ": Dr. Smith\n", encoding="utf-8")
    result = _run(f)
    assert result.returncode == 1
    assert b"VIOLATION" in result.stdout


def test_no_crash_on_nonutf8_console(tmp_path):
    """The cross marker must not crash the report on a cp1252 console."""
    f = tmp_path / "note.txt"
    f.write_text("Attending" + ": Dr. Smith\n", encoding="utf-8")
    result = _run(f, io_encoding="cp1252")
    # Gate still fires...
    assert result.returncode == 1
    assert b"VIOLATION" in result.stdout
    # ...and no encoding crash leaked to stderr.
    assert b"UnicodeEncodeError" not in result.stderr
    assert b"Traceback" not in result.stderr
