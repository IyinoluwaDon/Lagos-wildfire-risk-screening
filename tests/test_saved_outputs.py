"""Saved-artifact consistency and dashboard rendering checks for CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_saved_outputs_verify_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/verify_outputs.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"checks": "passed"' in result.stdout


def test_streamlit_dashboard_renders_saved_results():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    assert not app.exception, [exception.message for exception in app.exception]
    assert len(app.tabs) == 4
    assert len(app.metric) >= 10
    assert any("Lagos Wildfire Risk Screening" in title.value for title in app.title)
