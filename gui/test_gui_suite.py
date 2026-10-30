"""The GUI tests as unittest cases, so the checkout's `testing focused` runs them under its host test slot.

`gui/tests` are pytest modules in the GUI's own uv environment (NiceGUI's `User` plugin), which the checkout's
interpreter cannot import. Each case here runs one of them with pytest in `gui/` and fails with its output:

    uv run --locked --offline python -m bio_neb.devtools.testing focused gui/test_gui_suite.py
    uv run --locked --offline python -m bio_neb.devtools.testing focused gui.test_gui_suite.GuiTests.test_results

Standard library only: this file is imported by the checkout's interpreter, never by the GUI's.
"""

import subprocess
import unittest
from pathlib import Path

GUI = Path(__file__).resolve().parent
PYTEST = ["uv", "run", "--locked", "--offline", "pytest", "-q", "-p", "no:cacheprovider"]


class GuiTests(unittest.TestCase):
    def run_module(self, module: Path) -> None:
        process = subprocess.run([*PYTEST, str(module.relative_to(GUI))], cwd=GUI, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            self.fail(f"pytest {module.name} exit {process.returncode}\n{process.stdout}\n{process.stderr}")


# One case per GUI test module (`test_results.py` -> `test_results`), so a new module is picked up by itself.
for _module in sorted((GUI / "tests").glob("test_*.py")):
    setattr(GuiTests, _module.stem, lambda self, module=_module: self.run_module(module))
