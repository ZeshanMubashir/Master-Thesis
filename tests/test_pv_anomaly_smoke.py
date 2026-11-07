"""Smoke tests for the pv_anomaly package imports."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestPvAnomalyImports(unittest.TestCase):
    """Validate that all required modules import cleanly."""

    def test_import_all_modules(self) -> None:
        """Import every required module in src/pv_anomaly."""
        module_names = [
            "pv_anomaly.data",
            "pv_anomaly.eval",
            "pv_anomaly.models",
            "pv_anomaly.train",
            "pv_anomaly.utils",
        ]

        try:
            for module_name in module_names:
                importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "timm", "torchvision", "numpy"}:
                self.skipTest(f"Optional runtime dependency missing in test environment: {exc.name}")
            raise


if __name__ == "__main__":
    unittest.main()
