from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest

from narrative_dynamics.attestation import measure_implementation


class FunctionImplementationMeasurementTests(unittest.TestCase):
    def test_python_function_measurement_binds_loaded_module_bytes_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "function_attestation_fixture.py"
            module_path.write_text(
                "def kernel():\n    raise AssertionError('must not execute')\n",
                encoding="utf-8",
            )
            sys.path.insert(0, directory)
            try:
                module = importlib.import_module("function_attestation_fixture")
                first = measure_implementation(module.kernel)
                module_path.write_text(
                    "def kernel():\n    changed = True\n    raise AssertionError('must not execute')\n",
                    encoding="utf-8",
                )
                second = measure_implementation(module.kernel)
            finally:
                sys.path.remove(directory)
                sys.modules.pop("function_attestation_fixture", None)

        self.assertEqual(
            first.target,
            "python-callable:function_attestation_fixture.kernel",
        )
        self.assertEqual(
            tuple(item.locator for item in first.artifacts),
            ("python-module:function_attestation_fixture",),
        )
        self.assertNotEqual(first.content_hash, second.content_hash)


if __name__ == "__main__":
    unittest.main()
