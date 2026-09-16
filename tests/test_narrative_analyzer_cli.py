import io
import tempfile
import unittest
from pathlib import Path

from narrative_analyzer.cli import main, render_result
from narrative_analyzer.result import (
    AnalysisResult,
    CertificateCompileError,
    ClaimResult,
    ClaimStatus,
    ProvenanceMismatchError,
)


def claim(cid, status, theorem=None, note=None, exact=None):
    return ClaimResult(
        claim_id=cid,
        status=status,
        theorem=theorem,
        assumptions=("assumption-a",) if theorem else (),
        exact_values=exact or {},
        note=note,
    )


def result_with(*claims):
    return AnalysisResult(
        model_summary={"topology": "path", "n": 7, "exposure_semantics": "post-incoming lookup"},
        claims=claims,
    )


class FakeRunner:
    pass


class CliRenderingTests(unittest.TestCase):
    def test_deterministic_claim_order_and_provenance(self):
        result = result_with(
            claim("parameters_valid", ClaimStatus.PROVED, "Lean.valid"),
            claim("pathn_consensus_exists", ClaimStatus.PROVED, "Lean.consensus", exact={"eps": "1/4"}),
            claim("consensus_value_known", ClaimStatus.UNKNOWN, note="no closed form"),
        )
        text = render_result(result)
        self.assertLess(text.index("parameters_valid"), text.index("pathn_consensus_exists"))
        self.assertIn("Lean.valid", text)
        self.assertIn("Lean.consensus", text)
        self.assertIn("eps=1/4", text)
        self.assertIn("Consensus: PROVED", text)
        self.assertIn("Consensus value: unknown in closed form", text)

    def test_disproved_and_unknown_are_valid_rendered_results(self):
        result = result_with(
            claim("pathn_consensus_exists", ClaimStatus.UNKNOWN, note="no theorem route"),
            claim("consensus_value_known", ClaimStatus.UNKNOWN, note="no theorem route"),
        )
        text = render_result(result)
        self.assertIn("UNKNOWN", text)
        self.assertIn("no theorem route", text)


class CliMainTests(unittest.TestCase):
    def _model_file(self):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "model.toml"
        path.write_text('''
[topology]
kind = "path"
n = 2
[initial]
beliefs = ["1", "0"]
exposures = [0, 0]
[dynamics]
threshold = "0"
[schedule]
kind = "constant"
value = "1/4"
''', encoding="utf-8")
        return temp, path

    def test_usage_requires_exactly_one_model_path(self):
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(main([], stdout=out, stderr=err), 2)
        self.assertIn("usage:", err.getvalue().lower())
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(main(["a", "b"], stdout=out, stderr=err), 2)

    def test_valid_disproved_or_unknown_analysis_exits_zero(self):
        temp, path = self._model_file()
        self.addCleanup(temp.cleanup)
        fake_result = result_with(
            claim("path2_consensus", ClaimStatus.DISPROVED, "Lean.negative"),
            claim("consensus_value_known", ClaimStatus.UNKNOWN, note="no value theorem"),
        )
        out, err = io.StringIO(), io.StringIO()
        code = main(
            [str(path)],
            stdout=out,
            stderr=err,
            runner_factory=lambda root: FakeRunner(),
            analyze=lambda model, runner: fake_result,
        )
        self.assertEqual(code, 0)
        self.assertIn("DISPROVED", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_invalid_input_exits_two_without_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.toml"
            path.write_text('[topology]\nkind="cycle"\nn=2\n', encoding='utf-8')
            out, err = io.StringIO(), io.StringIO()
            code = main([str(path)], stdout=out, stderr=err)
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err.getvalue())
        self.assertIn("input error", err.getvalue().lower())

    def test_certificate_failure_exits_three_not_unknown(self):
        temp, path = self._model_file(); self.addCleanup(temp.cleanup)
        def fail(model, runner):
            raise CertificateCompileError("proof failed")
        out, err = io.StringIO(), io.StringIO()
        code = main([str(path)], stdout=out, stderr=err,
                    runner_factory=lambda root: FakeRunner(), analyze=fail)
        self.assertEqual(code, 3)
        self.assertIn("certificate error", err.getvalue().lower())
        self.assertNotIn("UNKNOWN", err.getvalue())

    def test_provenance_failure_exits_four(self):
        temp, path = self._model_file(); self.addCleanup(temp.cleanup)
        def fail(model, runner):
            raise ProvenanceMismatchError("digest mismatch")
        out, err = io.StringIO(), io.StringIO()
        code = main([str(path)], stdout=out, stderr=err,
                    runner_factory=lambda root: FakeRunner(), analyze=fail)
        self.assertEqual(code, 4)
        self.assertIn("internal analyzer error", err.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
