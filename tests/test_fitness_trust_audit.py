"""Exercise the CI trust checker on source and axiom-report counterexamples."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


AUDIT = Path(__file__).resolve().parents[1] / "tools" / "audit_fitness_trust.py"


class FitnessTrustAuditTests(unittest.TestCase):
    def run_audit(self, mode, content, *extra):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("Fixture.lean" if mode == "source" else "lean.log")
            path.write_text(content, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(AUDIT), mode, str(path), *extra],
                capture_output=True, text=True, check=False,
            )

    def assert_rejected(self, mode, content, diagnostic, *extra):
        result = self.run_audit(mode, content, *extra)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(diagnostic, result.stderr)

    def test_accepts_kernel_proof_and_bounded_resource_settings(self):
        result = self.run_audit("source", "set_option maxHeartbeats 200000 in\n"
                                "example : True := by trivial\n")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_proof_oracles_and_modified_declarations(self):
        for source, diagnostic in [
            ("example : True := by sorry", "sorry"),
            ("example : True := by admit", "admit"),
            ("example : True := by native_decide", "native_decide"),
            ("unlock_limits", "unlock_limits"),
            ("private axiom hidden : False", "axiom"),
            ("@[simp] protected axiom hidden : False", "axiom"),
            ("private unsafe def unchecked : Nat := 0", "unsafe"),
            ("private /- nested /- comment -/ -/\naxiom hidden : False", "axiom"),
        ]:
            with self.subTest(source=source):
                self.assert_rejected("source", source, diagnostic)

    def test_rejects_zero_resource_limits_across_comments_and_newlines(self):
        for source in [
            "set_option maxHeartbeats 0 in\nexample : True := by trivial",
            "set_option maxRecDepth\n0",
            "set_option maxHeartbeats /- budget -/ 0",
            "set_option synthInstance.maxHeartbeats 0",
            "set_option maxHeartbeats 0x0",
            "set_option maxHeartbeats 0b0",
            "set_option maxRecDepth 0o0",
            "set_option «maxHeartbeats» 0",
            "set_option «synthInstance».«maxHeartbeats» 0",
            "set_option maxHeartbeats 0_0",
        ]:
            with self.subTest(source=source):
                self.assert_rejected("source", source, "unlimited")

    def test_comment_and_string_examples_are_not_declarations(self):
        source = '''/- private axiom example : False
          /- set_option maxHeartbeats 0 -/
          unsafe native_decide -/
-- sorry admit
def explanation := "private axiom, set_option maxHeartbeats 0, \\\"sorry\\\""
def safe_name := 1
example : safe_name = 1 := by rfl
'''
        result = self.run_audit("source", source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_string_terminator_does_not_hide_following_declaration(self):
        self.assert_rejected("source", 'def s := "hello"\nprivate axiom hidden : False', "axiom")

    def test_interpolated_string_terms_cannot_hide_proof_escape(self):
        for prefix in ['s!', 's! ', 's! /- comment -/ ', 'dbg_trace ']:
            with self.subTest(prefix=prefix):
                self.assert_rejected("source", 'def s := ' + prefix + '"{show Nat from by sorry}"', "sorry")

    def test_escaped_identifier_quotes_cannot_hide_a_declaration(self):
        self.assert_rejected("source", 'def «first"» := 1\nprivate axiom hidden : False\n'
                             'def «second"» := 2\n', "axiom")

    def test_builtin_interpolation_without_bang_cannot_hide_proof_escape(self):
        for prefix in ['throwError ', 'throwErrorAt ref ',
                       'throwNamedError errorName ', 'logNamedError errorName ']:
            with self.subTest(prefix=prefix):
                self.assert_rejected("source", 'def f := ' + prefix + '"{show Nat from by sorry}"', "sorry")

    def test_accepts_raw_string_without_hashes(self):
        result = self.run_audit("source", 'def s := r"\\"\n')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_raw_strings_and_quote_characters_do_not_hide_following_code(self):
        result = self.run_audit("source", 'def s := r#"a " /- sorry "#\n'
                                'def quote : Char := \'"\'\nexample : True := by trivial\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_rejected("source", 'def quote : Char := \'"\'\nprivate axiom hidden : False', "axiom")

    def test_identifier_suffix_does_not_trigger_keyword_detection(self):
        result = self.run_audit("source", "def sorry_count := 0\ndef unsafeName := 1\n")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_multiline_allowed_axioms_and_axiom_free_report(self):
        result = self.run_audit("log", "'Test.normalized' depends on axioms: [propext,\n"
                                " Classical.choice, Quot.sound]\n"
                                "'Test.reflexive' does not depend on any axioms\n",
                                "--require", "Test.normalized", "--require", "Test.reflexive")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_accepts_timestamped_ansi_ci_report(self):
        result = self.run_audit("log", "2026-09-14T00:00:00.0Z \x1b[32m'Test.safe' depends on axioms: [propext,\n"
                                "2026-09-14T00:00:00.1Z  Quot.sound]\x1b[0m\n",
                                "--require", "Test.safe")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_transitive_user_axiom_and_native_oracle(self):
        for dependency in ["sorryAx", "Test.hidden", "Lean.ofReduceBool"]:
            with self.subTest(dependency=dependency):
                self.assert_rejected("log", "'Test.safe' depends on axioms: [propext,\n"
                                     + dependency + "]\n", dependency)

    def test_prime_suffixed_theorem_cannot_hide_an_extra_axiom(self):
        self.assert_rejected("log", "'Test.safe' depends on axioms: [propext]\n"
                             "'Test.helper'' depends on axioms: [Test.hidden]\n", "Test.hidden")

    def test_rejects_missing_requested_theorem_report(self):
        self.assert_rejected("log", "'Test.other' depends on axioms: [propext]\n",
                             "Test.required", "--require", "Test.required")

    def test_rejects_empty_and_truncated_reports(self):
        for content in ["", "Build completed successfully.\n",
                        "'Test.safe' depends on axioms: [propext,\n"]:
            with self.subTest(content=content):
                self.assert_rejected("log", content, "report")

    def test_rejects_missing_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(AUDIT), "source", str(Path(directory) / "missing.lean")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("missing.lean", result.stderr)


if __name__ == "__main__":
    unittest.main()
