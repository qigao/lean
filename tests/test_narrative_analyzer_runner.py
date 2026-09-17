import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from narrative_analyzer.certificate import Certificate
from narrative_analyzer.result import CertificateCompileError, CertificateTimeoutError
from narrative_analyzer.runner import CompileEvidence, LeanCertificateRunner


class _Completed:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RunnerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(tempfile.mkdtemp())
        self.paths: list[Path] = []
        self.commands: list[list[str]] = []

    def tearDown(self) -> None:
        shutil.rmtree(self.repo, ignore_errors=True)

    def _certificate(self, source='example : (1 : Nat) = 1 := by rfl\n') -> Certificate:
        return Certificate(source=source, claims=())

    def test_success_returns_digest_and_bounded_command(self) -> None:
        def fake_run(argv, *, cwd, capture_output, text, timeout, check):
            self.commands.append(list(argv))
            path = Path(argv[-1])
            self.paths.append(path)
            self.assertTrue(path.exists())
            self.assertEqual(Path(cwd), self.repo)
            self.assertTrue(capture_output)
            self.assertTrue(text)
            self.assertFalse(check)
            self.assertLessEqual(timeout, 250)
            return _Completed(0, 'lean-ok', '')

        certificate = self._certificate()
        evidence = LeanCertificateRunner(
            self.repo, run_process=fake_run
        ).compile(certificate)
        self.assertIsInstance(evidence, CompileEvidence)
        self.assertEqual(
            evidence.certificate_digest,
            hashlib.sha256(certificate.source.encode('utf-8')).hexdigest(),
        )
        self.assertEqual(evidence.returncode, 0)
        self.assertEqual(evidence.stdout, 'lean-ok')
        command = self.commands[0]
        self.assertEqual(command[:4], ['timeout', '--kill-after=10s', '240s', 'lake'])
        self.assertEqual(command[4:7], ['env', 'lean', '-DmaxErrors=1'])
        self.assertIn('-DstderrAsMessages=false', command)
        self.assertFalse(self.paths[0].exists())

    def test_nonzero_exit_raises_compile_error_and_cleans_certificate(self) -> None:
        observed = None

        def fake_run(argv, **kwargs):
            nonlocal observed
            observed = Path(argv[-1])
            self.assertTrue(observed.exists())
            return _Completed(1, 'out', 'type mismatch')

        with self.assertRaises(CertificateCompileError) as caught:
            LeanCertificateRunner(self.repo, run_process=fake_run).compile(
                self._certificate()
            )
        self.assertIn('type mismatch', str(caught.exception))
        self.assertIsNotNone(observed)
        assert observed is not None
        self.assertFalse(observed.exists())

    def test_timeout_raises_timeout_error_and_never_becomes_unknown(self) -> None:
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, timeout=240, output='partial', stderr='slow')

        with self.assertRaises(CertificateTimeoutError) as caught:
            LeanCertificateRunner(self.repo, run_process=fake_run).compile(
                self._certificate()
            )
        self.assertIn('timed out', str(caught.exception).lower())

    def test_debug_retention_returns_existing_certificate_path(self) -> None:
        def fake_run(argv, **kwargs):
            return _Completed(0, '', '')

        evidence = LeanCertificateRunner(
            self.repo, run_process=fake_run, retain_certificate=True
        ).compile(self._certificate())
        self.assertIsNotNone(evidence.retained_path)
        assert evidence.retained_path is not None
        self.assertTrue(evidence.retained_path.exists())
        self.assertEqual(
            evidence.retained_path.read_text(encoding='utf-8'),
            self._certificate().source,
        )
        shutil.rmtree(evidence.retained_path.parent, ignore_errors=True)

    def test_debug_retention_on_failure_is_diagnostic_only(self) -> None:
        def fake_run(argv, **kwargs):
            return _Completed(1, '', 'bad proof')

        with self.assertRaises(CertificateCompileError) as caught:
            LeanCertificateRunner(
                self.repo, run_process=fake_run, retain_certificate=True
            ).compile(self._certificate())
        retained = getattr(caught.exception, 'certificate_path', None)
        self.assertIsNotNone(retained)
        assert retained is not None
        self.assertTrue(retained.exists())
        shutil.rmtree(retained.parent, ignore_errors=True)

    def test_digest_changes_with_source(self) -> None:
        def fake_run(argv, **kwargs):
            return _Completed(0, '', '')

        runner = LeanCertificateRunner(self.repo, run_process=fake_run)
        first = runner.compile(self._certificate('example : True := True.intro\n'))
        second = runner.compile(self._certificate('example : (1:Nat)=1 := rfl\n'))
        self.assertNotEqual(first.certificate_digest, second.certificate_digest)


@unittest.skipUnless(shutil.which('lake'), 'real Lean smoke requires lake in checkout')
class RunnerLeanSmokeTests(unittest.TestCase):
    def test_real_generated_certificate_compiles(self) -> None:
        repo = Path.cwd()
        if not (repo / 'lakefile.toml').exists():
            self.skipTest('real Lean smoke requires repository checkout')
        source = '''import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

example : (1 : Nat) = 1 := by rfl
'''
        evidence = LeanCertificateRunner(repo).compile(Certificate(source, ()))
        self.assertEqual(evidence.returncode, 0)


if __name__ == '__main__':
    unittest.main()
