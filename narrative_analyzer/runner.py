"""Bounded fail-closed Lean certificate execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Protocol

from .certificate import Certificate
from .result import CertificateCompileError, CertificateTimeoutError


class _ProcessResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


RunProcess = Callable[..., _ProcessResult]


@dataclass(frozen=True)
class CompileEvidence:
    certificate_digest: str
    returncode: int
    stdout: str
    stderr: str
    retained_path: Path | None = None


class LeanCertificateRunner:
    """Compile generated Lean source under the repository fail-fast policy."""

    def __init__(
        self,
        repo_root: Path,
        *,
        run_process: RunProcess = subprocess.run,
        retain_certificate: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self._run_process = run_process
        self.retain_certificate = retain_certificate

    @staticmethod
    def _digest(certificate: Certificate) -> str:
        return hashlib.sha256(certificate.source.encode("utf-8")).hexdigest()

    @staticmethod
    def _command(path: Path) -> list[str]:
        return [
            "timeout",
            "--kill-after=10s",
            "240s",
            "lake",
            "env",
            "lean",
            "-DmaxErrors=1",
            "-DstderrAsMessages=false",
            str(path),
        ]

    @staticmethod
    def _attach_path(error: Exception, path: Path | None) -> Exception:
        if path is not None:
            setattr(error, "certificate_path", path)
        return error

    def _compile_path(
        self,
        certificate: Certificate,
        path: Path,
        *,
        retained_path: Path | None,
    ) -> CompileEvidence:
        path.write_text(certificate.source, encoding="utf-8")
        command = self._command(path)
        try:
            completed = self._run_process(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=250,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            diagnostics = "\n".join(
                part
                for part in (
                    str(getattr(exc, "output", "") or ""),
                    str(getattr(exc, "stderr", "") or ""),
                )
                if part
            )
            error = CertificateTimeoutError(
                "Lean certificate timed out after bounded execution"
                + (f":\n{diagnostics}" if diagnostics else "")
            )
            raise self._attach_path(error, retained_path) from exc

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode in (124, 137):
            error = CertificateTimeoutError(
                f"Lean certificate timed out (exit {completed.returncode})"
                + (f":\n{stderr or stdout}" if (stderr or stdout) else "")
            )
            raise self._attach_path(error, retained_path)
        if completed.returncode != 0:
            diagnostics = stderr or stdout or "no compiler diagnostics"
            error = CertificateCompileError(
                f"Lean certificate failed with exit {completed.returncode}:\n{diagnostics}"
            )
            raise self._attach_path(error, retained_path)

        return CompileEvidence(
            certificate_digest=self._digest(certificate),
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            retained_path=retained_path,
        )

    def compile(self, certificate: Certificate) -> CompileEvidence:
        if self.retain_certificate:
            directory = Path(tempfile.mkdtemp(prefix="narrative-analyzer-"))
            path = directory / "AnalyzerCertificate.lean"
            return self._compile_path(
                certificate,
                path,
                retained_path=path,
            )

        with tempfile.TemporaryDirectory(prefix="narrative-analyzer-") as directory:
            path = Path(directory) / "AnalyzerCertificate.lean"
            return self._compile_path(certificate, path, retained_path=None)
