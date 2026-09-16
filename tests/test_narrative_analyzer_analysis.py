import hashlib
import unittest

from narrative_analyzer.analyze import analyze_model
from narrative_analyzer.model import parse_model
from narrative_analyzer.result import (
    CertificateCompileError,
    ClaimStatus,
    ProvenanceMismatchError,
)
from narrative_analyzer.runner import CompileEvidence


def model_document(
    *,
    n=7,
    beliefs=None,
    exposures=None,
    threshold='0',
    schedule=None,
):
    if beliefs is None:
        beliefs = ['1'] * n
    if exposures is None:
        exposures = [0] * n
    if schedule is None:
        schedule = {'kind': 'constant', 'value': '1/4'}
    return {
        'topology': {'kind': 'path', 'n': n},
        'initial': {'beliefs': beliefs, 'exposures': exposures},
        'dynamics': {'threshold': threshold},
        'schedule': schedule,
    }


class FakeRunner:
    def __init__(self, *, bad_digest=False, fail=False):
        self.bad_digest = bad_digest
        self.fail = fail
        self.certificates = []

    def compile(self, certificate):
        self.certificates.append(certificate)
        if self.fail:
            raise CertificateCompileError('synthetic Lean failure')
        digest = hashlib.sha256(certificate.source.encode('utf-8')).hexdigest()
        if self.bad_digest:
            digest = '0' * 64
        return CompileEvidence(digest, 0, '', '')


def by_id(result):
    return {claim.claim_id: claim for claim in result.claims}


class AnalyzerOrchestrationTests(unittest.TestCase):
    def test_constant_path7_proves_structural_interior_and_consensus(self):
        runner = FakeRunner()
        result = analyze_model(parse_model(model_document()), runner=runner)
        claims = by_id(result)
        for claim_id in (
            'parameters_valid',
            'initial_all_broadcast',
            'exposure_law',
            'effective_alpha_lookup',
            'reachable_interior',
            'pathn_consensus_exists',
        ):
            self.assertIs(claims[claim_id].status, ClaimStatus.PROVED)
        self.assertIs(claims['consensus_value_known'].status, ClaimStatus.UNKNOWN)
        self.assertEqual(claims['reachable_interior'].exact_values['eps'], '1/4')
        self.assertIs(result.consensus_status, ClaimStatus.PROVED)
        self.assertGreaterEqual(len(runner.certificates), 2)

    def test_piecewise_path7_uses_exact_global_interior_route(self):
        schedule = {
            'kind': 'piecewise',
            'default': '1/4',
            'points': [
                {'exposure': 0, 'value': '1/3'},
                {'exposure': 4, 'value': '2/5'},
                {'exposure': 9, 'value': '4/5'},
            ],
        }
        result = analyze_model(
            parse_model(model_document(schedule=schedule)), runner=FakeRunner()
        )
        claims = by_id(result)
        self.assertIs(claims['pathn_consensus_exists'].status, ClaimStatus.PROVED)
        self.assertEqual(claims['reachable_interior'].exact_values['eps'], '1/5')

    def test_no_global_interior_route_stays_unknown_not_disproved(self):
        model = parse_model(
            model_document(schedule={'kind': 'constant', 'value': '0'})
        )
        result = analyze_model(model, runner=FakeRunner())
        claims = by_id(result)
        self.assertIs(claims['parameters_valid'].status, ClaimStatus.PROVED)
        self.assertIs(claims['reachable_interior'].status, ClaimStatus.UNKNOWN)
        self.assertIs(claims['pathn_consensus_exists'].status, ClaimStatus.UNKNOWN)
        self.assertIs(result.consensus_status, ClaimStatus.UNKNOWN)

    def test_invalid_parameters_are_disproved_but_consensus_is_not(self):
        model = parse_model(
            model_document(schedule={'kind': 'constant', 'value': '5/4'})
        )
        result = analyze_model(model, runner=FakeRunner())
        claims = by_id(result)
        self.assertIs(claims['parameters_valid'].status, ClaimStatus.DISPROVED)
        self.assertIs(claims['pathn_consensus_exists'].status, ClaimStatus.UNKNOWN)
        self.assertIs(result.consensus_status, ClaimStatus.UNKNOWN)

    def test_false_all_broadcast_is_disproved_and_dependent_claims_unknown(self):
        model = parse_model(
            model_document(n=2, beliefs=['1', '0'], threshold='1/2')
        )
        result = analyze_model(model, runner=FakeRunner())
        claims = by_id(result)
        self.assertIs(claims['parameters_valid'].status, ClaimStatus.PROVED)
        self.assertIs(claims['initial_all_broadcast'].status, ClaimStatus.DISPROVED)
        self.assertIs(claims['exposure_law'].status, ClaimStatus.UNKNOWN)
        self.assertIs(claims['effective_alpha_lookup'].status, ClaimStatus.UNKNOWN)
        self.assertIs(claims['pathn_consensus_exists'].status, ClaimStatus.UNKNOWN)

    def _named_split2(self, schedule_id):
        return parse_model(
            model_document(
                n=2,
                beliefs=['1', '0'],
                exposures=[0, 0],
                schedule={'kind': 'named', 'id': schedule_id},
            )
        )

    def test_slow_zero_fixture_is_theorem_backed_disproved(self):
        result = analyze_model(self._named_split2('slowZeroSchedule'), runner=FakeRunner())
        claim = by_id(result)['path2_consensus']
        self.assertIs(claim.status, ClaimStatus.DISPROVED)
        self.assertIn('slowZero_not_consensus', claim.theorem)
        self.assertIs(result.consensus_status, ClaimStatus.DISPROVED)

    def test_near_one_fixture_is_theorem_backed_disproved(self):
        result = analyze_model(self._named_split2('nearOneSchedule'), runner=FakeRunner())
        claim = by_id(result)['path2_consensus']
        self.assertIs(claim.status, ClaimStatus.DISPROVED)
        self.assertIn('nearOne_not_convergent', claim.theorem)

    def test_harmonic_fixture_proves_consensus_and_value(self):
        result = analyze_model(self._named_split2('harmonicSchedule'), runner=FakeRunner())
        claims = by_id(result)
        self.assertIs(claims['path2_consensus'].status, ClaimStatus.PROVED)
        self.assertIs(claims['consensus_value_known'].status, ClaimStatus.PROVED)
        self.assertEqual(claims['consensus_value_known'].exact_values['value'], '1/2')
        self.assertIs(result.consensus_status, ClaimStatus.PROVED)

    def test_changed_named_fixture_is_unknown_for_stronger_path2_claim(self):
        model = parse_model(
            model_document(
                n=2,
                beliefs=['1', '0'],
                exposures=[0, 1],
                schedule={'kind': 'named', 'id': 'harmonicSchedule'},
            )
        )
        result = analyze_model(model, runner=FakeRunner())
        self.assertIs(by_id(result)['path2_consensus'].status, ClaimStatus.UNKNOWN)

    def test_compile_failure_propagates_instead_of_unknown(self):
        with self.assertRaises(CertificateCompileError):
            analyze_model(parse_model(model_document()), runner=FakeRunner(fail=True))

    def test_digest_mismatch_is_provenance_failure(self):
        with self.assertRaises(ProvenanceMismatchError):
            analyze_model(parse_model(model_document()), runner=FakeRunner(bad_digest=True))


if __name__ == '__main__':
    unittest.main()
