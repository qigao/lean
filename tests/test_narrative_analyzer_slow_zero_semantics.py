import unittest

from narrative_analyzer.certificate import CertificateBuilder
from narrative_analyzer.model import parse_model
from narrative_analyzer.named_schedules import fixed_fixture_route


class SlowZeroClaimSemanticsTests(unittest.TestCase):
    def test_slow_zero_certificate_refutes_existence_of_any_common_limit(self) -> None:
        model = parse_model({
            'topology': {'kind': 'path', 'n': 2},
            'initial': {'beliefs': ['1', '0'], 'exposures': [0, 0]},
            'dynamics': {'threshold': '0'},
            'schedule': {'kind': 'named', 'id': 'slowZeroSchedule'},
        })
        route = fixed_fixture_route(model)
        assert route is not None
        source = CertificateBuilder().build_named_path2(model, route).source
        self.assertIn('¬ ∃ c : Real', source)
        self.assertIn('path2_mean_iterate', source)
        self.assertIn('slowZero_not_consensus', source)


if __name__ == '__main__':
    unittest.main()
