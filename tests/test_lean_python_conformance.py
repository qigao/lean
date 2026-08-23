import importlib
from pathlib import Path
import tempfile
import unittest


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "conformance"
    / "lean_reference_vectors.json"
)
EXPECTED_OPERATIONS = {
    "bayes_posterior",
    "learn_instrumentality",
    "effective_pressure2",
    "goal_score_single_drive",
}


def conformance_api(test_case):
    try:
        module = importlib.import_module("narrative_dynamics.conformance")
    except ModuleNotFoundError as error:
        test_case.fail(f"Lean Python conformance module is missing: {error}")

    names = (
        "ConformanceDefinitionError",
        "ConformanceMismatch",
        "assert_reference_conformance",
        "default_operation_evaluators",
        "load_reference_suite",
    )
    missing = tuple(name for name in names if getattr(module, name, None) is None)
    test_case.assertEqual(missing, (), f"conformance API is missing: {missing}")
    return module


class LeanPythonConformanceTests(unittest.TestCase):
    def test_committed_lean_vectors_match_python_implementations(self):
        api = conformance_api(self)
        suite = api.load_reference_suite(FIXTURE)

        self.assertEqual(suite.schema_version, 1)
        self.assertEqual(suite.generator, "NarrativeDynamics.Conformance.ReferenceVectors")
        self.assertEqual(suite.numeric_encoding, "reduced_fraction_v1")
        self.assertEqual(
            {vector.operation for vector in suite.vectors},
            EXPECTED_OPERATIONS,
        )
        self.assertGreaterEqual(len(suite.vectors), 12)
        api.assert_reference_conformance(suite)

    def test_checker_detects_injected_python_drift(self):
        api = conformance_api(self)
        suite = api.load_reference_suite(FIXTURE)
        evaluators = api.default_operation_evaluators()
        original = evaluators["learn_instrumentality"]
        evaluators["learn_instrumentality"] = (
            lambda inputs: original(inputs) + 0.125
        )

        with self.assertRaises(api.ConformanceMismatch) as raised:
            api.assert_reference_conformance(suite, evaluators=evaluators)

        self.assertEqual(raised.exception.operation, "learn_instrumentality")
        self.assertTrue(raised.exception.vector_id.startswith("learning_"))
        self.assertNotEqual(raised.exception.actual, raised.exception.expected)

    def test_fixture_contract_rejects_noncanonical_fraction(self):
        api = conformance_api(self)
        malformed = """{
  \"schema_version\": 1,
  \"generator\": \"NarrativeDynamics.Conformance.ReferenceVectors\",
  \"numeric_encoding\": \"reduced_fraction_v1\",
  \"definitions\": [
    \"NarrativeDynamics.bayesPosterior\",
    \"NarrativeDynamics.learnInstrumentality\",
    \"NarrativeDynamics.effectivePressure\",
    \"NarrativeDynamics.goalScore\"
  ],
  \"vectors\": [
    {
      \"id\": \"bayes_bad_fraction\",
      \"operation\": \"bayes_posterior\",
      \"inputs\": {
        \"prior\": \"2/4\",
        \"likelihood_h\": \"3/4\",
        \"likelihood_not_h\": \"1/4\"
      },
      \"expected\": \"3/4\"
    }
  ]
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.json"
            path.write_text(malformed, encoding="utf-8")
            with self.assertRaises(api.ConformanceDefinitionError) as raised:
                api.load_reference_suite(path)

        self.assertEqual(raised.exception.path, "$.vectors[0].inputs.prior")


if __name__ == "__main__":
    unittest.main()
