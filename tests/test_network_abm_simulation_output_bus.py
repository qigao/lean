from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.scenario_compiler import (
    compile_situated_scenario_package,
    initialize_compiled_scenario,
)
from narrative_dynamics.abm.scenario_package import load_situated_scenario_package
from narrative_dynamics.abm.simulation_output import project_simulation_output
from narrative_dynamics.abm.simulation_output_bus import (
    SimulationDeliveryFailure,
    SimulationDeliveryReport,
    SimulationOutputBus,
    SimulationOutputSubscription,
)
from narrative_dynamics.abm.simulation_output_contracts import (
    SimulationAudienceCapability,
    SimulationOutputAudience,
    SimulationOutputKind,
    SimulationOutputView,
)
from narrative_dynamics.abm.situated_network import simulate_situated_network_round
from tests.scenario_package_fixtures import (
    mutate_json,
    refresh_manifest_hash,
    write_law_firm_package,
)


class SimulationOutputBusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        root = Path(cls.temporary.name)
        package_root = write_law_firm_package(root / "law-firm")
        mutate_json(
            package_root / "run.json",
            "/allowed_output_kinds",
            [kind.value for kind in SimulationOutputKind],
        )
        refresh_manifest_hash(package_root, "run", "run")
        cls.scenario = compile_situated_scenario_package(
            load_situated_scenario_package(package_root)
        )
        initial_state = initialize_compiled_scenario(
            root / "law-firm.sqlite3",
            cls.scenario,
        )
        cls.round_result = simulate_situated_network_round(
            root / "law-firm.sqlite3",
            cls.scenario.runtime_model,
            initial_state,
        )
        cls.batch = project_simulation_output(
            cls.scenario,
            cls.round_result,
            stream_id="law-firm-bus",
            first_sequence=41,
        )
        cls.all_kinds = tuple(SimulationOutputKind)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_real_batch_is_filtered_by_capability_before_sorted_delivery(self) -> None:
        bus = SimulationOutputBus()
        received: dict[str, SimulationOutputView] = {}
        bus.subscribe(
            "z-public",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            lambda view: received.__setitem__("public", view),
        )
        bus.subscribe(
            "a-alice",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.AGENT, "alice"),
            lambda view: received.__setitem__("alice", view),
        )

        report = bus.publish(self.batch)

        # Delivering the unfiltered batch would disclose objective or private records.
        self.assertEqual(
            report.delivered_subscription_ids,
            ("a-alice", "z-public"),
        )
        self.assertEqual(report.batch_hash, self.batch.content_hash)
        self.assertEqual(report.failures, ())
        self.assertTrue(received["public"].records)
        self.assertTrue(
            all(
                record.audience is SimulationOutputAudience.PUBLIC
                for record in received["public"].records
            )
        )
        self.assertTrue(
            any(
                record.audience is SimulationOutputAudience.AGENT
                for record in received["alice"].records
            )
        )
        self.assertTrue(
            all(
                record.audience is SimulationOutputAudience.PUBLIC
                or (
                    record.audience is SimulationOutputAudience.AGENT
                    and record.owner_agent_id == "alice"
                )
                for record in received["alice"].records
            )
        )

    def test_kind_filter_retains_source_bounds_hash_and_record_sequences(self) -> None:
        bus = SimulationOutputBus()
        received: list[SimulationOutputView] = []
        bus.subscribe(
            "objective-events",
            (SimulationOutputKind.EVENT_OBJECTIVE,),
            SimulationAudienceCapability(SimulationOutputAudience.OBJECTIVE),
            lambda view: received.append(view),
        )

        bus.publish(self.batch)

        view = received[0]
        expected = tuple(
            record.sequence
            for record in self.batch.records
            if record.kind is SimulationOutputKind.EVENT_OBJECTIVE
        )
        # Resequencing a filtered view would sever its source-batch provenance.
        self.assertTrue(expected)
        self.assertEqual(tuple(record.sequence for record in view.records), expected)
        self.assertEqual(view.first_sequence, self.batch.first_sequence)
        self.assertEqual(view.last_sequence, self.batch.last_sequence)
        self.assertEqual(view.source_batch_hash, self.batch.content_hash)
        self.assertTrue(
            all(record.kind is SimulationOutputKind.EVENT_OBJECTIVE for record in view.records)
        )

    def test_callback_exception_is_redacted_and_does_not_stop_later_delivery(self) -> None:
        bus = SimulationOutputBus()
        later: list[SimulationOutputView] = []

        def fail(_: SimulationOutputView) -> None:
            raise RuntimeError("private callback failure text")

        bus.subscribe(
            "a-fails",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            fail,
        )
        bus.subscribe(
            "b-later",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            lambda view: later.append(view),
        )

        report = bus.publish(self.batch)

        # Stopping at the first callback error would make observation alter delivery.
        self.assertEqual(report.delivered_subscription_ids, ("b-later",))
        self.assertEqual(
            report.failures,
            (SimulationDeliveryFailure("a-fails", "callback_error"),),
        )
        self.assertEqual(len(later), 1)
        self.assertNotIn("private callback failure text", repr(report))
        self.assertNotIn("private callback failure text", repr(report.to_dict()))

    def test_non_none_callback_return_is_a_redacted_failure(self) -> None:
        bus = SimulationOutputBus()
        private_value = "private subscriber return value"
        bus.subscribe(
            "returns-value",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            lambda _: private_value,
        )

        report = bus.publish(self.batch)

        # Treating arbitrary returns as success would make the callback contract ambiguous.
        self.assertEqual(report.delivered_subscription_ids, ())
        self.assertEqual(
            report.failures,
            (SimulationDeliveryFailure("returns-value", "non_none_return"),),
        )
        self.assertNotIn(private_value, repr(report))
        self.assertNotIn(private_value, repr(report.to_dict()))

    def test_reentrant_publish_has_stable_error_and_outer_callback_error_only(self) -> None:
        bus = SimulationOutputBus()
        errors: list[str] = []
        later: list[SimulationOutputView] = []

        def publish_again(_: SimulationOutputView) -> None:
            try:
                bus.publish(self.batch)
            except RuntimeError as error:
                errors.append(str(error))
                raise

        bus.subscribe(
            "a-reentrant",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            publish_again,
        )
        bus.subscribe(
            "b-later",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            lambda view: later.append(view),
        )

        report = bus.publish(self.batch)

        # Allowing nested delivery would violate deterministic snapshot ordering.
        self.assertEqual(errors, ["reentrant_publish"])
        self.assertEqual(
            report.failures,
            (SimulationDeliveryFailure("a-reentrant", "callback_error"),),
        )
        self.assertEqual(report.delivered_subscription_ids, ("b-later",))
        self.assertEqual(len(later), 1)

    def test_unsubscribe_during_callback_changes_only_next_publication(self) -> None:
        bus = SimulationOutputBus()
        calls: list[str] = []

        def remove_later(_: SimulationOutputView) -> None:
            calls.append("a-remover")
            bus.unsubscribe("b-removed")

        bus.subscribe(
            "a-remover",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            remove_later,
        )
        bus.subscribe(
            "b-removed",
            self.all_kinds,
            SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
            lambda _: calls.append("b-removed"),
        )

        first = bus.publish(self.batch)
        second = bus.publish(self.batch)

        # Iterating the live table would skip a subscriber removed by an earlier callback.
        self.assertEqual(calls, ["a-remover", "b-removed", "a-remover"])
        self.assertEqual(
            first.delivered_subscription_ids,
            ("a-remover", "b-removed"),
        )
        self.assertEqual(second.delivered_subscription_ids, ("a-remover",))

    def test_contracts_are_frozen_canonical_and_content_addressed(self) -> None:
        capability = SimulationAudienceCapability(SimulationOutputAudience.PUBLIC)
        subscription = SimulationOutputSubscription(
            "subscriber",
            (
                SimulationOutputKind.STATE_DELTA,
                SimulationOutputKind.EVENT_OBJECTIVE,
            ),
            capability,
        )
        failure = SimulationDeliveryFailure("subscriber", "callback_error")
        report = SimulationDeliveryReport(
            self.batch.content_hash,
            ("y", "a"),
            (SimulationDeliveryFailure("z", "non_none_return"), failure),
        )

        # Leaving caller order in canonical contracts would make equal sets hash differently.
        self.assertEqual(
            subscription.kinds,
            (
                SimulationOutputKind.EVENT_OBJECTIVE,
                SimulationOutputKind.STATE_DELTA,
            ),
        )
        self.assertEqual(report.delivered_subscription_ids, ("a", "y"))
        self.assertEqual(
            tuple(item.subscription_id for item in report.failures),
            ("subscriber", "z"),
        )
        self.assertEqual(subscription.content_hash, subscription.content_hash)
        self.assertEqual(failure.content_hash, failure.content_hash)
        self.assertEqual(report.content_hash, report.content_hash)
        with self.assertRaises(FrozenInstanceError):
            subscription.subscription_id = "changed"  # type: ignore[misc]

    def test_subscribe_rejects_empty_kinds_duplicate_ids_and_non_callbacks(self) -> None:
        bus = SimulationOutputBus()
        capability = SimulationAudienceCapability(SimulationOutputAudience.PUBLIC)

        # Accepting an empty kind set would create a misleading successful delivery.
        with self.assertRaisesRegex(ValueError, "non-empty"):
            bus.subscribe("empty", (), capability, lambda _: None)
        bus.subscribe(
            "duplicate",
            (SimulationOutputKind.STATE_DELTA,),
            capability,
            lambda _: None,
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            bus.subscribe(
                "duplicate",
                (SimulationOutputKind.STATE_DELTA,),
                capability,
                lambda _: None,
            )
        with self.assertRaisesRegex(TypeError, "callable"):
            bus.subscribe(
                "not-callable",
                (SimulationOutputKind.STATE_DELTA,),
                capability,
                None,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
