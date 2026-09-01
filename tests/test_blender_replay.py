import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated_network import (
    initialize_situated_network_runtime,
    simulate_situated_network_runtime,
)
from narrative_dynamics.abm.situated_spatial_map import (
    auto_layout_situated_spatial_map,
)
from narrative_dynamics.integrations.blender_replay import (
    compile_situated_blend_replay,
)
from tests.test_network_abm_situated_network import (
    initial_runtime_case,
)
from tests.test_network_abm_situated_spatial_map import office_world


SECRET = "The restructuring is approved."


class BlenderReplayCompilationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "replay.sqlite3"
        self.model, story, cognitive, social = initial_runtime_case()
        initial = initialize_situated_network_runtime(
            self.database,
            self.model,
            story,
            cognitive,
            social,
        )
        self.trajectory = simulate_situated_network_runtime(
            self.database,
            self.model,
            initial,
            round_count=2,
        )
        self.world = self.model.percept_memory_model.cognitive_model.world_model
        self.spatial = auto_layout_situated_spatial_map(
            self.world,
            room_size=6.0,
            gap=2.0,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_replay_compiles_map_movement_door_state_and_events(self):
        packet = compile_situated_blend_replay(
            self.model,
            self.trajectory,
            self.spatial,
            frames_per_round=12,
        )

        alice = next(
            item for item in packet["actors"] if item["agent_id"] == "alice"
        )
        alice_move = next(
            item
            for item in packet["events"]
            if item["actor_agent_id"] == "alice" and item["kind"] == "move"
        )
        closed_door = next(
            item
            for item in packet["passages"]
            if item["passage_id"] == "open-manager"
        )

        self.assertEqual(
            (packet["first_frame"], packet["last_frame"]),
            (1, 25),
        )
        self.assertEqual(
            [state["place_id"] for state in alice["states"]],
            ["records", "records", "open"],
        )
        self.assertEqual([state["frame"] for state in alice["states"]], [1, 13, 25])
        self.assertEqual(alice_move["frame"], 25)
        self.assertEqual(
            [state["open"] for state in closed_door["states"]],
            [False, False, False],
        )

    def test_replay_packet_never_contains_tell_message_or_private_details(self):
        packet = compile_situated_blend_replay(
            self.model,
            self.trajectory,
            self.spatial,
        )
        serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True)

        self.assertNotIn(SECRET, serialized)
        self.assertNotIn('"details"', serialized)
        self.assertNotIn('"message"', serialized)
        self.assertNotIn(str(self.database), serialized)

    def test_replay_is_deterministic_and_binds_all_source_hashes(self):
        first = compile_situated_blend_replay(
            self.model,
            self.trajectory,
            self.spatial,
        )
        second = compile_situated_blend_replay(
            self.model,
            self.trajectory,
            self.spatial,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["runtime_model_hash"], self.model.content_hash)
        self.assertEqual(first["trajectory_hash"], self.trajectory.content_hash)
        self.assertEqual(first["world_model_hash"], self.world.content_hash)
        self.assertEqual(first["spatial_map_hash"], self.spatial.content_hash)
        self.assertRegex(first["content_hash"], r"^sha256:[0-9a-f]{64}$")

    def test_replay_rejects_map_for_a_different_world(self):
        unrelated = auto_layout_situated_spatial_map(office_world())

        with self.assertRaisesRegex(ValueError, "spatial map"):
            compile_situated_blend_replay(
                self.model,
                self.trajectory,
                unrelated,
            )


if __name__ == "__main__":
    unittest.main()
