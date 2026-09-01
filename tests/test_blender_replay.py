import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import narrative_dynamics.integrations as integrations
from narrative_dynamics.abm.situated_network import (
    initialize_situated_network_runtime,
    simulate_situated_network_runtime,
)
from narrative_dynamics.abm.situated_spatial_map import (
    auto_layout_situated_spatial_map,
)
from narrative_dynamics.integrations.blender_replay import (
    BlenderExportError,
    compile_situated_blend_replay,
    export_situated_network_blend,
)
from tests.test_network_abm_situated_network import (
    initial_runtime_case,
)
from tests.test_network_abm_situated_spatial_map import office_world


SECRET = "The restructuring is approved."


def write_fake_blender(root: Path, *, mode: str = "success") -> Path:
    worker = Path(__file__).with_name("fake_blender.py").resolve()
    extra = {
        "success": "",
        "fail": "--fake-fail",
        "invalid": "--fake-invalid",
    }[mode]
    if os.name == "nt":
        path = root / f"fake-blender-{mode}.cmd"
        path.write_text(
            f'@echo off\r\n"{sys.executable}" "{worker}" {extra} %*\r\n',
            encoding="utf-8",
        )
    else:
        path = root / f"fake-blender-{mode}"
        command = " ".join(
            item
            for item in (
                shlex.quote(sys.executable),
                shlex.quote(str(worker)),
                extra,
                '"$@"',
            )
            if item
        )
        path.write_text(f"#!/bin/sh\nexec {command}\n", encoding="utf-8")
        path.chmod(0o755)
    return path


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

    def test_blender_export_is_available_from_integrations_public_api(self):
        self.assertIs(
            integrations.export_situated_network_blend,
            export_situated_network_blend,
        )
        self.assertIn("BlenderExportReport", integrations.__all__)

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

    def test_export_stages_packet_and_atomically_publishes_blend(self):
        output = self.root / "office.blend"

        report = export_situated_network_blend(
            write_fake_blender(self.root),
            output,
            self.model,
            self.trajectory,
            self.spatial,
        )

        self.assertEqual(output.read_bytes()[:7], b"BLENDER")
        self.assertEqual(report.output_path, str(output.resolve()))
        self.assertEqual(report.trajectory_hash, self.trajectory.content_hash)
        self.assertEqual(report.spatial_map_hash, self.spatial.content_hash)
        self.assertEqual((report.first_frame, report.last_frame), (1, 49))

    def test_failed_blender_process_preserves_existing_output(self):
        output = self.root / "office.blend"
        output.write_bytes(b"BLENDER-PRIOR")

        with self.assertRaises(BlenderExportError):
            export_situated_network_blend(
                write_fake_blender(self.root, mode="fail"),
                output,
                self.model,
                self.trajectory,
                self.spatial,
            )

        self.assertEqual(output.read_bytes(), b"BLENDER-PRIOR")

    def test_invalid_blender_output_preserves_existing_output(self):
        output = self.root / "office.blend"
        output.write_bytes(b"BLENDER-PRIOR")

        with self.assertRaises(BlenderExportError):
            export_situated_network_blend(
                write_fake_blender(self.root, mode="invalid"),
                output,
                self.model,
                self.trajectory,
                self.spatial,
            )

        self.assertEqual(output.read_bytes(), b"BLENDER-PRIOR")

    def test_export_requires_blend_suffix_before_launch(self):
        output = self.root / "office.json"

        with self.assertRaisesRegex(ValueError, r"\.blend"):
            export_situated_network_blend(
                self.root / "missing-blender",
                output,
                self.model,
                self.trajectory,
                self.spatial,
            )

    @unittest.skipUnless(
        os.environ.get("BLENDER_EXECUTABLE"),
        "BLENDER_EXECUTABLE is required for the real Blender smoke test",
    )
    def test_real_blender_builds_and_reopens_complete_scene(self):
        blender = Path(os.environ["BLENDER_EXECUTABLE"])
        output = self.root / "real-office.blend"

        export_situated_network_blend(
            blender,
            output,
            self.model,
            self.trajectory,
            self.spatial,
            timeout_seconds=120.0,
        )
        verifier = Path(__file__).with_name("verify_blender_replay.py").resolve()
        completed = subprocess.run(
            (
                str(blender),
                "--background",
                str(output),
                "--python-exit-code",
                "22",
                "--python",
                str(verifier),
            ),
            capture_output=True,
            check=False,
            timeout=120.0,
        )

        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout.decode("utf-8", errors="replace")
            + completed.stderr.decode("utf-8", errors="replace"),
        )
        self.assertIn(b"ND_BLEND_VERIFY_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
