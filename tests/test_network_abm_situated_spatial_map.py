import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm.situated_contracts import (
    EmbodiedAgentSpec,
    PassageSpec,
    PlaceSpec,
    SituatedWorldModel,
)
from narrative_dynamics.abm.situated_spatial_map import (
    auto_layout_situated_spatial_map,
    load_tiled_situated_spatial_map,
)
from narrative_dynamics.abm.situated_spatial_map_contracts import (
    SpatialPassage,
    SpatialPlace,
    SituatedSpatialMap,
)


def office_world() -> SituatedWorldModel:
    return SituatedWorldModel(
        "spatial-office",
        "1",
        (
            PlaceSpec("corridor", "Corridor"),
            PlaceSpec("meeting", "Meeting room"),
        ),
        (PassageSpec("door", "corridor", "meeting", initially_open=False),),
        (
            EmbodiedAgentSpec("alice", "analyst", "corridor"),
            EmbodiedAgentSpec("bob", "manager", "meeting"),
        ),
    )


def tiled_document(*, reverse_layers: bool = False) -> dict[str, object]:
    places = {
        "type": "objectgroup",
        "name": "places",
        "objects": [
            {
                "id": 8,
                "name": "meeting",
                "class": "place",
                "x": 100,
                "y": 50,
                "width": 80,
                "height": 60,
                "rotation": 0,
            },
            {
                "id": 2,
                "name": "corridor",
                "type": "place",
                "x": 0,
                "y": 50,
                "width": 100,
                "height": 40,
            },
        ],
    }
    passages = {
        "type": "group",
        "name": "connections",
        "layers": [
            {
                "type": "objectgroup",
                "name": "passages",
                "objects": [
                    {
                        "id": 10,
                        "name": "door",
                        "class": "passage",
                        "x": 95,
                        "y": 65,
                        "width": 10,
                        "height": 10,
                    }
                ],
            }
        ],
    }
    layers = [places, passages]
    if reverse_layers:
        layers.reverse()
        places["objects"].reverse()
    return {
        "type": "map",
        "orientation": "orthogonal",
        "width": 20,
        "height": 12,
        "tilewidth": 10,
        "tileheight": 10,
        "layers": layers,
    }


class SituatedSpatialMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.world = office_world()

    def tearDown(self):
        self.temporary.cleanup()

    def write_map(self, name: str, document: dict[str, object]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_tiled_rectangles_compile_to_exact_z_up_world_geometry(self):
        path = self.write_map("office.tmj", tiled_document())

        spatial = load_tiled_situated_spatial_map(
            path,
            self.world,
            meters_per_pixel=0.1,
        )

        meeting = next(
            item for item in spatial.places if item.place_id == "meeting"
        )
        door = spatial.passages[0]
        self.assertEqual(
            (
                meeting.center_x,
                meeting.center_y,
                meeting.width,
                meeting.depth,
            ),
            (14.0, -8.0, 8.0, 6.0),
        )
        self.assertEqual(
            (door.center_x, door.center_y, door.width, door.depth),
            (10.0, -7.0, 1.0, 1.0),
        )
        self.assertEqual(
            {item.place_id for item in spatial.places},
            {"corridor", "meeting"},
        )
        self.assertEqual(
            {item.passage_id for item in spatial.passages},
            {"door"},
        )

    def test_tiled_import_hash_ignores_path_layer_order_and_object_ids(self):
        first = self.write_map("first.tmj", tiled_document())
        second_document = tiled_document(reverse_layers=True)
        second_document["nextobjectid"] = 999
        second = self.write_map("nested-second.json", second_document)

        first_map = load_tiled_situated_spatial_map(first, self.world)
        second_map = load_tiled_situated_spatial_map(second, self.world)

        self.assertEqual(first_map.content_hash, second_map.content_hash)
        self.assertEqual(first_map, second_map)

    def test_spatial_map_rejects_missing_world_place(self):
        corridor = SpatialPlace("corridor", 0.0, 0.0, 6.0, 6.0, 2.8)
        door = SpatialPassage(
            "door",
            "corridor",
            "meeting",
            3.0,
            0.0,
            1.0,
            0.3,
            2.1,
        )

        with self.assertRaisesRegex(ValueError, "cover every world place"):
            SituatedSpatialMap(
                "office-map",
                "1",
                self.world,
                (corridor,),
                (door,),
            )

    def test_tiled_import_rejects_rotated_geometry(self):
        document = tiled_document()
        document["layers"][0]["objects"][0]["rotation"] = 45
        path = self.write_map("rotated.tmj", document)

        with self.assertRaisesRegex(ValueError, "rotation"):
            load_tiled_situated_spatial_map(path, self.world)

    def test_auto_layout_is_deterministic_and_places_passage_at_midpoint(self):
        first = auto_layout_situated_spatial_map(
            self.world,
            room_size=6.0,
            gap=2.0,
        )
        second = auto_layout_situated_spatial_map(
            self.world,
            room_size=6.0,
            gap=2.0,
        )
        centres = {
            item.place_id: (item.center_x, item.center_y)
            for item in first.places
        }
        door = first.passages[0]

        self.assertEqual(first, second)
        self.assertEqual(centres, {"corridor": (0.0, 0.0), "meeting": (8.0, 0.0)})
        self.assertEqual((door.center_x, door.center_y), (4.0, 0.0))


if __name__ == "__main__":
    unittest.main()
