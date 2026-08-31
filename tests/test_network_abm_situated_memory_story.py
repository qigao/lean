from tempfile import TemporaryDirectory
import unittest

from narrative_dynamics.abm import (
    SituatedActionIntent,
    SituatedActionKind,
    SituatedMemoryQuery,
    ingest_situated_story,
    list_situated_memories,
    search_situated_memories,
)
from narrative_dynamics.abm.situated_story import (
    advance_situated_story,
    initialize_situated_story,
    perspective_timeline,
)
from tests.situated_fixtures import office_state


def act(action_id, agent_id, kind, target_id=None, *, message=None, source_event_ids=()):
    return SituatedActionIntent(action_id, agent_id, kind, target_id, message, source_event_ids)


def office_memory_story():
    model, initial = office_state()
    story = initialize_situated_story(model, initial)
    story = advance_situated_story(model, story, (
        act("01-inspect", "alice", SituatedActionKind.INSPECT, "memo"),
    ))
    inspection = next(
        event for event in story.rounds[-1].events if event.actor_agent_id == "alice"
    )
    story = advance_situated_story(model, story, (
        act("02-move", "alice", SituatedActionKind.MOVE, "records-open"),
    ))
    story = advance_situated_story(model, story, (
        act(
            "03-tell",
            "alice",
            SituatedActionKind.TELL,
            message="The restructuring is approved.",
            source_event_ids=(inspection.event_id,),
        ),
    ))
    telling = next(
        event for event in story.rounds[-1].events if event.actor_agent_id == "alice"
    )
    return model, story, inspection, telling


class SituatedMemoryStoryTests(unittest.TestCase):
    def test_story_ingestion_projects_only_each_agents_private_perspective(self):
        _, story, inspection, telling = office_memory_story()
        with TemporaryDirectory() as temporary:
            database_path = f"{temporary}/office-memory.sqlite3"
            reports = {
                agent_id: ingest_situated_story(database_path, story, agent_id)
                for agent_id in ("alice", "bob", "dana")
            }

            alice = search_situated_memories(
                database_path, SituatedMemoryQuery("alice", text="restructuring")
            )
            bob = search_situated_memories(
                database_path, SituatedMemoryQuery("bob", text="restructuring")
            )
            dana = search_situated_memories(
                database_path, SituatedMemoryQuery("dana", text="restructuring")
            )

            self.assertIn(inspection.event_id, {item.memory.event_id for item in alice})
            self.assertIn(telling.event_id, {item.memory.event_id for item in alice})
            self.assertNotIn(inspection.event_id, {item.memory.event_id for item in bob})
            self.assertEqual({item.memory.event_id for item in bob}, {telling.event_id})
            self.assertEqual(dana, ())
            self.assertEqual(
                reports["alice"].inserted_count,
                len(perspective_timeline(story, "alice")),
            )
            self.assertEqual(
                reports["bob"].inserted_count,
                len(perspective_timeline(story, "bob")),
            )

    def test_story_ingestion_is_deterministic_across_reopen_and_replay(self):
        _, story, _, _ = office_memory_story()
        with TemporaryDirectory() as temporary:
            database_path = f"{temporary}/durable-memory.sqlite3"
            first = ingest_situated_story(database_path, story, "alice")
            before = tuple(
                item.content_hash
                for item in list_situated_memories(database_path, "alice")
            )

            second = ingest_situated_story(database_path, story, "alice")
            after = tuple(
                item.content_hash
                for item in list_situated_memories(database_path, "alice")
            )

            self.assertGreater(first.inserted_count, 0)
            self.assertEqual(second.inserted_count, 0)
            self.assertEqual(second.existing_count, first.inserted_count)
            self.assertEqual(after, before)

    def test_round_one_memory_remains_retrievable_after_many_later_rounds(self):
        model, story, inspection, _ = office_memory_story()
        for _ in range(50):
            story = advance_situated_story(model, story, ())

        with TemporaryDirectory() as temporary:
            database_path = f"{temporary}/long-history.sqlite3"
            ingest_situated_story(database_path, story, "alice")
            found = search_situated_memories(
                database_path,
                SituatedMemoryQuery(
                    "alice",
                    text="restructuring",
                    event_kinds=(SituatedActionKind.INSPECT,),
                ),
            )

            self.assertEqual(tuple(item.memory.event_id for item in found), (inspection.event_id,))
            self.assertEqual(found[0].memory.round_index, 1)


if __name__ == "__main__":
    unittest.main()
