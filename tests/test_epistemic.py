import unittest

from epistemic import apply_observation, delivered_observation


class EpistemicTests(unittest.TestCase):
    def test_disconnected_event_cannot_change_belief(self):
        edges = {"event": {"relay"}, "relay": set(), "agent": set()}
        obs = delivered_observation(edges, "event", "agent", {"panic": True})
        self.assertIsNone(obs)
        prior = 0.2
        posterior = apply_observation(prior, obs, lambda p, _o: min(1.0, p + 0.5))
        self.assertEqual(posterior, prior)

    def test_multihop_information_path_delivers_observation(self):
        edges = {"event": {"relay"}, "relay": {"agent"}, "agent": set()}
        payload = {"panic": True}
        obs = delivered_observation(edges, "event", "agent", payload)
        self.assertEqual(obs, payload)


if __name__ == "__main__":
    unittest.main()
