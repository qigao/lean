"""Performance guards without wall-clock thresholds or changes to RNG semantics."""
from collections import Counter
import logging
import random

import pytest

from yolo_flywire import graphs
from yolo_flywire.graphs import DirectedGraph, graph_fingerprint, rewire_degree_preserving


def _graph(dense: bool = False) -> DirectedGraph:
    edges = [(a, b) for a in range(8) for b in range(8)
             if (a == b and a % 2 == 0) or (a != b and (a + 2 * b) % 5 < (3 if dense else 2))]
    return DirectedGraph(8, tuple(a for a, _ in edges), tuple(b for _, b in edges),
                         tuple(float(i + 1) for i in range(len(edges))))


def _copying_reference(graph, seed, swaps, max_attempt_factor=100):
    """Test-only oracle from 641768d; never a production fallback."""
    src, dst = list(graph.src), list(graph.dst)
    eligible = [i for i, (a, b) in enumerate(zip(src, dst)) if a != b]
    rng = random.Random(seed)
    edge_set = set(zip(src, dst))
    successful = attempts = 0
    while successful < swaps and attempts < max(max_attempt_factor * swaps, 1):
        attempts += 1
        i, j = rng.sample(eligible, 2)
        a, b, c, d = src[i], dst[i], src[j], dst[j]
        one, two = (a, d), (c, b)
        if a == d or c == b or one == two:
            continue
        old_one, old_two = (a, b), (c, d)
        occupied = edge_set - {old_one, old_two}
        if one in occupied or two in occupied:
            continue
        if one == old_one and two == old_two:
            continue
        edge_set.remove(old_one)
        edge_set.remove(old_two)
        dst[i], dst[j] = d, b
        edge_set.add(one)
        edge_set.add(two)
        successful += 1
    if successful != swaps:
        raise ValueError("attempt budget exhausted")
    return DirectedGraph(graph.num_nodes, tuple(src), tuple(dst), graph.weight)


@pytest.mark.parametrize("seed", [7, 11, 19, 23, 31])
@pytest.mark.parametrize("dense", [False, True])
def test_rewiring_matches_original_proposal_stream_exactly(seed, dense):
    graph = _graph(dense)
    expected = _copying_reference(graph, seed, 100)
    actual = rewire_degree_preserving(graph, seed, 100)
    assert actual == expected  # Includes ordered edges and weights, not only degrees.
    assert graph_fingerprint(actual) == graph_fingerprint(expected)
    assert (Counter(actual.src), Counter(actual.dst)) == (Counter(graph.src), Counter(graph.dst))


def test_rewiring_does_not_copy_the_edge_set_for_collision_checks(monkeypatch):
    graph = _graph(True)
    expected = _copying_reference(graph, 7, 100)

    class NoBulkCopySet(set):
        def __sub__(self, other):
            raise AssertionError("O(E) edge-set copy in rewiring hot loop")

        def difference(self, *others):
            raise AssertionError("O(E) edge-set copy in rewiring hot loop")

        def copy(self):
            raise AssertionError("O(E) edge-set copy in rewiring hot loop")

    # Instrument allocations, but execute the actual sampler and graph operations.
    monkeypatch.setattr(graphs, "set", NoBulkCopySet, raising=False)
    assert rewire_degree_preserving(graph, 7, 100) == expected


def test_progress_reports_attempts_and_completion_without_changing_output(caplog):
    graph = _graph()
    with caplog.at_level(logging.INFO, logger="yolo_flywire.graphs"):
        actual = rewire_degree_preserving(graph, 7, 100)
    assert actual == _copying_reference(graph, 7, 100)
    assert "seed=7" in caplog.text
    assert "accepted=0/100" in caplog.text
    assert "accepted=100/100" in caplog.text
    assert "attempts=" in caplog.text


def test_exhausted_attempt_budget_still_fails_instead_of_reducing_swaps(caplog):
    complete = DirectedGraph(3, (0, 0, 1, 1, 2, 2), (1, 2, 0, 2, 0, 1), (1.0,) * 6)
    with caplog.at_level(logging.INFO, logger="yolo_flywire.graphs"):
        with pytest.raises(ValueError, match="could not complete 10"):
            rewire_degree_preserving(complete, 7, 10, max_attempt_factor=1)
    assert "accepted=0/10" in caplog.text
    assert "attempts=10" in caplog.text


def _identity(graph):
    return {"graph_fingerprint": graph_fingerprint(graph), "seeds": [7, 11],
            "swaps": 100, "protocol_hash": "fixture", "generator_hash": "fixture",
            "runtime": "fixture"}


def test_cached_controls_are_verified_without_running_the_sampler(monkeypatch):
    from yolo_flywire import provenance

    graph = _graph(True)
    identity = _identity(graph)
    bundle = provenance.generate_controls(graph, identity)
    expected = {str(seed): graph_fingerprint(_copying_reference(graph, seed, 100))
                for seed in identity["seeds"]}

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("cache verification must not regenerate controls")

    monkeypatch.setattr(provenance, "rewire_degree_preserving", unexpected_generation)
    assert provenance.verify_controls(bundle, graph, identity) == expected


@pytest.mark.parametrize("field", ["protocol_hash", "generator_hash", "runtime", "swaps", "seeds"])
def test_cached_controls_reject_stale_identity(field):
    from yolo_flywire import provenance

    graph = _graph()
    identity = _identity(graph)
    bundle = provenance.generate_controls(graph, identity)
    changed = dict(identity)
    changed[field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        provenance.verify_controls(bundle, graph, changed)


@pytest.mark.parametrize("corruption", ["missing_seed", "fingerprint", "weight", "diagonal"])
def test_cached_controls_reject_corruption_instead_of_falling_back(corruption):
    from yolo_flywire import provenance

    graph = _graph()
    identity = _identity(graph)
    bundle = provenance.generate_controls(graph, identity)
    control = bundle["controls"]["7"]
    if corruption == "missing_seed":
        del bundle["controls"]["7"]
    elif corruption == "fingerprint":
        control["fingerprint"] = "wrong"
    else:
        values = list(control["graph"]["weight"])
        if corruption == "weight":
            values[0] += 1
        else:
            # Preserve the global weight multiset but move a diagonal weight.
            diagonal = next(i for i, (a, b) in enumerate(zip(graph.src, graph.dst)) if a == b)
            off = next(i for i, (a, b) in enumerate(zip(graph.src, graph.dst)) if a != b)
            values[diagonal], values[off] = values[off], values[diagonal]
        control["graph"]["weight"] = values
        altered = DirectedGraph(graph.num_nodes, tuple(control["graph"]["src"]),
                                tuple(control["graph"]["dst"]), tuple(values))
        control["fingerprint"] = graph_fingerprint(altered)
    with pytest.raises(ValueError):
        provenance.verify_controls(bundle, graph, identity)
