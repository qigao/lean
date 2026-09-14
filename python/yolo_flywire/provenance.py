"""Generate matched graph controls once; validate their provenance on every run.

This command never trains a classifier or opens a behavioral final-test dataset.
Cache identity changes are misses at the CI boundary; an invalid restored bundle
is an error, not an invitation to silently regenerate or substitute controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np

from .graphs import DirectedGraph, graph_fingerprint, rewire_degree_preserving

_LOGGER = logging.getLogger(__name__)
_SEED_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
_ALGORITHM = "directed-double-edge-swap-v2-diagonal-fixed"


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_rewiring_protocol(config: dict[str, Any]) -> None:
    """Check the frozen control contract, not empirical or byte-level validity.

    A well-formed digest is only a declaration. The provenance command separately
    verifies actual source and control graph bytes against these declared values.
    """
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("rewiring protocol requires unique integer seeds")
    if config.get("rewiring_algorithm") != _ALGORITHM:
        raise ValueError("unsupported rewiring algorithm; no compatibility fallback")
    multiplier = config.get("rewiring_successful_swaps_per_offdiagonal_edge")
    if type(multiplier) is not int or multiplier != 10:
        raise ValueError("rewiring requires integer budget of ten successful swaps per off-diagonal edge")
    original = config.get("selected_graph_fingerprint")
    if not _is_sha256(original):
        raise ValueError("selected_graph_fingerprint must be a canonical lowercase SHA-256")
    fingerprints = config.get("rewired_graph_fingerprints")
    if (not isinstance(fingerprints, dict)
            or set(fingerprints) != {str(seed) for seed in seeds}):
        raise ValueError("rewired_graph_fingerprints must cover exactly every declared seed")
    for seed, fingerprint in fingerprints.items():
        if not _is_sha256(fingerprint):
            raise ValueError(f"rewired_graph_fingerprints[{seed}] must be a canonical lowercase SHA-256")
        if fingerprint == original:
            raise ValueError(f"rewired fingerprint for seed {seed} must differ from the original graph")


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _diagonal(graph: DirectedGraph) -> set[tuple[int, int, float]]:
    return {(a, b, w) for a, b, w in zip(graph.src, graph.dst, graph.weight) if a == b}


def _check_identity(graph: DirectedGraph, identity: dict[str, Any]) -> None:
    seeds = identity.get("seeds")
    swaps = identity.get("swaps")
    if identity.get("graph_fingerprint") != graph_fingerprint(graph):
        raise ValueError("cache identity graph mismatch")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("cache identity requires unique integer seeds")
    if type(swaps) is not int or swaps <= 0:
        raise ValueError("cache identity requires a positive successful-swap budget")


def _check_matched(actual: DirectedGraph, original: DirectedGraph) -> None:
    if actual.num_nodes != original.num_nodes or actual.num_edges != original.num_edges:
        raise ValueError("cached graph size changed")
    if (Counter(actual.src), Counter(actual.dst)) != (Counter(original.src), Counter(original.dst)):
        raise ValueError("cached directed degrees changed")
    if sorted(actual.weight) != sorted(original.weight):
        raise ValueError("cached edge-weight multiset changed")
    if _diagonal(actual) != _diagonal(original):
        raise ValueError("cached diagonal edge+weight set changed")
    if graph_fingerprint(actual) == graph_fingerprint(original):
        raise ValueError("rewired control must differ from the original graph")


def generate_controls(graph: DirectedGraph, identity: dict[str, Any]) -> dict[str, Any]:
    _check_identity(graph, identity)
    controls = {}
    for seed in identity["seeds"]:
        started = perf_counter()
        _LOGGER.info("seed=%s starting full budget: %s successful swaps", seed, identity["swaps"])
        rewired = rewire_degree_preserving(graph, seed=seed, swaps=identity["swaps"])
        _check_matched(rewired, graph)
        fingerprint = graph_fingerprint(rewired)
        controls[str(seed)] = {"graph": asdict(rewired), "fingerprint": fingerprint,
                               "successful_swaps": identity["swaps"]}
        _LOGGER.info("seed=%s completed in %.3fs fingerprint=%s", seed, perf_counter() - started, fingerprint)
    # Round-trip gives an independent JSON-compatible snapshot of identity/graphs.
    return json.loads(json.dumps({"format_version": 1, "identity": identity, "controls": controls},
                                 allow_nan=False))


def verify_controls(bundle: dict[str, Any], graph: DirectedGraph,
                    identity: dict[str, Any]) -> dict[str, str]:
    if not isinstance(bundle, dict) or bundle.get("identity") != identity:
        raise ValueError("cache identity mismatch; refusing stale controls")
    _check_identity(graph, identity)
    if bundle.get("format_version") != 1:
        raise ValueError("unsupported control bundle format")
    controls = bundle.get("controls")
    if not isinstance(controls, dict) or set(controls) != {str(s) for s in identity["seeds"]}:
        raise ValueError("cached controls must cover exactly the frozen seed set")
    fingerprints = {}
    for seed in identity["seeds"]:
        control = controls[str(seed)]
        try:
            data = control["graph"]
            if type(data["num_nodes"]) is not int or any(
                type(v) is not int for v in data["src"] + data["dst"]
            ):
                raise ValueError("cached graph endpoints must be integers")
            actual = DirectedGraph(data["num_nodes"], tuple(data["src"]),
                                   tuple(data["dst"]), tuple(data["weight"]))
            fingerprint = graph_fingerprint(actual)
            if control["fingerprint"] != fingerprint:
                raise ValueError("cached graph fingerprint mismatch")
            if control["successful_swaps"] != identity["swaps"]:
                raise ValueError("cached successful-swap budget mismatch")
            _check_matched(actual, graph)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"seed {seed}: malformed cached graph") from exc
        fingerprints[str(seed)] = fingerprint
    return fingerprints


def _context(config: dict[str, Any], source: Path):
    from .flywire import load_visual_type_graph

    validate_rewiring_protocol(config)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["flywire_connectivity_sha256"]:
        raise ValueError("FlyWire source SHA-256 mismatch")
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != config["flywire_connectivity_git_blob_sha1"]:
        raise ValueError("FlyWire source Git blob mismatch")
    selection = load_visual_type_graph(source, seed_types=_SEED_TYPES, min_seed_synapses=5)
    graph = selection.graph
    stats = {"selected_graph_num_nodes": graph.num_nodes,
             "selected_graph_num_edges": graph.num_edges,
             "selected_graph_num_diagonal_edges": len(_diagonal(graph)),
             "selected_graph_fingerprint": graph_fingerprint(graph)}
    if any(config.get(key) != value for key, value in stats.items()):
        raise ValueError("selected FlyWire graph does not match frozen fingerprint/statistics")
    package = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
               for name in ("__init__.py", "graphs.py", "flywire.py", "provenance.py")}
    identity = {"format_version": 1, "graph_fingerprint": graph_fingerprint(graph),
                "protocol_hash": _hash_json(config), "generator_hash": _hash_json(sources),
                "source_sha256": config["flywire_connectivity_sha256"],
                "seeds": config["seeds"], "swaps": 10 * (graph.num_edges - len(_diagonal(graph))),
                "runtime": {"python": sys.version, "numpy": np.__version__,
                            "implementation": sys.implementation.name,
                            "platform": sys.platform, "machine": platform.machine()}}
    _check_identity(graph, identity)
    return selection, identity


def _write_json(path: Path, value: Any) -> None:
    # Commit complete records only. Interrupted generation never publishes a bundle.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("key", "generate", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = perf_counter()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        selection, identity = _context(config, args.source)
        graph = selection.graph
        key = _hash_json(identity)
        if args.mode == "key":
            print(key)
            return 0
        _LOGGER.info("source verified: nodes=%s edges=%s swaps/seed=%s mode=%s",
                     graph.num_nodes, graph.num_edges, identity["swaps"], args.mode)
        bundle_path = args.output / "control-bundle.json"
        if args.mode == "generate":
            bundle = generate_controls(graph, identity)
        else:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        fingerprints = verify_controls(bundle, graph, identity)
        if fingerprints != config["rewired_graph_fingerprints"]:
            raise ValueError("rewired fingerprints do not match the frozen protocol")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.mode == "generate":
            _write_json(bundle_path, bundle)
        # Rebuild reports from verified data; do not trust a cached summary alone.
        report = {"node_types": selection.node_types, "num_nodes": graph.num_nodes,
                  "num_edges": graph.num_edges, "num_diagonal_edges": len(_diagonal(graph)),
                  "graph_fingerprint": graph_fingerprint(graph), "seed_types": _SEED_TYPES,
                  "min_seed_synapses": 5, "rewiring_algorithm": _ALGORITHM,
                  "rewiring_successful_swaps": identity["swaps"],
                  "rewiring_successful_swaps_per_offdiagonal_edge": 10,
                  "rewired_graph_fingerprints": fingerprints, "cache_identity": identity}
        _write_json(args.output / "flywire-selected-graph.json", report)
        _write_json(args.output / "verification-run.json",
                    {"code_commit": os.environ.get("GITHUB_SHA", "working-tree"),
                     "cache_identity_hash": key, "mode": args.mode,
                     "evidence_scope": "graph_provenance_only", "verified_seeds": identity["seeds"]})
        _LOGGER.info("%s verified %s controls in %.3fs (no classifier training)",
                     args.mode, len(fingerprints), perf_counter() - started)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"provenance error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
