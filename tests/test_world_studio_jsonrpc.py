from __future__ import annotations

import json
from pathlib import Path

import pytest

from narrative_dynamics.studio import (
    JsonRpcDispatcher,
    JsonRpcLimits,
    ScenarioProjectWorkspace,
    StudioCapability,
    WorldStudioService,
)
from tests.test_world_studio_service import (
    LocalCoordinatorFactory,
    _capability,
    _create_and_import,
    _service,
)


def _error_code(payload: bytes) -> int:
    return json.loads(payload)["error"]["code"]


def test_dispatches_one_exact_jsonrpc_request(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    capability = _capability()
    _create_and_import(service, capability)
    dispatcher = JsonRpcDispatcher(service)

    response = dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "rpc-1",
            "method": "project.snapshot",
            "params": {"project_id": "law-firm"},
        },
        capability,
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "rpc-1"
    assert response["result"]["project_id"] == "law-firm"
    encoded = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","id":"rpc-1","method":"project.snapshot","params":{"project_id":"law-firm"}}',
        capability,
    )
    assert encoded == json.dumps(
        response,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.mark.parametrize(
    ("raw", "code"),
    (
        (b'{"jsonrpc":"2.0"', -32700),
        (b'\xff', -32700),
        (b'[{"jsonrpc":"2.0","id":1,"method":"run.list"}]', -32600),
        (b'null', -32600),
        (b'{"jsonrpc":"1.0","id":1,"method":"run.list"}', -32600),
        (b'{"jsonrpc":"2.0","id":true,"method":"run.list"}', -32600),
        (b'{"jsonrpc":"2.0","id":1,"method":"run.list","extra":1}', -32600),
        (b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":[]}', -32600),
        (b'{"jsonrpc":"2.0","id":1,"method":"worker.claim","params":{}}', -32601),
        (b'{"jsonrpc":"2.0","id":1,"method":"project.snapshot","params":{}}', -32602),
    ),
)
def test_standard_protocol_errors_are_exact(
    tmp_path: Path, raw: bytes, code: int
) -> None:
    service, _ = _service(tmp_path)
    response = JsonRpcDispatcher(service).parse_and_dispatch(raw, _capability())

    assert _error_code(response) == code
    assert json.loads(response)["jsonrpc"] == "2.0"


def test_application_errors_use_stable_codes_without_lookup_oracles(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    dispatcher = JsonRpcDispatcher(service)
    full = _capability()
    created = service.invoke("project.create", {"project_id": "law-firm"}, full)

    unauthorized = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","id":1,"method":"project.snapshot","params":{"project_id":"law-firm"}}',
        _capability(project_ids=(), permissions=("project.read",)),
    )
    unknown = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","id":2,"method":"project.snapshot","params":{"project_id":"unknown"}}',
        _capability(project_ids=("unknown",), permissions=("project.read",)),
    )
    validation = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","id":3,"method":"scenario.compile","params":{"project_id":"law-firm"}}',
        full,
    )
    stale_request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "project.undo",
            "params": {
                "project_id": "law-firm",
                "expected_revision": created["revision"] + 1,
                "expected_snapshot_hash": created["content_hash"],
            },
        },
        separators=(",", ":"),
    ).encode()
    stale = dispatcher.parse_and_dispatch(stale_request, full)
    conflict = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","id":5,"method":"project.create","params":{"project_id":"law-firm"}}',
        full,
    )

    assert _error_code(unauthorized) == -32010
    assert _error_code(unknown) == -32014
    assert _error_code(validation) == -32012
    assert _error_code(stale) == -32011
    assert _error_code(conflict) == -32015
    assert json.loads(unauthorized)["error"]["message"] == "Unauthorized"


def test_notifications_reject_mutation_but_execute_read_only_without_response(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    capability = _capability()
    _create_and_import(service, capability)
    dispatcher = JsonRpcDispatcher(service)

    read_only = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","method":"project.snapshot","params":{"project_id":"law-firm"}}',
        capability,
    )
    mutation = dispatcher.parse_and_dispatch(
        b'{"jsonrpc":"2.0","method":"project.create","params":{"project_id":"law-firm"}}',
        capability,
    )

    assert read_only == b""
    assert _error_code(mutation) == -32600
    assert json.loads(mutation)["id"] is None


@pytest.mark.parametrize(
    "raw",
    (
        b'{"jsonrpc":"2.0","id":1,"id":2,"method":"run.list","params":{}}',
        b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":{"x":NaN}}',
        b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":{"x":Infinity}}',
        b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":{"x":-Infinity}}',
    ),
)
def test_strict_parser_rejects_duplicates_and_non_finite_numbers(
    tmp_path: Path, raw: bytes
) -> None:
    service, _ = _service(tmp_path)

    response = JsonRpcDispatcher(service).parse_and_dispatch(raw, _capability())

    assert _error_code(response) == -32700


@pytest.mark.parametrize(
    ("limits", "raw"),
    (
        (
            JsonRpcLimits(maximum_bytes=16),
            b'{"jsonrpc":"2.0","id":1,"method":"run.list"}',
        ),
        (
            JsonRpcLimits(maximum_depth=2),
            b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":{"x":{"y":1}}}',
        ),
        (
            JsonRpcLimits(maximum_members=3),
            b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":{}}',
        ),
        (
            JsonRpcLimits(maximum_string_bytes=4),
            b'{"jsonrpc":"2.0","id":1,"method":"run.list","params":{}}',
        ),
    ),
)
def test_raw_parser_limits_fail_closed(
    tmp_path: Path, limits: JsonRpcLimits, raw: bytes
) -> None:
    service, _ = _service(tmp_path)

    response = JsonRpcDispatcher(service, limits=limits).parse_and_dispatch(
        raw, _capability()
    )

    assert _error_code(response) == -32700


class SecretFailingFactory(LocalCoordinatorFactory):
    def create(self, scenario, *, project_id: str, run_id: str, stream_id: str):
        del scenario, project_id, run_id, stream_id
        raise ValueError(
            r"provider token sk-secret-token at C:\private\operator\workspace.sqlite3"
        )


def test_untrusted_factory_exception_is_redacted_from_raw_response(tmp_path: Path) -> None:
    service, registry = _service(tmp_path)
    capability = _capability()
    imported = _create_and_import(service, capability)
    reopened_workspace = ScenarioProjectWorkspace.open(
        tmp_path / "studio.sqlite3",
        import_roots=(Path("examples/law_firm_scenario").resolve().parent,),
        export_root=tmp_path,
    )
    failing_service = WorldStudioService(
        reopened_workspace,
        registry,
        SecretFailingFactory(tmp_path),
        import_sources={"law-firm-fixture": Path("examples/law_firm_scenario").resolve()},
    )
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "secret-test",
            "method": "run.create",
            "params": {
                "project_id": "law-firm",
                "run_id": "run-1",
                "stream_id": "stream-1",
                "expected_revision": imported["revision"],
                "expected_snapshot_hash": imported["content_hash"],
            },
        },
        separators=(",", ":"),
    ).encode()

    response = JsonRpcDispatcher(failing_service).parse_and_dispatch(raw, capability)

    assert _error_code(response) == -32603
    assert b"sk-secret-token" not in response
    assert b"private" not in response
    assert b"workspace.sqlite3" not in response
    data = json.loads(response)["error"]["data"]
    assert set(data) == {"diagnostic_id"}
    assert data["diagnostic_id"].startswith("sha256:")


def test_limits_are_positive_bounded_values() -> None:
    assert JsonRpcLimits().maximum_bytes == 1_048_576
    with pytest.raises((TypeError, ValueError)):
        JsonRpcLimits(maximum_depth=0)
    with pytest.raises((TypeError, ValueError)):
        JsonRpcLimits(maximum_members=True)
