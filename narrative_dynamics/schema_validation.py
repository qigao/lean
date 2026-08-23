from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import TYPE_CHECKING

from narrative_dynamics.contracts import (
    ExecutionCapture,
    Scenario,
    TraceEvent,
    stable_content_hash,
)

if TYPE_CHECKING:
    from narrative_dynamics.model_contract import ModelSchema


RUNTIME_SCHEMA_DIALECT = "narrative-dynamics/runtime-schema-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALUE_TYPES = {
    "any",
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}
_EVENT_KEYS = {"event_kinds", "event_data", "allow_unlisted_events"}


class ModelSchemaDefinitionError(ValueError):
    """A declared schema cannot be interpreted by the runtime dialect."""

    def __init__(
        self,
        message: str,
        *,
        path: str = "$",
        schema_name: str | None = None,
    ) -> None:
        self.path = path
        self.schema_name = schema_name
        prefix = "model schema"
        if schema_name is not None:
            prefix += f" {schema_name!r}"
        super().__init__(f"{prefix} definition error at {path}: {message}")


class ModelSchemaViolation(ValueError):
    """A runtime value violates one declared model boundary schema."""

    def __init__(
        self,
        message: str,
        *,
        boundary: str,
        path: str,
        schema_name: str,
        execution: ExecutionCapture | None = None,
    ) -> None:
        self.boundary = boundary
        self.path = path
        self.schema_name = schema_name
        self.execution = execution
        super().__init__(
            f"{boundary} schema {schema_name!r} violation at {path}: {message}"
        )

    def attach_execution(
        self,
        execution: ExecutionCapture | None,
    ) -> "ModelSchemaViolation":
        if execution is not None:
            self.execution = execution
        return self


def _definition_error(
    message: str,
    *,
    path: str,
    schema_name: str,
) -> None:
    raise ModelSchemaDefinitionError(
        message,
        path=path,
        schema_name=schema_name,
    )


def _path_key(path: str, key: str) -> str:
    if _IDENTIFIER.fullmatch(key) is not None:
        return f"{path}.{key}"
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"{path}['{escaped}']"


def _sequence_definition(
    value: object,
    *,
    path: str,
    schema_name: str,
) -> tuple[object, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        _definition_error(
            "must be an array",
            path=path,
            schema_name=schema_name,
        )
    return tuple(value)


def _mapping_definition(
    value: object,
    *,
    path: str,
    schema_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _definition_error(
            "must be an object",
            path=path,
            schema_name=schema_name,
        )
    for key in value:
        if not isinstance(key, str) or not key:
            _definition_error(
                "object keys must be non-empty strings",
                path=path,
                schema_name=schema_name,
            )
    return value


def _nonnegative_integer_definition(
    value: object,
    *,
    path: str,
    schema_name: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _definition_error(
            "must be a non-negative integer",
            path=path,
            schema_name=schema_name,
        )
    return value


def _finite_number_definition(
    value: object,
    *,
    path: str,
    schema_name: str,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _definition_error(
            "must be a finite number",
            path=path,
            schema_name=schema_name,
        )
    return float(value)


def _validate_enum_definition(
    value: object,
    *,
    path: str,
    schema_name: str,
) -> None:
    choices = _sequence_definition(
        value,
        path=path,
        schema_name=schema_name,
    )
    if not choices:
        _definition_error(
            "must contain at least one value",
            path=path,
            schema_name=schema_name,
        )
    hashes: list[str] = []
    for index, choice in enumerate(choices):
        try:
            hashes.append(stable_content_hash(choice))
        except (TypeError, ValueError) as error:
            _definition_error(
                f"contains a non-canonical value: {error}",
                path=f"{path}[{index}]",
                schema_name=schema_name,
            )
    if len(set(hashes)) != len(hashes):
        _definition_error(
            "cannot contain duplicate typed values",
            path=path,
            schema_name=schema_name,
        )


def _validate_unknown_definition_keys(
    definition: Mapping[str, object],
    *,
    allowed: set[str],
    path: str,
    schema_name: str,
) -> None:
    unknown = sorted(set(definition) - allowed)
    if unknown:
        _definition_error(
            f"unsupported keyword {unknown[0]!r}",
            path=_path_key(path, unknown[0]),
            schema_name=schema_name,
        )


def _validate_value_definition(
    definition: Mapping[str, object],
    *,
    path: str,
    schema_name: str,
) -> None:
    raw_type = definition.get("type")
    if not isinstance(raw_type, str) or raw_type not in _VALUE_TYPES:
        _definition_error(
            f"type must be one of {tuple(sorted(_VALUE_TYPES))}",
            path=_path_key(path, "type"),
            schema_name=schema_name,
        )
    value_type = raw_type

    base = {"type", "enum"}
    type_keys = {
        "object": {"properties", "required", "additional_properties"},
        "array": {"items", "min_items", "max_items"},
        "number": {"minimum", "maximum"},
        "integer": {"minimum", "maximum"},
        "string": {"min_length", "max_length"},
        "any": set(),
        "boolean": set(),
        "null": set(),
    }[value_type]
    _validate_unknown_definition_keys(
        definition,
        allowed=base | type_keys,
        path=path,
        schema_name=schema_name,
    )

    if "enum" in definition:
        _validate_enum_definition(
            definition["enum"],
            path=_path_key(path, "enum"),
            schema_name=schema_name,
        )

    if value_type == "object":
        properties = _mapping_definition(
            definition.get("properties", {}),
            path=_path_key(path, "properties"),
            schema_name=schema_name,
        )
        properties_path = _path_key(path, "properties")
        for name in sorted(properties):
            child_path = _path_key(properties_path, name)
            child = _mapping_definition(
                properties[name],
                path=child_path,
                schema_name=schema_name,
            )
            _validate_value_definition(
                child,
                path=child_path,
                schema_name=schema_name,
            )

        required = _sequence_definition(
            definition.get("required", ()),
            path=_path_key(path, "required"),
            schema_name=schema_name,
        )
        required_names: list[str] = []
        for index, name in enumerate(required):
            if not isinstance(name, str) or not name:
                _definition_error(
                    "entries must be non-empty strings",
                    path=f"{_path_key(path, 'required')}[{index}]",
                    schema_name=schema_name,
                )
            required_names.append(name)
        if len(set(required_names)) != len(required_names):
            _definition_error(
                "cannot contain duplicate property names",
                path=_path_key(path, "required"),
                schema_name=schema_name,
            )
        undeclared = sorted(set(required_names) - set(properties))
        if undeclared:
            _definition_error(
                f"required property {undeclared[0]!r} is not declared in properties",
                path=_path_key(path, "required"),
                schema_name=schema_name,
            )

        additional = definition.get("additional_properties", True)
        if not isinstance(additional, bool):
            _definition_error(
                "must be boolean",
                path=_path_key(path, "additional_properties"),
                schema_name=schema_name,
            )

    elif value_type == "array":
        if "items" in definition:
            items = _mapping_definition(
                definition["items"],
                path=_path_key(path, "items"),
                schema_name=schema_name,
            )
            _validate_value_definition(
                items,
                path=_path_key(path, "items"),
                schema_name=schema_name,
            )
        minimum = _nonnegative_integer_definition(
            definition.get("min_items", 0),
            path=_path_key(path, "min_items"),
            schema_name=schema_name,
        )
        maximum = None
        if "max_items" in definition:
            maximum = _nonnegative_integer_definition(
                definition["max_items"],
                path=_path_key(path, "max_items"),
                schema_name=schema_name,
            )
        if maximum is not None and maximum < minimum:
            _definition_error(
                "must be greater than or equal to min_items",
                path=_path_key(path, "max_items"),
                schema_name=schema_name,
            )

    elif value_type in {"number", "integer"}:
        minimum = None
        maximum = None
        if "minimum" in definition:
            minimum = _finite_number_definition(
                definition["minimum"],
                path=_path_key(path, "minimum"),
                schema_name=schema_name,
            )
        if "maximum" in definition:
            maximum = _finite_number_definition(
                definition["maximum"],
                path=_path_key(path, "maximum"),
                schema_name=schema_name,
            )
        if minimum is not None and maximum is not None and maximum < minimum:
            _definition_error(
                "must be greater than or equal to minimum",
                path=_path_key(path, "maximum"),
                schema_name=schema_name,
            )

    elif value_type == "string":
        minimum = _nonnegative_integer_definition(
            definition.get("min_length", 0),
            path=_path_key(path, "min_length"),
            schema_name=schema_name,
        )
        maximum = None
        if "max_length" in definition:
            maximum = _nonnegative_integer_definition(
                definition["max_length"],
                path=_path_key(path, "max_length"),
                schema_name=schema_name,
            )
        if maximum is not None and maximum < minimum:
            _definition_error(
                "must be greater than or equal to min_length",
                path=_path_key(path, "max_length"),
                schema_name=schema_name,
            )


def _validate_event_definition(
    definition: Mapping[str, object],
    *,
    path: str,
    schema_name: str,
) -> None:
    _validate_unknown_definition_keys(
        definition,
        allowed=_EVENT_KEYS,
        path=path,
        schema_name=schema_name,
    )
    if "event_kinds" not in definition:
        _definition_error(
            "event schemas require event_kinds",
            path=_path_key(path, "event_kinds"),
            schema_name=schema_name,
        )
    kinds = _sequence_definition(
        definition["event_kinds"],
        path=_path_key(path, "event_kinds"),
        schema_name=schema_name,
    )
    kind_names: list[str] = []
    for index, kind in enumerate(kinds):
        if not isinstance(kind, str) or not kind:
            _definition_error(
                "entries must be non-empty strings",
                path=f"{_path_key(path, 'event_kinds')}[{index}]",
                schema_name=schema_name,
            )
        kind_names.append(kind)
    if len(set(kind_names)) != len(kind_names):
        _definition_error(
            "cannot contain duplicate event kinds",
            path=_path_key(path, "event_kinds"),
            schema_name=schema_name,
        )

    event_data = _mapping_definition(
        definition.get("event_data", {}),
        path=_path_key(path, "event_data"),
        schema_name=schema_name,
    )
    unknown_data = sorted(set(event_data) - set(kind_names))
    if unknown_data:
        _definition_error(
            f"event data schema {unknown_data[0]!r} is not listed in event_kinds",
            path=_path_key(_path_key(path, "event_data"), unknown_data[0]),
            schema_name=schema_name,
        )
    for kind in sorted(event_data):
        child_path = _path_key(_path_key(path, "event_data"), kind)
        child = _mapping_definition(
            event_data[kind],
            path=child_path,
            schema_name=schema_name,
        )
        _validate_value_definition(
            child,
            path=child_path,
            schema_name=schema_name,
        )

    allow_unlisted = definition.get("allow_unlisted_events", False)
    if not isinstance(allow_unlisted, bool):
        _definition_error(
            "must be boolean",
            path=_path_key(path, "allow_unlisted_events"),
            schema_name=schema_name,
        )


def _is_event_definition(definition: Mapping[str, object]) -> bool:
    return bool(set(definition) & _EVENT_KEYS)


def validate_schema_definition(
    definition: Mapping[str, object],
    *,
    schema_name: str,
    dialect: str = RUNTIME_SCHEMA_DIALECT,
) -> None:
    """Validate one immutable declaration against the runtime schema dialect."""

    if dialect != RUNTIME_SCHEMA_DIALECT:
        raise ModelSchemaDefinitionError(
            f"unsupported schema dialect {dialect!r}",
            path="$",
            schema_name=schema_name,
        )
    mapping = _mapping_definition(
        definition,
        path="$",
        schema_name=schema_name,
    )
    if _is_event_definition(mapping):
        _validate_event_definition(mapping, path="$", schema_name=schema_name)
    else:
        _validate_value_definition(mapping, path="$", schema_name=schema_name)


def _raise_violation(
    message: str,
    *,
    boundary: str,
    path: str,
    schema_name: str,
) -> None:
    raise ModelSchemaViolation(
        message,
        boundary=boundary,
        path=path,
        schema_name=schema_name,
    )


def _matches_type(value: object, value_type: str) -> bool:
    if value_type == "any":
        return True
    if value_type == "null":
        return value is None
    if value_type == "boolean":
        return isinstance(value, bool)
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if value_type == "string":
        return isinstance(value, str)
    if value_type == "array":
        return isinstance(value, (list, tuple))
    if value_type == "object":
        return isinstance(value, Mapping)
    return False


def _validate_value(
    definition: Mapping[str, object],
    value: object,
    *,
    boundary: str,
    path: str,
    schema_name: str,
) -> None:
    value_type = str(definition["type"])
    if not _matches_type(value, value_type):
        _raise_violation(
            f"expected {value_type}, got {type(value).__name__}",
            boundary=boundary,
            path=path,
            schema_name=schema_name,
        )

    if "enum" in definition:
        value_hash = stable_content_hash(value)
        allowed_hashes = {
            stable_content_hash(choice) for choice in definition["enum"]
        }
        if value_hash not in allowed_hashes:
            _raise_violation(
                "value is not one of the declared enum choices",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )

    if value_type == "object":
        assert isinstance(value, Mapping)
        properties = definition.get("properties", {})
        assert isinstance(properties, Mapping)
        for key in value:
            if not isinstance(key, str) or not key:
                _raise_violation(
                    "object keys must be non-empty strings",
                    boundary=boundary,
                    path=path,
                    schema_name=schema_name,
                )
        required = tuple(definition.get("required", ()))
        for name in required:
            assert isinstance(name, str)
            if name not in value:
                _raise_violation(
                    "required property is missing",
                    boundary=boundary,
                    path=_path_key(path, name),
                    schema_name=schema_name,
                )
        if not bool(definition.get("additional_properties", True)):
            unknown = sorted(set(value) - set(properties))
            if unknown:
                _raise_violation(
                    "property is not declared by this schema",
                    boundary=boundary,
                    path=_path_key(path, unknown[0]),
                    schema_name=schema_name,
                )
        for name in sorted(set(value) & set(properties)):
            child = properties[name]
            assert isinstance(child, Mapping)
            _validate_value(
                child,
                value[name],
                boundary=boundary,
                path=_path_key(path, name),
                schema_name=schema_name,
            )

    elif value_type == "array":
        assert isinstance(value, (list, tuple))
        minimum = int(definition.get("min_items", 0))
        maximum = definition.get("max_items")
        if len(value) < minimum:
            _raise_violation(
                f"array has fewer than {minimum} items",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )
        if maximum is not None and len(value) > int(maximum):
            _raise_violation(
                f"array has more than {int(maximum)} items",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )
        items = definition.get("items")
        if items is not None:
            assert isinstance(items, Mapping)
            for index, item in enumerate(value):
                _validate_value(
                    items,
                    item,
                    boundary=boundary,
                    path=f"{path}[{index}]",
                    schema_name=schema_name,
                )

    elif value_type in {"number", "integer"}:
        numeric = float(value)
        minimum = definition.get("minimum")
        maximum = definition.get("maximum")
        if minimum is not None and numeric < float(minimum):
            _raise_violation(
                f"number is less than minimum {float(minimum)!r}",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )
        if maximum is not None and numeric > float(maximum):
            _raise_violation(
                f"number is greater than maximum {float(maximum)!r}",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )

    elif value_type == "string":
        assert isinstance(value, str)
        minimum = int(definition.get("min_length", 0))
        maximum = definition.get("max_length")
        if len(value) < minimum:
            _raise_violation(
                f"string is shorter than {minimum} characters",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )
        if maximum is not None and len(value) > int(maximum):
            _raise_violation(
                f"string is longer than {int(maximum)} characters",
                boundary=boundary,
                path=path,
                schema_name=schema_name,
            )


def _validate_events(
    definition: Mapping[str, object],
    value: object,
    *,
    boundary: str,
    schema_name: str,
) -> None:
    if not isinstance(value, (list, tuple)):
        _raise_violation(
            "expected an event sequence",
            boundary=boundary,
            path="$",
            schema_name=schema_name,
        )
    allowed = set(definition["event_kinds"])
    allow_unlisted = bool(definition.get("allow_unlisted_events", False))
    event_data = definition.get("event_data", {})
    assert isinstance(event_data, Mapping)
    for index, event in enumerate(value):
        if not isinstance(event, TraceEvent):
            _raise_violation(
                "event sequence contains a non-TraceEvent value",
                boundary=boundary,
                path=f"$[{index}]",
                schema_name=schema_name,
            )
        if event.kind not in allowed and not allow_unlisted:
            _raise_violation(
                f"event kind {event.kind!r} is not declared",
                boundary=boundary,
                path=f"$[{index}].kind",
                schema_name=schema_name,
            )
        data_schema = event_data.get(event.kind)
        if data_schema is not None:
            assert isinstance(data_schema, Mapping)
            _validate_value(
                data_schema,
                event.data,
                boundary=boundary,
                path=f"$[{index}].data",
                schema_name=schema_name,
            )


def validate_schema_value(
    schema: "ModelSchema",
    value: object,
    *,
    boundary: str,
) -> None:
    """Validate one runtime value and report an exact boundary/path on failure."""

    if not isinstance(boundary, str) or not boundary:
        raise ValueError("schema validation boundary must be a non-empty string")
    schema_name = getattr(schema, "name", None)
    dialect = getattr(schema, "dialect", None)
    definition = getattr(schema, "definition", None)
    specified = getattr(schema, "specified", None)
    if not isinstance(schema_name, str) or not schema_name:
        raise TypeError("runtime schema must expose a non-empty name")
    if not isinstance(specified, bool):
        raise TypeError("runtime schema must expose a boolean specified flag")
    if not specified:
        return
    if not isinstance(definition, Mapping):
        raise TypeError("runtime schema definition must be a mapping")
    validate_schema_definition(
        definition,
        schema_name=schema_name,
        dialect=dialect,
    )
    if _is_event_definition(definition):
        _validate_events(
            definition,
            value,
            boundary=boundary,
            schema_name=schema_name,
        )
    else:
        _validate_value(
            definition,
            value,
            boundary=boundary,
            path="$",
            schema_name=schema_name,
        )


def _contract_from_source(source: object):
    contract = getattr(source, "contract", None)
    if contract is None:
        return None
    for attribute in (
        "parameter_schema",
        "scenario_schema",
        "event_schema",
        "content_hash",
    ):
        if not hasattr(contract, attribute):
            raise TypeError(f"model contract is missing {attribute}")
    schemas = (
        contract.parameter_schema,
        contract.scenario_schema,
        contract.event_schema,
    )
    if not any(bool(getattr(schema, "specified", False)) for schema in schemas):
        return None
    return contract


def _schema_attestation(schema: object, *, status: str) -> dict[str, object]:
    entry: dict[str, object] = {"status": status}
    if bool(getattr(schema, "specified", False)):
        entry.update(
            {
                "schema_name": getattr(schema, "name"),
                "schema_version": getattr(schema, "version"),
                "schema_hash": getattr(schema, "content_hash"),
                "dialect": getattr(schema, "dialect"),
            }
        )
    return entry


def _scenario_payload(
    scenario: object,
    *,
    schema_name: str,
) -> Mapping[str, object]:
    if isinstance(scenario, Scenario):
        return scenario.payload
    schema_payload = getattr(scenario, "schema_payload", None)
    if callable(schema_payload):
        payload = schema_payload()
    else:
        payload = getattr(scenario, "payload", None)
    if not isinstance(payload, Mapping):
        raise ModelSchemaViolation(
            "scenario does not expose a mapping payload for validation",
            boundary="scenario",
            path="$",
            schema_name=schema_name,
        )
    return payload


def validate_contract_inputs(
    source: object,
    scenario: object,
    parameters: Mapping[str, float],
) -> dict[str, object] | None:
    """Validate trusted inputs before model construction or process launch."""

    contract = _contract_from_source(source)
    if contract is None:
        return None

    parameter_schema = contract.parameter_schema
    scenario_schema = contract.scenario_schema
    event_schema = contract.event_schema

    if parameter_schema.specified:
        validate_schema_value(
            parameter_schema,
            parameters,
            boundary="parameters",
        )
    if scenario_schema.specified:
        validate_schema_value(
            scenario_schema,
            _scenario_payload(scenario, schema_name=scenario_schema.name),
            boundary="scenario",
        )

    return {
        "dialect": RUNTIME_SCHEMA_DIALECT,
        "contract_hash": contract.content_hash,
        "parameters": _schema_attestation(
            parameter_schema,
            status="validated" if parameter_schema.specified else "unspecified",
        ),
        "scenario": _schema_attestation(
            scenario_schema,
            status="validated" if scenario_schema.specified else "unspecified",
        ),
        "events": _schema_attestation(
            event_schema,
            status="pending" if event_schema.specified else "unspecified",
        ),
    }


def validate_contract_events(
    source: object,
    events: tuple[TraceEvent, ...],
    input_attestation: Mapping[str, object] | None,
    *,
    execution: ExecutionCapture | None,
) -> dict[str, object] | None:
    """Validate untrusted events before constructing the trusted trace."""

    contract = _contract_from_source(source)
    if contract is None:
        return None
    event_schema = contract.event_schema
    if event_schema.specified:
        try:
            validate_schema_value(
                event_schema,
                events,
                boundary="events",
            )
        except ModelSchemaViolation as error:
            raise error.attach_execution(execution)

    attestation: dict[str, object] = {}
    if input_attestation is not None:
        for key, value in input_attestation.items():
            attestation[key] = dict(value) if isinstance(value, Mapping) else value
    else:
        attestation = {
            "dialect": RUNTIME_SCHEMA_DIALECT,
            "contract_hash": contract.content_hash,
        }
    attestation["events"] = _schema_attestation(
        event_schema,
        status="validated" if event_schema.specified else "unspecified",
    )
    return attestation


__all__ = [
    "ModelSchemaDefinitionError",
    "ModelSchemaViolation",
    "RUNTIME_SCHEMA_DIALECT",
    "validate_contract_events",
    "validate_contract_inputs",
    "validate_schema_definition",
    "validate_schema_value",
]
