from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from narrative_dynamics.abm.situated_realization import (
    build_narrative_realization_prompt,
    compile_narrative_realization,
)
from narrative_dynamics.abm.situated_realization_contracts import (
    NarrativeRealizationProviderIdentity,
)
from narrative_dynamics.integrations import OpenAINarrativeProvider
from tests.test_network_abm_situated_realization import (
    policy,
    request,
    single_scene_projection,
)


class SDKMessage:
    def __init__(self, content):
        self.content = content


class SDKChoice:
    def __init__(self, content):
        self.message = SDKMessage(content)


class SDKResponse:
    def __init__(self, content):
        self.choices = [SDKChoice(content)]


class SDKCompletions:
    def __init__(self, content):
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SDKResponse(self.content)


class SDKChat:
    def __init__(self, content):
        self.completions = SDKCompletions(content)


class InjectedSDKClient:
    def __init__(self, content):
        self.chat = SDKChat(content)


@pytest.mark.parametrize(
    ("environment", "missing_name", "sensitive_value"),
    (
        ({"OPENAI_MODEL": "private-model-name"}, "OPENAI_API_KEY", "private-model-name"),
        ({"OPENAI_API_KEY": "sk-private-value"}, "OPENAI_MODEL", "sk-private-value"),
    ),
)
def test_from_env_rejects_missing_required_configuration_without_echoing_values(
    monkeypatch, environment, missing_name, sensitive_value
):
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_PROVIDER",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError) as caught:
        OpenAINarrativeProvider.from_env(client=InjectedSDKClient("{}"))

    assert missing_name in str(caught.value)
    assert sensitive_value not in str(caught.value)


def test_explicit_dotenv_path_and_secret_are_configuration_only(
    tmp_path, monkeypatch
):
    secret = "sk-artifact-must-not-contain-this"
    base_url = "https://private-compatible-endpoint.invalid/v1"
    env_file = tmp_path / "private-provider-configuration.env"
    env_file.write_text(
        "OPENAI_API_KEY=" + secret + "\n"
        "OPENAI_MODEL=gpt-test-model\n"
        "OPENAI_PROVIDER=private-openai-gateway\n"
        "OPENAI_BASE_URL=" + base_url + "\n",
        encoding="utf-8",
    )
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_PROVIDER",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    projection = single_scene_projection()
    response = {
        "passages": [
            {
                "beat_ids": list(projection.scenes[0].beat_ids),
                "text": "A bounded scene.",
            }
        ]
    }
    client = InjectedSDKClient(json.dumps(response))

    provider = OpenAINarrativeProvider.from_env(env_file, client=client)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    artifact = compile_narrative_realization(prompt, provider)
    serialized = json.dumps(artifact.to_dict(), sort_keys=True)

    assert provider.identity == NarrativeRealizationProviderIdentity(
        "private-openai-gateway", "2", "gpt-test-model"
    )
    for value in (secret, base_url, str(env_file), env_file.name):
        assert value not in repr(provider)
        assert value not in json.dumps(provider.identity.to_dict(), sort_keys=True)
        assert value not in serialized


def test_complete_json_sends_exact_task_and_payload_and_decodes_one_object():
    decoded = {"passages": [{"beat_ids": ["beat-1"], "text": "Text."}]}
    client = InjectedSDKClient(json.dumps(decoded))
    provider = OpenAINarrativeProvider(
        api_key="sk-never-serialize",
        model="gpt-test-model",
        provider_id="openai-test",
        client=client,
    )
    payload = {"scene": {"scene_id": "scene-1"}, "limits": {"maximum": 2}}

    actual = provider.complete_json(task="task-1", payload=payload)

    assert actual == decoded
    sdk_request = client.chat.completions.request
    assert sdk_request["model"] == "gpt-test-model"
    assert sdk_request["response_format"] == {"type": "json_object"}
    assert sdk_request["messages"] == [
        {
            "role": "system",
            "content": (
                "Return exactly one JSON object matching the supplied response schema."
            ),
        },
        {
            "role": "user",
            "content": (
                '{"payload":{"limits":{"maximum":2},"scene":'
                '{"scene_id":"scene-1"}},"task":"task-1"}'
            ),
        },
    ]
    assert json.loads(sdk_request["messages"][1]["content"]) == {
        "task": "task-1",
        "payload": payload,
    }
    assert provider.identity.version == "2"
    assert "sk-never-serialize" not in json.dumps(sdk_request, sort_keys=True)


def test_from_env_forwards_base_url_only_to_lazy_sdk_construction(
    monkeypatch,
):
    secret = "sk-private-construction"
    base_url = "https://private-compatible-endpoint.invalid/v1"
    captured = {}

    def openai_factory(**kwargs):
        captured.update(kwargs)
        return InjectedSDKClient("{}")

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=openai_factory),
    )
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    monkeypatch.setenv("OPENAI_PROVIDER", "private-openai-gateway")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    provider = OpenAINarrativeProvider.from_env()

    assert captured == {"api_key": secret, "base_url": base_url}
    assert provider.identity == NarrativeRealizationProviderIdentity(
        "private-openai-gateway", "2", "gpt-test-model"
    )
    assert base_url not in repr(provider)
    assert base_url not in json.dumps(provider.identity.to_dict(), sort_keys=True)


def test_base_url_is_removed_from_lazy_sdk_construction_errors(monkeypatch):
    base_url = "https://private-failing-endpoint.invalid/v1"
    captured = {}

    def failing_openai_factory(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("failed at " + kwargs["base_url"])

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=failing_openai_factory),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-private-construction")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    with pytest.raises(RuntimeError) as caught:
        OpenAINarrativeProvider.from_env()

    assert captured["base_url"] == base_url
    assert base_url not in str(caught.value)


@pytest.mark.parametrize("content", ("[]", '"text"', "null", "not-json"))
def test_complete_json_rejects_malformed_or_non_object_json(content):
    provider = OpenAINarrativeProvider(
        api_key="sk-private-malformed",
        model="gpt-test-model",
        client=InjectedSDKClient(content),
    )

    with pytest.raises(ValueError) as caught:
        provider.complete_json(task="task-1", payload={"value": 1})

    assert "sk-private-malformed" not in str(caught.value)
    assert content not in str(caught.value)


def test_importing_abm_does_not_import_openai_sdk():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import narrative_dynamics.abm; "
            "assert 'openai' not in sys.modules",
        ],
        cwd=Path(__file__).parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
