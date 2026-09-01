"""Optional OpenAI adapter for entitlement-bound narrative realization."""

from __future__ import annotations

import json
import os
from os import PathLike

from narrative_dynamics.abm.situated_realization_contracts import (
    NarrativeRealizationProviderIdentity,
)


class OpenAINarrativeProvider:
    """Decode OpenAI JSON-object responses through the core provider protocol."""

    __slots__ = ("identity", "_client")

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        provider_id: str = "openai",
        client: object | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OpenAI narrative provider requires OPENAI_API_KEY")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI narrative provider requires OPENAI_MODEL")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("OpenAI narrative provider requires OPENAI_PROVIDER")
        self.identity = NarrativeRealizationProviderIdentity(
            provider_id,
            "1",
            model,
        )
        self._client = client if client is not None else self._new_client(api_key)

    @staticmethod
    def _new_client(api_key: str) -> object:
        try:
            from openai import OpenAI

            return OpenAI(api_key=api_key)
        except Exception:
            raise RuntimeError(
                "OpenAI narrative provider client construction failed"
            ) from None

    @classmethod
    def from_env(
        cls,
        env_file: str | PathLike[str] | None = None,
        *,
        client: object | None = None,
    ) -> OpenAINarrativeProvider:
        """Construct from environment, optionally loading one explicit dotenv file."""

        if env_file is not None:
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=env_file)
            except Exception:
                raise ValueError(
                    "OpenAI narrative provider configuration could not be loaded"
                ) from None

        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL")
        provider_id = os.environ.get("OPENAI_PROVIDER", "openai")
        if not api_key:
            raise ValueError("OpenAI narrative provider requires OPENAI_API_KEY")
        if not model:
            raise ValueError("OpenAI narrative provider requires OPENAI_MODEL")
        return cls(
            api_key=api_key,
            model=model,
            provider_id=provider_id,
            client=client,
        )

    def complete_json(self, *, task: str, payload: dict[str, object]) -> object:
        """Send one task packet and decode exactly one JSON object."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError("OpenAI narrative provider task must be non-empty")
        if not isinstance(payload, dict):
            raise TypeError("OpenAI narrative provider payload must be an object")
        request_json = json.dumps(
            {"task": task, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            response = self._client.chat.completions.create(
                model=self.identity.model_name,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": request_json}],
            )
            content = response.choices[0].message.content
        except Exception:
            raise RuntimeError("OpenAI narrative provider request failed") from None
        if not isinstance(content, str):
            raise ValueError("OpenAI narrative provider returned malformed JSON")
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError):
            raise ValueError("OpenAI narrative provider returned malformed JSON") from None
        if not isinstance(decoded, dict):
            raise ValueError("OpenAI narrative provider must return one JSON object")
        return decoded

    def __repr__(self) -> str:
        return (
            "OpenAINarrativeProvider("
            f"provider_id={self.identity.provider_id!r}, "
            f"version={self.identity.version!r}, "
            f"model_name={self.identity.model_name!r})"
        )


__all__ = ("OpenAINarrativeProvider",)
