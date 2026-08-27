from __future__ import annotations

from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.observation_projection import (
    ObservationProjectionError,
    ObservationProjectionModelSpec,
    ObservationProjectionResult,
    project_world_observations,
)
from narrative_dynamics.narrative.world import (
    WorldStepResult,
    world_state_from_story,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELATIONS = frozenset({"equals", "clear"})


class RuntimePerceptAdmissionError(ValueError):
    """A runtime world percept could not be admitted safely."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _step(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _cutoff(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _step(value, label=label)


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _validate_relation_value(
    relation: object,
    value: object,
    *,
    label: str,
) -> tuple[str, TypedValue | None]:
    relation_text = _text(relation, label=f"{label} relation")
    if relation_text not in _RELATIONS:
        raise ValueError(f"{label} relation must be equals or clear")
    if relation_text == "equals":
        if not isinstance(value, TypedValue):
            raise TypeError(f"{label} equals relation requires a TypedValue")
        return relation_text, value
    if value is not None:
        raise ValueError(f"{label} clear relation cannot contain a value")
    return relation_text, None


@dataclass(frozen=True)
class RuntimePerceptView:
    observer_id: str
    channel: str
    cell: StateCellRef
    relation: str
    value: TypedValue | None
    step_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observer_id",
            _text(self.observer_id, label="runtime percept observer id"),
        )
        object.__setattr__(
            self,
            "channel",
            _text(self.channel, label="runtime percept channel"),
        )
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("runtime percept cell must be a StateCellRef")
        relation, value = _validate_relation_value(
            self.relation,
            self.value,
            label="runtime percept",
        )
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="runtime percept step index"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_id": self.observer_id,
            "channel": self.channel,
            "cell": self.cell.to_dict(),
            "relation": self.relation,
            "value": None if self.value is None else self.value.to_dict(),
            "step_index": self.step_index,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeEpistemicEvidence:
    observer_id: str
    channel: str
    cell: StateCellRef
    relation: str
    value: TypedValue | None
    step_index: int
    projected_observation_hash: str
    projection_result_hash: str
    projection_model_hash: str
    source_world_state_hash: str
    source_world_step_hash: str
    source_transition_hashes: tuple[str, ...]
    projection_spec_hash: str
    projection_sample_hash: str | None = None

    def __post_init__(self) -> None:
        view = RuntimePerceptView(
            self.observer_id,
            self.channel,
            self.cell,
            self.relation,
            self.value,
            self.step_index,
        )
        object.__setattr__(self, "observer_id", view.observer_id)
        object.__setattr__(self, "channel", view.channel)
        object.__setattr__(self, "cell", view.cell)
        object.__setattr__(self, "relation", view.relation)
        object.__setattr__(self, "value", view.value)
        object.__setattr__(self, "step_index", view.step_index)
        object.__setattr__(
            self,
            "projected_observation_hash",
            _hash(
                self.projected_observation_hash,
                label="runtime evidence projected observation hash",
            ),
        )
        object.__setattr__(
            self,
            "projection_result_hash",
            _hash(
                self.projection_result_hash,
                label="runtime evidence projection result hash",
            ),
        )
        object.__setattr__(
            self,
            "projection_model_hash",
            _hash(
                self.projection_model_hash,
                label="runtime evidence projection model hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_state_hash",
            _hash(
                self.source_world_state_hash,
                label="runtime evidence source world state hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_step_hash",
            _hash(
                self.source_world_step_hash,
                label="runtime evidence source world step hash",
            ),
        )
        if not isinstance(self.source_transition_hashes, tuple):
            raise TypeError("runtime evidence source transition hashes must be a tuple")
        hashes = tuple(
            _hash(item, label="runtime evidence source transition hash")
            for item in self.source_transition_hashes
        )
        if len(set(hashes)) != len(hashes):
            raise ValueError("runtime evidence source transition hashes must be unique")
        object.__setattr__(
            self,
            "source_transition_hashes",
            tuple(sorted(hashes)),
        )
        object.__setattr__(
            self,
            "projection_spec_hash",
            _hash(
                self.projection_spec_hash,
                label="runtime evidence projection spec hash",
            ),
        )
        if self.projection_sample_hash is not None:
            object.__setattr__(
                self,
                "projection_sample_hash",
                _hash(
                    self.projection_sample_hash,
                    label="runtime evidence projection sample hash",
                ),
            )

    @property
    def percept_view(self) -> RuntimePerceptView:
        return RuntimePerceptView(
            self.observer_id,
            self.channel,
            self.cell,
            self.relation,
            self.value,
            self.step_index,
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "observer_id": self.observer_id,
            "channel": self.channel,
            "cell": self.cell.to_dict(),
            "relation": self.relation,
            "value": None if self.value is None else self.value.to_dict(),
            "step_index": self.step_index,
            "projected_observation_hash": self.projected_observation_hash,
            "projection_result_hash": self.projection_result_hash,
            "projection_model_hash": self.projection_model_hash,
            "source_world_state_hash": self.source_world_state_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "source_transition_hashes": list(self.source_transition_hashes),
            "projection_spec_hash": self.projection_spec_hash,
        }
        if self.projection_sample_hash is not None:
            payload["projection_sample_hash"] = self.projection_sample_hash
        return payload

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _evidence_key(
    item: RuntimeEpistemicEvidence,
) -> tuple[str, str, tuple[str, str, str]]:
    return (item.observer_id, item.channel, _cell_key(item.cell))


@dataclass(frozen=True)
class RuntimeEvidenceBatch:
    prior_ledger_hash: str
    step_index: int
    source_prior_world_state_hash: str
    source_world_state_hash: str
    source_world_step_hash: str
    projection_model_hash: str
    projection_result_hash: str
    evidence: tuple[RuntimeEpistemicEvidence, ...]
    projection_sample_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_ledger_hash",
            _hash(self.prior_ledger_hash, label="runtime batch prior ledger hash"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="runtime batch step index"),
        )
        object.__setattr__(
            self,
            "source_prior_world_state_hash",
            _hash(
                self.source_prior_world_state_hash,
                label="runtime batch source prior world state hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_state_hash",
            _hash(
                self.source_world_state_hash,
                label="runtime batch source world state hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_step_hash",
            _hash(
                self.source_world_step_hash,
                label="runtime batch source world step hash",
            ),
        )
        object.__setattr__(
            self,
            "projection_model_hash",
            _hash(
                self.projection_model_hash,
                label="runtime batch projection model hash",
            ),
        )
        object.__setattr__(
            self,
            "projection_result_hash",
            _hash(
                self.projection_result_hash,
                label="runtime batch projection result hash",
            ),
        )
        if not isinstance(self.evidence, tuple):
            raise TypeError("runtime batch evidence must be a tuple")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, RuntimeEpistemicEvidence) for item in evidence):
            raise TypeError(
                "runtime batch evidence must contain RuntimeEpistemicEvidence values"
            )
        evidence = tuple(sorted(evidence, key=_evidence_key))
        keys = tuple(_evidence_key(item) for item in evidence)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "runtime batch cannot contain duplicate observer/channel/cell evidence"
            )
        if not isinstance(self.projection_sample_hashes, tuple):
            raise TypeError("runtime batch projection sample hashes must be a tuple")
        sample_hashes = tuple(
            _hash(item, label="runtime batch projection sample hash")
            for item in self.projection_sample_hashes
        )
        if len(set(sample_hashes)) != len(sample_hashes):
            raise ValueError("runtime batch projection sample hashes must be unique")
        sample_hashes = tuple(sorted(sample_hashes))
        sample_hash_set = set(sample_hashes)
        for item in evidence:
            if item.step_index != self.step_index:
                raise ValueError("runtime evidence must bind batch step index")
            if item.source_world_state_hash != self.source_world_state_hash:
                raise ValueError("runtime evidence must bind batch source world state")
            if item.source_world_step_hash != self.source_world_step_hash:
                raise ValueError("runtime evidence must bind batch source world step")
            if item.projection_model_hash != self.projection_model_hash:
                raise ValueError("runtime evidence must bind batch projection model")
            if item.projection_result_hash != self.projection_result_hash:
                raise ValueError("runtime evidence must bind batch projection result")
            if (
                item.projection_sample_hash is not None
                and item.projection_sample_hash not in sample_hash_set
            ):
                raise ValueError(
                    "runtime evidence projection sample must be included in batch"
                )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "projection_sample_hashes", sample_hashes)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "prior_ledger_hash": self.prior_ledger_hash,
            "step_index": self.step_index,
            "source_prior_world_state_hash": self.source_prior_world_state_hash,
            "source_world_state_hash": self.source_world_state_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "projection_model_hash": self.projection_model_hash,
            "projection_result_hash": self.projection_result_hash,
            "evidence": [item.to_dict() for item in self.evidence],
        }
        if self.projection_sample_hashes:
            payload["projection_sample_hashes"] = list(
                self.projection_sample_hashes
            )
        return payload

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _ledger_payload(
    domain_id: str,
    domain_version: str,
    domain_spec_hash: str,
    source_story_hash: str,
    source_at_time: int | None,
    initial_world_state_hash: str,
    current_world_state_hash: str,
    batches: tuple[RuntimeEvidenceBatch, ...],
) -> dict[str, object]:
    return {
        "domain_id": domain_id,
        "domain_version": domain_version,
        "domain_spec_hash": domain_spec_hash,
        "source_story_hash": source_story_hash,
        "source_at_time": source_at_time,
        "initial_world_state_hash": initial_world_state_hash,
        "current_world_state_hash": current_world_state_hash,
        "batches": [item.to_dict() for item in batches],
    }


@dataclass(frozen=True)
class RuntimeEvidenceLedger:
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    source_story_hash: str
    source_at_time: int | None
    initial_world_state_hash: str
    current_world_state_hash: str
    batches: tuple[RuntimeEvidenceBatch, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "domain_id",
            _text(self.domain_id, label="runtime ledger domain id"),
        )
        object.__setattr__(
            self,
            "domain_version",
            _text(self.domain_version, label="runtime ledger domain version"),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _hash(self.domain_spec_hash, label="runtime ledger domain spec hash"),
        )
        object.__setattr__(
            self,
            "source_story_hash",
            _hash(self.source_story_hash, label="runtime ledger source story hash"),
        )
        object.__setattr__(
            self,
            "source_at_time",
            _cutoff(self.source_at_time, label="runtime ledger source cutoff"),
        )
        object.__setattr__(
            self,
            "initial_world_state_hash",
            _hash(
                self.initial_world_state_hash,
                label="runtime ledger initial world state hash",
            ),
        )
        object.__setattr__(
            self,
            "current_world_state_hash",
            _hash(
                self.current_world_state_hash,
                label="runtime ledger current world state hash",
            ),
        )
        if not isinstance(self.batches, tuple):
            raise TypeError("runtime ledger batches must be a tuple")
        batches = tuple(self.batches)
        if any(not isinstance(item, RuntimeEvidenceBatch) for item in batches):
            raise TypeError(
                "runtime ledger batches must contain RuntimeEvidenceBatch values"
            )

        previous_world_hash = self.initial_world_state_hash
        for index, batch in enumerate(batches):
            expected_step = index + 1
            if batch.step_index != expected_step:
                raise ValueError("runtime ledger batch steps must be consecutive from one")
            if batch.source_prior_world_state_hash != previous_world_hash:
                raise ValueError("runtime ledger world state hashes must form a chain")
            prefix = batches[:index]
            prefix_current = (
                self.initial_world_state_hash
                if index == 0
                else batches[index - 1].source_world_state_hash
            )
            expected_prior_hash = stable_content_hash(
                _ledger_payload(
                    self.domain_id,
                    self.domain_version,
                    self.domain_spec_hash,
                    self.source_story_hash,
                    self.source_at_time,
                    self.initial_world_state_hash,
                    prefix_current,
                    prefix,
                )
            )
            if batch.prior_ledger_hash != expected_prior_hash:
                raise ValueError(
                    "runtime batch must bind the exact prior ledger prefix hash"
                )
            previous_world_hash = batch.source_world_state_hash

        expected_current = (
            self.initial_world_state_hash
            if not batches
            else batches[-1].source_world_state_hash
        )
        if self.current_world_state_hash != expected_current:
            raise ValueError(
                "runtime ledger current world state must equal the chain tail"
            )
        object.__setattr__(self, "batches", batches)

    @property
    def current_step_index(self) -> int:
        return 0 if not self.batches else self.batches[-1].step_index

    def to_dict(self) -> dict[str, object]:
        return _ledger_payload(
            self.domain_id,
            self.domain_version,
            self.domain_spec_hash,
            self.source_story_hash,
            self.source_at_time,
            self.initial_world_state_hash,
            self.current_world_state_hash,
            self.batches,
        )

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimePerceptAdmissionResult:
    prior_ledger_hash: str
    projection_result: ObservationProjectionResult
    evidence_batch: RuntimeEvidenceBatch
    next_ledger: RuntimeEvidenceLedger

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_ledger_hash",
            _hash(
                self.prior_ledger_hash,
                label="runtime admission result prior ledger hash",
            ),
        )
        if not isinstance(self.projection_result, ObservationProjectionResult):
            raise TypeError(
                "runtime admission result projection_result must be ObservationProjectionResult"
            )
        if not isinstance(self.evidence_batch, RuntimeEvidenceBatch):
            raise TypeError(
                "runtime admission result evidence_batch must be RuntimeEvidenceBatch"
            )
        if not isinstance(self.next_ledger, RuntimeEvidenceLedger):
            raise TypeError(
                "runtime admission result next_ledger must be RuntimeEvidenceLedger"
            )
        batch = self.evidence_batch
        projection = self.projection_result
        ledger = self.next_ledger

        if batch.prior_ledger_hash != self.prior_ledger_hash:
            raise ValueError("runtime admission batch must bind exact prior ledger hash")
        if batch.projection_result_hash != projection.content_hash:
            raise ValueError("runtime admission batch must bind projection result hash")
        if batch.projection_model_hash != projection.model_hash:
            raise ValueError("runtime admission batch must bind projection model hash")
        if batch.source_world_step_hash != projection.source_world_step_hash:
            raise ValueError("runtime admission batch must bind projection world step")
        if batch.source_world_state_hash != projection.source_world_state_hash:
            raise ValueError("runtime admission batch must bind projection world state")
        if batch.step_index != projection.step_index:
            raise ValueError("runtime admission batch must bind projection step index")
        expected_sample_hashes = tuple(
            sorted(sample.content_hash for sample in projection.stochastic_samples)
        )
        if batch.projection_sample_hashes != expected_sample_hashes:
            raise ValueError(
                "runtime admission batch must bind every projection sample"
            )
        if not ledger.batches or ledger.batches[-1] != batch:
            raise ValueError("runtime admission next ledger must end with evidence batch")

        prefix_batches = ledger.batches[:-1]
        prefix_current = (
            ledger.initial_world_state_hash
            if not prefix_batches
            else prefix_batches[-1].source_world_state_hash
        )
        expected_prior_hash = stable_content_hash(
            _ledger_payload(
                ledger.domain_id,
                ledger.domain_version,
                ledger.domain_spec_hash,
                ledger.source_story_hash,
                ledger.source_at_time,
                ledger.initial_world_state_hash,
                prefix_current,
                prefix_batches,
            )
        )
        if expected_prior_hash != self.prior_ledger_hash:
            raise ValueError(
                "runtime admission result must bind exact prior ledger prefix"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_ledger_hash": self.prior_ledger_hash,
            "projection_result": self.projection_result.to_dict(),
            "evidence_batch": self.evidence_batch.to_dict(),
            "next_ledger": self.next_ledger.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def runtime_evidence_ledger_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
    seed: int | None = None,
) -> RuntimeEvidenceLedger:
    validate_narrative(story, domain)
    cutoff = _cutoff(at_time, label="runtime ledger source cutoff")
    initial = world_state_from_story(
        story,
        domain,
        at_time=cutoff,
        seed=seed,
    )
    return RuntimeEvidenceLedger(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        source_story_hash=story.content_hash,
        source_at_time=cutoff,
        initial_world_state_hash=initial.content_hash,
        current_world_state_hash=initial.content_hash,
        batches=(),
    )


def _validate_admission_source(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    projection_model: ObservationProjectionModelSpec,
    prior_ledger: RuntimeEvidenceLedger,
) -> None:
    if not isinstance(story, GenericNarrative):
        raise TypeError("runtime percept admission requires GenericNarrative")
    if not isinstance(domain, DomainSpec):
        raise TypeError("runtime percept admission requires DomainSpec")
    if not isinstance(world_step, WorldStepResult):
        raise TypeError("runtime percept admission requires WorldStepResult")
    if not isinstance(projection_model, ObservationProjectionModelSpec):
        raise TypeError(
            "runtime percept admission requires ObservationProjectionModelSpec"
        )
    if not isinstance(prior_ledger, RuntimeEvidenceLedger):
        raise TypeError("runtime percept admission requires RuntimeEvidenceLedger")

    validate_narrative(story, domain)
    if prior_ledger.domain_id != domain.domain_id:
        raise ValueError("runtime ledger domain id mismatch")
    if prior_ledger.domain_version != domain.version:
        raise ValueError("runtime ledger domain version mismatch")
    if prior_ledger.domain_spec_hash != domain.content_hash:
        raise ValueError("runtime ledger domain spec hash mismatch")
    if prior_ledger.source_story_hash != story.content_hash:
        raise ValueError("runtime ledger source story mismatch")
    if prior_ledger.current_world_state_hash != world_step.prior_state.content_hash:
        raise ValueError(
            "runtime ledger current world state does not match world-step prior"
        )
    if prior_ledger.current_step_index != world_step.prior_state.step_index:
        raise ValueError("runtime ledger step does not match world-step prior")
    if prior_ledger.source_at_time != world_step.prior_state.source_at_time:
        raise ValueError("runtime ledger source cutoff does not match world-step prior")
    if world_step.next_state.source_at_time != prior_ledger.source_at_time:
        raise ValueError(
            "runtime ledger source cutoff does not match world-step next state"
        )
    if world_step.next_state.step_index != prior_ledger.current_step_index + 1:
        raise ValueError("runtime admission world step must advance exactly one step")


def admit_world_percepts(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    projection_model: ObservationProjectionModelSpec,
    prior_ledger: RuntimeEvidenceLedger,
) -> RuntimePerceptAdmissionResult:
    try:
        _validate_admission_source(
            story,
            domain,
            world_step,
            projection_model,
            prior_ledger,
        )
    except RuntimePerceptAdmissionError:
        raise
    except (TypeError, ValueError) as error:
        raise RuntimePerceptAdmissionError(
            "runtime percept admission source validation failed"
        ) from error

    try:
        projection_result = project_world_observations(
            story,
            domain,
            world_step,
            projection_model,
        )
    except ObservationProjectionError as error:
        raise RuntimePerceptAdmissionError(
            "runtime percept projection failed"
        ) from error

    try:
        projection_result_hash = projection_result.content_hash
        runtime_evidence = tuple(
            RuntimeEpistemicEvidence(
                observer_id=projected.observer_id,
                channel=projected.channel,
                cell=projected.fact.cell,
                relation=projected.fact.relation,
                value=projected.fact.value,
                step_index=projected.step_index,
                projected_observation_hash=projected.content_hash,
                projection_result_hash=projection_result_hash,
                projection_model_hash=projection_result.model_hash,
                source_world_state_hash=projected.source_world_state_hash,
                source_world_step_hash=projected.source_world_step_hash,
                source_transition_hashes=projected.source_transition_hashes,
                projection_spec_hash=projected.projection_spec_hash,
                projection_sample_hash=projected.stochastic_sample_hash,
            )
            for projected in projection_result.observations
        )
        batch = RuntimeEvidenceBatch(
            prior_ledger_hash=prior_ledger.content_hash,
            step_index=world_step.next_state.step_index,
            source_prior_world_state_hash=world_step.prior_state.content_hash,
            source_world_state_hash=world_step.next_state.content_hash,
            source_world_step_hash=world_step.content_hash,
            projection_model_hash=projection_result.model_hash,
            projection_result_hash=projection_result_hash,
            evidence=runtime_evidence,
            projection_sample_hashes=tuple(
                sample.content_hash
                for sample in projection_result.stochastic_samples
            ),
        )
        next_ledger = RuntimeEvidenceLedger(
            domain_id=prior_ledger.domain_id,
            domain_version=prior_ledger.domain_version,
            domain_spec_hash=prior_ledger.domain_spec_hash,
            source_story_hash=prior_ledger.source_story_hash,
            source_at_time=prior_ledger.source_at_time,
            initial_world_state_hash=prior_ledger.initial_world_state_hash,
            current_world_state_hash=world_step.next_state.content_hash,
            batches=prior_ledger.batches + (batch,),
        )
        return RuntimePerceptAdmissionResult(
            prior_ledger_hash=prior_ledger.content_hash,
            projection_result=projection_result,
            evidence_batch=batch,
            next_ledger=next_ledger,
        )
    except RuntimePerceptAdmissionError:
        raise
    except (TypeError, ValueError) as error:
        raise RuntimePerceptAdmissionError(
            "runtime percept admission result construction failed"
        ) from error
