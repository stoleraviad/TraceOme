"""Typed metadata inspection for registered FASTQ paths."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


class _ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


STAGED_OBJECT_SCHEME = "s3://"


def is_staged_object(registered_path: str) -> bool:
    """Whether a registered FASTQ names a staged storage object, not a local file."""
    return registered_path.startswith(STAGED_OBJECT_SCHEME)


def registered_locator(value: str) -> str:
    """An absolute local path, or a staged object's exact ``s3://bucket/key``."""
    import os

    from bio_neb.providers.object_locations import object_address

    if value and is_staged_object(value):
        try:
            object_address(value)
        except ValueError:
            raise ValueError(
                "registered_path must be one exact s3://bucket/key"
            ) from None
        return value
    if not value or not os.path.isabs(value):
        raise ValueError("registered_path must be absolute or s3://bucket/key")
    return value


class RegisteredInputPath(_ToolModel):
    role: StrictStr
    registered_path: StrictStr

    @field_validator("role")
    @classmethod
    def canonical_role(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("role must be a non-empty canonical string")
        return value

    @field_validator("registered_path")
    @classmethod
    def exact_locator(cls, value: str) -> str:
        return registered_locator(value)


class InspectInputMetadataInput(_ToolModel):
    input_set_id: StrictStr
    inputs: Tuple[RegisteredInputPath, ...] = Field(min_length=1)

    @field_validator("input_set_id")
    @classmethod
    def canonical_identity(cls, value: str) -> str:
        try:
            parsed = uuid.UUID(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("input_set_id must be a UUID") from error
        if str(parsed) != value:
            raise ValueError("input_set_id must be a canonical UUID")
        return value

    @model_validator(mode="after")
    def unique_roles(self) -> "InspectInputMetadataInput":
        roles = [item.role for item in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("registered input roles must be unique")
        return self


class FilesystemIdentity(_ToolModel):
    device: StrictStr
    inode: StrictStr


class ObservedInputMetadata(_ToolModel):
    role: StrictStr
    registered_path: StrictStr
    exists: StrictBool
    is_regular: StrictBool
    size_bytes: StrictInt | None
    filesystem_identity: FilesystemIdentity | None
    modified_at_ns: StrictInt | None

    @model_validator(mode="after")
    def coherent_metadata(self) -> "ObservedInputMetadata":
        facts = (self.size_bytes, self.filesystem_identity, self.modified_at_ns)
        if not self.exists:
            if self.is_regular or any(value is not None for value in facts):
                raise ValueError("missing input contains fabricated metadata")
        elif any(value is None for value in facts) or self.size_bytes < 0:
            raise ValueError("existing input lacks filesystem metadata")
        return self


class InputMetadataObservation(_ToolModel):
    started_at: StrictStr
    completed_at: StrictStr
    inputs: Tuple[ObservedInputMetadata, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_window(self) -> "InputMetadataObservation":
        moments = []
        for value in (self.started_at, self.completed_at):
            try:
                moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("observation timestamp must be ISO-8601") from error
            if moment.tzinfo is None or moment.utcoffset() is None:
                raise ValueError("observation timestamp must be timezone-aware")
            moments.append(moment)
        if moments[1] < moments[0]:
            raise ValueError("observation window completed before it started")
        return self


@runtime_checkable
class InputMetadataAdapter(Protocol):
    """Supply ordered metadata observations for registered input requests."""

    def observe(
        self, requests: Sequence[RegisteredInputPath]
    ) -> InputMetadataObservation: ...


class InspectedInput(_ToolModel):
    role: StrictStr
    registered_path: StrictStr
    exists: Literal[True]
    is_regular: Literal[True]
    size_bytes: StrictInt = Field(gt=0)
    filesystem_identity: FilesystemIdentity
    modified_at_ns: StrictInt


FailureCode = Literal[
    "adapter_output_invalid",
    "input_paths_duplicate",
    "input_missing",
    "input_not_regular",
    "input_empty",
]


class InputMetadataFailure(_ToolModel):
    code: FailureCode
    role: StrictStr | None = None
    registered_path: StrictStr | None = None

    @model_validator(mode="after")
    def coherent_scope(self) -> "InputMetadataFailure":
        if self.code == "adapter_output_invalid":
            if self.role is not None or self.registered_path is not None:
                raise ValueError("adapter failure cannot fabricate an input scope")
        elif self.code == "input_paths_duplicate":
            if self.role is not None or self.registered_path is None:
                raise ValueError("duplicate-path failure has invalid scope")
        elif self.role is None or self.registered_path is None:
            raise ValueError("input failure requires role and registered path")
        return self


class InputMetadataInspectionSucceeded(_ToolModel):
    status: Literal["succeeded"] = "succeeded"
    input_set_id: StrictStr
    inputs: Tuple[InspectedInput, ...] = Field(min_length=1)
    observation: InputMetadataObservation
    summary: Literal["Supplied inputs passed metadata inspection."] = (
        "Supplied inputs passed metadata inspection."
    )


class InputMetadataInspectionFailed(_ToolModel):
    status: Literal["failed"] = "failed"
    input_set_id: StrictStr
    failures: Tuple[InputMetadataFailure, ...] = Field(min_length=1)
    observation: InputMetadataObservation | None
    summary: Literal["Supplied input metadata inspection failed."] = (
        "Supplied input metadata inspection failed."
    )

    @model_validator(mode="after")
    def evidence_matches_failure(self) -> "InputMetadataInspectionFailed":
        malformed_count = sum(
            failure.code == "adapter_output_invalid" for failure in self.failures
        )
        if self.observation is None:
            if malformed_count != 1 or len(self.failures) != 1:
                raise ValueError("malformed adapter output must be the sole failure")
        elif malformed_count:
            raise ValueError("validated observation cannot have an adapter failure")
        return self


InputMetadataResult = Annotated[
    Union[InputMetadataInspectionSucceeded, InputMetadataInspectionFailed],
    Field(discriminator="status"),
]


class InputMetadataInspectionTool:
    """Derive metadata facts from one ordered observation."""

    name = "inspect_input_metadata"

    def inspect(
        self,
        tool_input: InspectInputMetadataInput,
        adapter: InputMetadataAdapter,
    ) -> InputMetadataResult:
        requests = tool_input.inputs
        raw = adapter.observe(requests)
        try:
            observation = self._validated_observation(raw, requests)
        except Exception:
            return InputMetadataInspectionFailed(
                input_set_id=tool_input.input_set_id,
                failures=(InputMetadataFailure(code="adapter_output_invalid"),),
                observation=None,
            )

        return self.derive_from_observation(tool_input, observation)

    def derive_from_observation(
        self,
        tool_input: InspectInputMetadataInput,
        observation: InputMetadataObservation,
    ) -> InputMetadataResult:
        """Derive the only lawful result from already validated exact evidence."""

        observation = self._validated_observation(observation, tool_input.inputs)
        failures: list[InputMetadataFailure] = []
        paths = [item.registered_path for item in tool_input.inputs]
        duplicate_paths = {path for path in paths if paths.count(path) > 1}
        for path in dict.fromkeys(paths):
            if path in duplicate_paths:
                failures.append(
                    InputMetadataFailure(
                        code="input_paths_duplicate", registered_path=path
                    )
                )
        for observed in observation.inputs:
            if not observed.exists:
                failures.append(
                    InputMetadataFailure(
                        code="input_missing",
                        role=observed.role,
                        registered_path=observed.registered_path,
                    )
                )
            elif not observed.is_regular:
                failures.append(
                    InputMetadataFailure(
                        code="input_not_regular",
                        role=observed.role,
                        registered_path=observed.registered_path,
                    )
                )
            elif observed.size_bytes == 0:
                failures.append(
                    InputMetadataFailure(
                        code="input_empty",
                        role=observed.role,
                        registered_path=observed.registered_path,
                    )
                )
        if failures:
            return InputMetadataInspectionFailed(
                input_set_id=tool_input.input_set_id,
                failures=tuple(failures),
                observation=observation,
            )

        inspected = tuple(
            InspectedInput(
                role=item.role,
                registered_path=item.registered_path,
                exists=True,
                is_regular=True,
                size_bytes=item.size_bytes,
                filesystem_identity=FilesystemIdentity(
                    device=item.filesystem_identity.device,
                    inode=item.filesystem_identity.inode,
                ),
                modified_at_ns=item.modified_at_ns,
            )
            for item in observation.inputs
        )
        return InputMetadataInspectionSucceeded(
            input_set_id=tool_input.input_set_id,
            inputs=inspected,
            observation=observation,
        )

    @staticmethod
    def _validated_observation(
        value: Any, requests: Sequence[RegisteredInputPath]
    ) -> InputMetadataObservation:
        observation = (
            InputMetadataObservation.model_validate(value.model_dump())
            if isinstance(value, BaseModel)
            else InputMetadataObservation.model_validate(value)
        )
        expected = [(item.role, item.registered_path) for item in requests]
        actual = [(item.role, item.registered_path) for item in observation.inputs]
        if actual != expected:
            raise ValueError("adapter observation identity does not match request")
        return observation
