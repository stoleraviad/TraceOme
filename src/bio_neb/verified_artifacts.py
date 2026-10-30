"""Verified object descriptors shared by execution, workers and scientific consumers."""

from pydantic import Field, StrictInt, StrictStr, model_validator

from bio_neb.content import canonical_json_digest
from bio_neb.contracts import FrozenContract


class VerifiedObject(FrozenContract):
    role: StrictStr
    bucket: StrictStr
    object_key: StrictStr
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactIdentity(FrozenContract):
    byte_count: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactProof(FrozenContract):
    """Exact created content and placement; proof never grants action authority.

    A null expected identity explicitly means the producer established an acquired
    identity without an independent upstream comparison. Source proof binds an
    unchanged replica to its original admission or validated worker evidence.
    """

    identity: ArtifactIdentity
    location: StrictStr = Field(min_length=1)
    producer: StrictStr = Field(min_length=1)
    expected_identity: ArtifactIdentity | None
    source_proof: StrictStr | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def matches_expected(self):
        if (
            self.expected_identity is not None
            and self.expected_identity != self.identity
        ):
            raise ValueError("artifact identity differs from independent expectation")
        return self

    @property
    def proof_id(self) -> str:
        return "artifact-proof:" + canonical_json_digest(self.model_dump(mode="json"))
