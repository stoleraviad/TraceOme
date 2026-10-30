"""Canonical JSON identity, content ids and digests of owned bytes, not file verification,
and the one view of JSON content a model is shown."""

import hashlib
import json
import re
from dataclasses import dataclass


def canonical_json(value: object) -> str:
    """Render the one retained JSON representation accepted by Journey."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# Each control character but the line break.
_CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def shown_to_model(value: object) -> object:
    """JSON content as a model is shown it: in every string, each name included,
    each control character but the line break is one space.

    A raw control character a model copies from what it read into a call's
    arguments is no JSON string, and the endpoint dropped such a call whole
    (#395 run 2). One character stays one character, so offsets still count.
    """
    if isinstance(value, str):
        return _CONTROL.sub(" ", value)
    if isinstance(value, dict):
        return {
            shown_to_model(key): shown_to_model(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [shown_to_model(item) for item in value]
    return value


def sha256_bytes(value: bytes) -> str:
    """Digest exact bytes without selecting identity material or reading a file."""
    return hashlib.sha256(value).hexdigest()


def canonical_json_digest(value: object) -> str:
    """Hash canonical JSON material already selected and normalized by its owner."""
    return sha256_bytes(canonical_json(value).encode("utf-8"))


CONTENT_ID_PATTERN = r"^sha256:[0-9a-f]{64}$"
_CONTENT_ID_PREFIX = "sha256:"
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ContentId:
    """The one ``sha256:<digest>`` identity of exact bytes, built and parsed here."""

    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _DIGEST.fullmatch(self.digest):
            raise ValueError("content id digest must be 64 lowercase hex characters")

    @classmethod
    def of(cls, content: bytes) -> "ContentId":
        return cls(sha256_bytes(content))

    @classmethod
    def parse(cls, value: str) -> "ContentId":
        if not isinstance(value, str) or not value.startswith(_CONTENT_ID_PREFIX):
            raise ValueError("content id must use sha256")
        return cls(value.removeprefix(_CONTENT_ID_PREFIX))

    def __str__(self) -> str:
        return _CONTENT_ID_PREFIX + self.digest
