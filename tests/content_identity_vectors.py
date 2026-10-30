"""Independent JSON identity oracle; callers select explicit public contract fields."""

import hashlib
import json


def expected_digest(material):
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def fields(value, names):
    return {name: value[name] for name in names.split()}


def assert_json_identity(test, actual, value, names, namespace):
    public = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    material = fields(public, names)
    test.assertEqual(actual, namespace + expected_digest(material))
