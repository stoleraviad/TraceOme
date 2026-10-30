"""Pure object-location encoding; no clients, credentials or network effects."""

from urllib.parse import urlsplit


def object_location(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def object_address(location: str) -> tuple[str, str]:
    parsed = urlsplit(location)
    key = parsed.path.removeprefix("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.query
        or parsed.fragment
        or object_location(parsed.netloc, key) != location
    ):
        raise ValueError("object_location_invalid")
    return parsed.netloc, key
