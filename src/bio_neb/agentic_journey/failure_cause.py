"""One shared cause helper: an exception's code identity, never its data.

Ticket 137's frozen evidence was `code: adapter_output_invalid`, `phase: submission`,
`reason: TypeError`.  Five distinct provider-payload defects produce that identical
record, because every refusal site retained `type(error).__name__` and discarded the one
fact that separates them -- *which line raised*.  Ticket 139 keeps the existing `reason`
string and populates it as `Class@module.py:line` instead.

An exception class name, a source file's basename, and a line number are **code
identity, never data**.  No input, provider value, or credential can reach any of the
three, so this needs no sanitiser and no per-site judgement -- unlike a free-text
exception message, whose secret-scanning machinery lives at `_safe_details` in
`adapters/nebius_serverless_job_contact.py`.  It is also still a plain string, so it
needs no new event field, no contract version, and no migration.

This module deliberately owns the naming for the whole repository rather than being
copied per adapter: a patch at one site would leave the same defect at every other.
"""

from __future__ import annotations

import re
from pathlib import PurePath
from types import TracebackType

# Both patterns are refusals, not sanitisers.  A dynamically created exception class or
# a synthesised code object can in principle carry any string, and rather than scrub one
# this drops the part that is not plainly code identity.
_CLASS_NAME = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")
_MODULE_NAME = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")
_UNNAMED_CLASS = "UnnamedError"


def _raise_site(traceback: TracebackType | None) -> tuple[str, int] | None:
    """Return the basename and line of the frame that actually raised.

    A traceback grows as the exception travels outward, so `tb_next` walks toward the
    raise point and its last entry is the site the record should name.
    """

    current = traceback
    if current is None:
        return None
    while current.tb_next is not None:
        current = current.tb_next
    name = PurePath(current.tb_frame.f_code.co_filename).name
    if not _MODULE_NAME.fullmatch(name):
        return None
    return name, current.tb_lineno


def cause_identity(error: BaseException) -> str:
    """Name one refusal's cause as `Class@module.py:line`.

    The class name alone is returned when the exception has no traceback, which happens
    only for an exception object that was constructed and never raised.
    """

    class_name = type(error).__name__
    if not _CLASS_NAME.fullmatch(class_name):
        class_name = _UNNAMED_CLASS
    site = _raise_site(error.__traceback__)
    if site is None:
        return class_name
    module_name, line = site
    return f"{class_name}@{module_name}:{line}"


__all__ = ["cause_identity"]
