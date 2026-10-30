"""The question a reading reads for: the ids it lists, and the question as the model
reads it once code has found some of them (ADR 0046: list intent in code)."""

from __future__ import annotations

import re

# The ids a question lists: rsIDs, and positions written as contig:number.
_LISTED_ID = re.compile(
    r"(?<![A-Za-z0-9_])(?:rs\d+|(?:chr)?[A-Za-z0-9_.]*[A-Za-z0-9]:\d{1,3}(?:,\d{3})+"
    r"|(?:chr)?[A-Za-z0-9_.]*[A-Za-z0-9]:\d{3,})(?![\d,]*\d)",
    re.IGNORECASE,
)


def requested_ids(question: str) -> tuple[str, ...]:
    """The rsIDs and positions (``4:54108297``, ``chr4:54,108,297``) the question lists."""

    return tuple(dict.fromkeys(m.group() for m in _LISTED_ID.finditer(question)))


# What may stand between two listed ids: separators, and one "and" or "or".
_BETWEEN_IDS = re.compile(r"[\s,;]*(?:(?:and|or)\s+)?", re.IGNORECASE)


def model_question(question: str, found: set[str]) -> str:
    """The question as the model reads it: each run of listed ids that code found is
    replaced by their count, so the model reads only the misses."""

    runs: list[list[int]] = []  # [start, end, ids] of each run of found ids
    for match in _LISTED_ID.finditer(question):
        if match.group() not in found:
            continue
        if runs and _BETWEEN_IDS.fullmatch(question[runs[-1][1] : match.start()]):
            runs[-1][1:] = [match.end(), runs[-1][2] + 1]
        else:
            runs.append([match.start(), match.end(), 1])
    for start, end, count in reversed(runs):
        label = f"[{count} id{'s' if count > 1 else ''} found in code]"
        question = question[:start] + label + question[end:]
    return question
