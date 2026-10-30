"""The shape every identifier kind's format matcher takes; domains declare the rules.

These are membership checks (ADR 0049): a value the model typed is checked against the
words it quotes. None of them reads a value out of free text.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class IdentifierRule:
    """One identifier kind's named format matcher, declared once by its owning domain.

    A value of this kind is typed by whoever states it: the model in a plan, a
    candidate in a Review. ``stated`` lists the distinct values a text states in
    this kind's format, in order. A typed value matches a quote when the quote
    states it, so ``rs1234`` matches ``RS1234`` and not ``RS4321``: the matcher
    checks a value, it never chooses one for a quote. A matched value stays a User
    assertion; ``source`` names the source Tool that can verify it, if any.
    """

    name: str
    stated: Callable[[str], tuple[str, ...]]
    source: str | None = None
    # Prose punctuation that may close a stated value when the format also allows
    # it inside one (a sentence's full stop after ``HG001``).
    closing: str = ""

    def matches(self, value: str, text: str) -> bool:
        """Whether ``text`` states the typed ``value`` in this kind's format."""

        stated = self.stated(text)
        return value in stated or any(value + mark in stated for mark in self.closing)


def word_aligned_spans(text: str, quote: str) -> tuple[int, ...]:
    """Start offsets where ``quote`` occurs without cutting a word at either edge."""

    def cuts(left: str, right: str) -> bool:
        return (
            bool(left and right)
            and (left.isalnum() or left == "_")
            and (right.isalnum() or right == "_")
        )

    starts = []
    start = text.find(quote)
    while start != -1:
        end = start + len(quote)
        if not cuts(text[start - 1 : start], quote[0]) and not cuts(
            quote[-1], text[end : end + 1]
        ):
            starts.append(start)
        start = text.find(quote, start + 1)
    return tuple(starts)


def _written_numbers(text: str) -> set[int]:
    """Every whole number ``text`` writes, a comma between digits read as a thousands
    separator (``207,469``), so a typed position can be found as written."""

    numbers: set[int] = set()
    run = ""
    for index, char in enumerate(text):
        if char.isdigit():
            run += char
            continue
        if char == "," and run and text[index + 1 : index + 2].isdigit():
            continue
        if run:
            numbers.add(int(run))
        run = ""
    if run:
        numbers.add(int(run))
    return numbers


def _written_base_runs(text: str) -> set[str]:
    """Every whole run of letters in ``text`` made only of bases (A, C, G, T, N in any
    case), upper-cased: ``C>T`` writes C and T, ``g.54108297G>T`` writes G and T, and
    the G of ``hg38`` is no run of its own."""

    runs: set[str] = set()
    run = ""
    for char in text + " ":
        if char.isalpha():
            run += char
            continue
        if run and all(letter in "ACGTN" for letter in run.upper()):
            runs.add(run.upper())
        run = ""
    return runs


def site_in_quote(
    quote: str, contig: str, position: int, ref: str | None, alt: str | None
) -> bool:
    """Whether ``quote`` writes the site the model typed from it (#472 Q1 a): the
    contig as written on whole words, the position as a written number, and REF and
    ALT, when typed, as whole base runs. A membership check: it never reads a site out
    of the quote, and a quote that writes another site does not hold this one."""

    runs = _written_base_runs(quote)
    return (
        bool(word_aligned_spans(quote, contig))
        and position in _written_numbers(quote)
        and all(bases in runs for bases in (ref, alt) if bases)
    )


def gene_change_in_quote(
    quote: str,
    gene: str,
    change: str,
    transcript: str | None,
    numbers: tuple[int, ...],
) -> bool:
    """Whether ``quote`` writes the gene change the model typed from it (#473 Q2 a): the
    gene, the change as written and, when typed, the transcript, each on whole words,
    and each typed codon or CDS number written in the change. A membership check: it
    never reads a change out of the quote."""

    return all(
        word_aligned_spans(quote, value)
        for value in (gene, change, transcript)
        if value is not None
    ) and set(numbers) <= _written_numbers(change)
