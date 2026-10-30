"""One interface for a Run at a prerequisite state: `prerequisite(state, **choices)`.

A state is one `State` entry: its builder (current product commands), whether
tests only read or Fork it, the read-only state it is forked from, and the
choice sets a batch builds once. `CATALOG` holds the shared states; a module may
define a `State` of its own beside its tests. The choices are the builder's
keywords; each distinct set is its own Run, as is each `slot`.

Two adapters serve the interface:

- The batch catalog (`tests/prerequisite_store.py`). In a batch, a read-only
  state requested with one of its stored choice sets is the batch's state:
  built once per source key before any module starts, checked once per process
  and never rebuilt here. It stays in the batch root.
- The in-process build. Otherwise the state is built at the first request, in a
  directory the requesting test class owns and removes when the class finishes
  (from a helper called with no test, the module owns it). Nothing is built at
  import time or kept between runs, so an exact test run builds its own.

A read-only state (an executed terminal Run) is shared: tests read it, or Fork
their own child from it through the product command. After every test that used
it, cleanup checks that its files are byte-identical; a changed Run fails that
test and is dropped, so the next request builds a fresh one. A module-scoped Run
is checked when its module ends.

Every other state, and a read-only one requested `writable=True`, is a Run the
test may change: review-pending, approved and evidence Runs, and executed Runs a
test records into. These join the batch catalog only once a Fork child can
continue them read-only (#294). Most of their state is bound to its absolute
path (work directory, contact journal locators, registered inputs), so the Run
is never relocated. Before every use it is restored at the same path from
pristine bytes taken after the build. The files the Run registered (prompt,
runtime configuration, request, FASTQs) are never rewritten, because the
approval binds their device and inode: each request checks their recorded
identity and bytes and rebuilds the Run when any changed. A state forked from a
read-only state records that parent's registered inputs, which stay in the
parent's directory; their identity is checked the same way, and the parent's
byte check covers the test that asked. The behavior under test still runs per
scenario on the restored Run.

After every test, either kind checks that no writer owns the Journey, the Run
advancement composition is back to its defaults and no thread started by the
test is still alive. A restored Run also refuses to restore under a live writer
or a thread left by an earlier scenario of the same test.

`build_store` is the testing runner's prerequisite builder entry: it builds every
stored choice set of the catalog into the batch's store.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import shutil
import tempfile
import threading
import unittest
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from bio_neb.application import advancement
from bio_neb.application.input_journey import (
    ResearchRequestRegistered,
    RunInputsRegistered,
    validate_fact,
)
from bio_neb.investigation.investigation_request import (
    InvestigationRequestRegistered,
)
from bio_neb.journey import Journey
from tests import prerequisite_store
from tests.parabricks_operation_fakes import BAM_BYTES, VCF_BYTES, approved_input
from tests.planning_assignment_fakes import (
    execution_configuration,
    prepare_configured_run,
)
from tests.prerequisite_builds import measured_build
from tests.terminal_journey_fakes import EVIDENCE_REQUEST, execute_terminal

_SHARED = "_shared_prerequisites"
_SHARED_CHECKED = "_shared_prerequisite_checks"
_RESTORED = "_restored_prerequisites"
_RESTORED_CHECKED = "_restored_prerequisite_checks"
_MISSING = object()
# Read-only Runs requested with no test, owned by the running module.
_MODULE: dict[tuple, _Shared] = {}
_MODULE_DIRECTORIES: list[tempfile.TemporaryDirectory] = []


@dataclass(frozen=True)
class State:
    """One prerequisite state: how it is built and how tests may use it.

    `build(root, **choices)` creates the Run under root through product commands
    and returns its Journey file and a frozen value shared by every request; a
    state with `forks_from` receives the parent's Journey file after root.
    `read_only` states are shared and byte-checked; others are restored in place.
    `stored` names the choice sets a batch builds once, by label.
    """

    name: str
    build: Callable[..., tuple[Path, Any]]
    read_only: bool = False
    forks_from: str | None = None
    # The parent state's choices, such as the request its plan names, or a
    # function of this state's own choices that gives them.
    forks_from_choices: Mapping[str, Any] | Callable[..., Mapping[str, Any]] = field(
        default_factory=dict
    )
    stored: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class Prerequisite(NamedTuple):
    """A Run at a prerequisite state: its root, Journey file and build value."""

    root: Path
    journey_file: Path
    value: Any


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every path below root, with file bytes; directories map to None."""
    return {
        path.relative_to(root).as_posix(): (
            None if path.is_dir() else path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
    }


def _frozen(value: Any) -> Any:
    """A hashable form of a choice value that keeps every type a builder can see.

    Mapping order does not matter; a mapping, list and tuple of the same items,
    or `1` and `True` as a value or a mapping key, are different choices. An
    unhashable leaf is refused.
    """
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: repr(item[0]))
        return (
            type(value),
            tuple((_frozen(key), _frozen(item)) for key, item in items),
        )
    if isinstance(value, list | tuple):
        return (type(value), tuple(_frozen(item) for item in value))
    hash(value)
    return (type(value), value)


def _choice_key(state: State, choices: Mapping[str, Any]) -> Any:
    """The state's final choices as its builder binds them, defaults applied.

    Omitting a choice and passing its default are one Run; a choice the builder
    does not take, or one naming the root or parent the interface supplies, is
    refused before any build, whatever is cached.
    """
    signature = inspect.signature(state.build)
    owned = (None, None) if state.forks_from else (None,)
    leading = list(signature.parameters)[: len(owned)]
    bound = signature.bind(*owned, **choices)
    bound.apply_defaults()
    return _frozen(
        {name: value for name, value in bound.arguments.items() if name not in leading}
    )


def _advancement_bound() -> list[str]:
    """Run advancement context variables not at their defaults."""
    return [
        variable.name
        for variable in vars(advancement).values()
        if isinstance(variable, ContextVar)
        and variable.get()
        is not (
            advancement._DEFAULT_COMPOSITION
            if variable is advancement._COMPOSITION
            else None
        )
    ]


def _class_store(cls: type, name: str) -> dict:
    store = cls.__dict__.get(name)
    if store is None:
        store = {}
        setattr(cls, name, store)
        cls.addClassCleanup(delattr, cls, name)
    return store


def _leftover_threads(threads: set) -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread not in threads and thread.is_alive()
    ]


def _check_released(test, journey_file: Path, what: str, threads: set) -> None:
    test.assertIsNone(
        Journey.processing_owner(journey_file),
        f"a live writer still owns the {what}",
    )
    test.assertEqual(_advancement_bound(), [], "Run advancement composition left bound")
    test.assertEqual(_leftover_threads(threads), [], "threads outlived the test")


class _Shared:
    def __init__(
        self,
        state: State,
        key: tuple,
        root: Path,
        journey_file: Path,
        value: Any = None,
        *,
        stored: str | None = None,
    ) -> None:
        self.state = state
        self.key = key
        self.root = root
        self.journey_file = journey_file
        self.value = value
        self.stored = stored
        self.tree = _tree(root)

    def changed(self) -> list[str]:
        after = _tree(self.root)
        return sorted(
            path
            for path in self.tree.keys() | after.keys()
            if self.tree.get(path, _MISSING) != after.get(path, _MISSING)
        )


def _stored_label(state: State, slot: str, choices: dict) -> str | None:
    if slot != "default":
        return None
    return next(
        (
            f"{state.name}-{label}"
            for label, stored in state.stored.items()
            if _choice_key(state, stored) == _choice_key(state, choices)
        ),
        None,
    )


def _drop(shared: dict, entry: _Shared) -> None:
    """Serve a changed Run no more; the next request builds a fresh one."""
    if shared.get(entry.key) is entry:
        del shared[entry.key]
    if entry.stored is not None:
        prerequisite_store.reject(entry.stored)


def _module_check(entry: _Shared, threads: set) -> None:
    """At module end: the module's Run is unchanged and released, before removal."""
    problems = []
    changed = entry.changed()
    if changed:
        _drop(_MODULE, entry)
        problems.append(f"shared {entry.state.name} prerequisite changed: {changed}")
    if Journey.processing_owner(entry.journey_file) is not None:
        problems.append(f"a live writer still owns the shared {entry.state.name}")
    if _advancement_bound():
        problems.append(
            f"Run advancement composition left bound: {_advancement_bound()}"
        )
    if _leftover_threads(threads):
        problems.append(f"threads outlived the module: {_leftover_threads(threads)}")
    if problems:
        raise AssertionError("; ".join(problems))


def _release_module() -> None:
    _MODULE.clear()
    while _MODULE_DIRECTORIES:
        _MODULE_DIRECTORIES.pop().cleanup()


def _shared(test, state: State, slot: str, choices: dict) -> _Shared:
    """The read-only Run for this state, choices and slot in the owner's scope."""
    shared = _MODULE if test is None else _class_store(type(test), _SHARED)
    key = (slot, state.name, _choice_key(state, choices))
    threads = set(threading.enumerate())
    entry = shared.get(key)
    if entry is None:
        if test is None and not (_MODULE or _MODULE_DIRECTORIES):
            # Registered first, so it runs after the module's byte checks, and
            # before any build, so a failed one is removed too.
            unittest.addModuleCleanup(_release_module)
        label = _stored_label(state, slot, choices)
        stored = None if label is None else prerequisite_store.admitted(label)
        if stored is not None:
            # The batch's catalog state: read-only, Forked from, never rebuilt here.
            entry = _Shared(state, key, *stored, stored=label)
        else:
            directory = tempfile.TemporaryDirectory(prefix="shared-prerequisite-")
            if test is None:
                _MODULE_DIRECTORIES.append(directory)
            else:
                type(test).addClassCleanup(directory.cleanup)
            root = Path(directory.name).resolve()
            journey_file, value = state.build(root / "source", **choices)
            entry = _Shared(state, key, root, Path(journey_file), value)
        if test is None:
            unittest.addModuleCleanup(_module_check, entry, threads)
        shared[key] = entry
    if test is not None:
        checked = test.__dict__.setdefault(_SHARED_CHECKED, set())
        if id(entry) not in checked:
            checked.add(id(entry))
            test.addCleanup(_check_shared, test, shared, entry, threads)
    return entry


@measured_build("parent-byte-check")
def _check_shared(test, shared: dict, entry: _Shared, threads: set) -> None:
    changed = entry.changed()
    if changed:
        _drop(shared, entry)
    test.assertEqual(changed, [], f"shared {entry.state.name} prerequisite changed")
    _check_released(test, entry.journey_file, f"shared {entry.state.name}", threads)


def _recorded_inputs(journey_file: Path) -> tuple[Path, ...]:
    """The files the Run registered, read through the registration contracts."""
    paths: list[str] = []
    for event in Journey.open(journey_file).read():
        fact = validate_fact(event)
        if isinstance(fact, ResearchRequestRegistered):
            paths.append(fact.prompt_file)
        elif isinstance(fact, RunInputsRegistered):
            paths.append(fact.runtime_config_file)
            paths.extend(item.registered_path for item in fact.inputs)
        elif (
            event["event_type"]
            == InvestigationRequestRegistered.model_fields["event_type"].default
        ):
            paths.append(
                InvestigationRequestRegistered.model_validate(
                    event["payload"]
                ).request_file
            )
    return tuple(Path(path) for path in paths)


def _identity(path: Path) -> tuple | None:
    """Device, inode, size, times and bytes of a registered input."""
    try:
        status = path.stat()
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
        hashlib.sha256(content).hexdigest(),
    )


def _writable_retry(function, path, _error) -> None:
    os.chmod(os.path.dirname(path), 0o700)
    function(path)


def _clear(directory: Path, keep: frozenset[Path]) -> None:
    """Remove everything below directory except the kept files."""
    for entry in directory.iterdir():
        if entry in keep:
            continue
        if entry.is_dir() and not entry.is_symlink():
            if any(path.is_relative_to(entry) for path in keep):
                _clear(entry, keep)
            else:
                shutil.rmtree(entry, onexc=_writable_retry)
        else:
            entry.unlink()


class _Restorable:
    def __init__(
        self,
        build: Callable[[Path], tuple[Path, Any]],
        directory: Path,
        shared_roots: tuple[Path, ...] = (),
    ):
        self.build = build
        self.shared_roots = shared_roots
        self.root = directory / "run"
        self.pristine = directory / "pristine"
        self.inputs: dict[Path, tuple | None] = {}
        self.journey_file: Path | None = None
        self.value: Any = None

    @measured_build("baseline-make")
    def make(self) -> None:
        for path in (self.root, self.pristine):
            if path.exists():
                shutil.rmtree(path, onexc=_writable_retry)
        self.root.mkdir(parents=True)
        journey_file, self.value = self.build(self.root)
        self.journey_file = Path(journey_file)
        recorded = _recorded_inputs(self.journey_file)
        outside = [
            str(path)
            for path in recorded
            if not path.is_relative_to(self.root)
            and not any(path.is_relative_to(shared) for shared in self.shared_roots)
        ]
        if outside:
            raise AssertionError(f"registered inputs outside the baseline: {outside}")
        self.inputs = {path: _identity(path) for path in recorded}
        keep = set(recorded)
        shutil.copytree(
            self.root,
            self.pristine,
            symlinks=True,
            ignore=lambda directory, names: [
                name for name in names if Path(directory) / name in keep
            ],
        )

    def inputs_unchanged(self) -> bool:
        return all(_identity(path) == known for path, known in self.inputs.items())

    @measured_build("baseline-restore")
    def restore(self) -> None:
        _clear(self.root, frozenset(self.inputs))
        shutil.copytree(self.pristine, self.root, symlinks=True, dirs_exist_ok=True)


def _make(baselines: dict, key: tuple, baseline: _Restorable) -> None:
    try:
        baseline.make()
    except BaseException:
        # A failed build is never served; the next request builds again.
        del baselines[key]
        raise


def _restored(test, state: State, slot: str, choices: dict) -> Prerequisite:
    """The class's Run for this state, restored to its pristine bytes.

    Every call restores; the thread baseline for the leak checks is taken at the
    test's first request for this Run.
    """
    if test is None:
        raise TypeError(f"a writable {state.name} Run needs the test that owns it")
    if state.forks_from is not None:
        # A parent rebuilt after a failed byte check has a new path, so its
        # children are new Runs.
        parent_choices = (
            state.forks_from_choices(**choices)
            if callable(state.forks_from_choices)
            else state.forks_from_choices
        )
        parent = _shared(
            test, CATALOG[state.forks_from], "default", dict(parent_choices)
        )
        build = functools.partial(
            _forked_build, state.build, parent.journey_file, choices
        )
        key = (slot, state.name, _choice_key(state, choices), str(parent.journey_file))
        shared_roots = (parent.root,)
    else:
        build = functools.partial(_build, state.build, choices)
        key = (slot, state.name, _choice_key(state, choices))
        shared_roots = ()
    cls = type(test)
    baselines = _class_store(cls, _RESTORED)
    checked = test.__dict__.setdefault(_RESTORED_CHECKED, {})
    threads = set(threading.enumerate())
    baseline = baselines.get(key)
    if baseline is None:
        directory = tempfile.TemporaryDirectory(prefix="restored-baseline-")
        cls.addClassCleanup(directory.cleanup)
        baseline = baselines[key] = _Restorable(
            build, Path(directory.name).resolve(), shared_roots
        )
        _make(baselines, key, baseline)
    else:
        test.assertIsNone(
            Journey.processing_owner(baseline.journey_file),
            "a live writer still owns the restored baseline",
        )
        if id(baseline) in checked:
            test.assertEqual(
                _leftover_threads(checked[id(baseline)]),
                [],
                "threads outlived an earlier scenario",
            )
        if baseline.inputs_unchanged():
            baseline.restore()
        else:
            _make(baselines, key, baseline)
    if id(baseline) not in checked:
        checked[id(baseline)] = threads
        test.addCleanup(
            lambda: _check_released(
                test, baseline.journey_file, "restored baseline", threads
            )
        )
    return Prerequisite(baseline.root, baseline.journey_file, baseline.value)


def _build(build: Callable, choices: dict, root: Path) -> tuple[Path, Any]:
    return build(root, **choices)


def _forked_build(
    build: Callable, parent_file: Path, choices: dict, root: Path
) -> tuple[Path, Any]:
    return build(root, parent_file, **choices)


def _state(state: str | State) -> State:
    return CATALOG[state] if isinstance(state, str) else state


def prerequisite(
    state: str | State,
    *,
    test: unittest.TestCase | None = None,
    slot: str = "default",
    writable: bool = False,
    **choices: Any,
) -> Prerequisite:
    """A Run at this state with these final choices.

    `state` is a `CATALOG` name or a module's own `State`. With `test`, the
    test's class owns the Run and the test's cleanup checks it; without, only a
    read-only state is served and the running module owns it. `writable`
    serves a read-only state restored in place instead of shared.
    """
    resolved = _state(state)
    if resolved.read_only and not writable:
        entry = _shared(test, resolved, slot, choices)
        return Prerequisite(entry.root, entry.journey_file, entry.value)
    return _restored(test, resolved, slot, choices)


class RunPrerequisites:
    """unittest.TestCase mixin: `self.prerequisite(...)` owned by this test's class."""

    def prerequisite(
        self,
        state: str | State,
        *,
        slot: str = "default",
        writable: bool = False,
        **choices: Any,
    ) -> Prerequisite:
        return prerequisite(state, test=self, slot=slot, writable=writable, **choices)


def _executed(
    root: Path,
    *,
    output_payloads=None,
    investigation_request=None,
    scientific_intent=None,
    scientific_rsid=None,
    failed_turns=None,
    artifact_verification=None,
) -> tuple[Path, None]:
    """An executed terminal Run at its after-Execution child point, planned for
    its registered ``investigation_request`` when given; ``scientific_intent``
    adds a prompt sentence naming ``scientific_rsid``, which planning keeps as
    its own rsID slot; ``failed_turns`` (``agent_invocation``'s) and
    ``artifact_verification`` are its runtime's."""
    request = {
        **(
            {}
            if investigation_request is None
            else {"investigation_request": investigation_request}
        ),
        **(
            {}
            if scientific_intent is None
            else {
                "scientific_intent": scientific_intent,
                "scientific_rsid": scientific_rsid,
            }
        ),
        **({} if failed_turns is None else {"failed_turns": failed_turns}),
        **(
            {}
            if artifact_verification is None
            else {"artifact_verification": artifact_verification}
        ),
    }
    return Path(
        execute_terminal(root, output_payloads=output_payloads, **request)
    ), None


def _review_pending(root: Path) -> tuple[Path, Any]:
    """A Run waiting at its Execution review; value is that review."""
    registration, review, _ = prepare_configured_run(root)
    return Path(registration.journey_file), review


def _approved(
    root: Path,
    *,
    configure: Callable | None = None,
    investigation_request: dict | None = None,
    old_journey: bool = False,
    **launch_overrides,
) -> tuple[Path, Any]:
    """An approved Run; value is its approved Execution input.

    `configure` derives the configuration from the Run's default one, so its
    work directory stays inside the restored Run root. `old_journey` plans it as
    before #342, naming no target: only its verified Execution rests at the
    request-needed stop.
    """
    from contextlib import nullcontext

    from tests.planning_assignment_fakes import planned_before_required_targets

    configuration = (
        None if configure is None else configure(execution_configuration(root))
    )
    with planned_before_required_targets() if old_journey else nullcontext():
        approved = approved_input(
            root,
            configuration=configuration,
            launch_overrides=launch_overrides or None,
            investigation_request=investigation_request,
        )
    return Path(approved.journey_file), approved


def _planned_evidence(root: Path) -> tuple[Path, None]:
    """An executed Run that holds its own observed Investigation."""
    from tests.finalization_fakes import planned_evidence_journey

    return planned_evidence_journey(root).path, None


def _retained_evidence(
    root: Path, parent_file: Path, *, query_stdout: bytes = b""
) -> tuple[Path, None]:
    """A child of the executed parent holding observed Investigation evidence.

    Finalization has not started.
    """
    from tests.finalization_fakes import retained_execution_evidence_journey

    journey = retained_execution_evidence_journey(
        root, query_stdout=query_stdout, parent_file=parent_file
    )
    return journey.path, None


CATALOG: dict[str, State] = {
    state.name: state
    for state in (
        State(
            "executed",
            _executed,
            read_only=True,
            # By the output payloads their Execution produced.
            stored={
                "default": {},
                "vcf-bam": {
                    "output_payloads": {"VCF": VCF_BYTES, "aligned_bam": BAM_BYTES}
                },
                "vcf": {"output_payloads": {"VCF": VCF_BYTES}},
                "evidence-request": {"investigation_request": EVIDENCE_REQUEST},
            },
        ),
        State("review-pending", _review_pending),
        State("approved", _approved),
        State("planned-evidence", _planned_evidence),
        State(
            "retained-evidence",
            _retained_evidence,
            forks_from="executed",
            forks_from_choices={"investigation_request": EVIDENCE_REQUEST},
        ),
    )
}


def _build_stored(state: State, choices: Mapping[str, Any], root: Path) -> Path:
    return Path(state.build(root, **choices)[0])


def build_store(store: str) -> int:
    """The prerequisite builder entry: build the catalog's stored states in store."""
    return prerequisite_store.build_store(
        store,
        {
            f"{state.name}-{label}": functools.partial(_build_stored, state, choices)
            for state in CATALOG.values()
            for label, choices in state.stored.items()
        },
    )


__all__ = [
    "CATALOG",
    "Prerequisite",
    "RunPrerequisites",
    "State",
    "build_store",
    "prerequisite",
]
