"""Opt-in Logfire diagnostics. Nothing here supplies Journey authority."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from bio_neb import cli
from bio_neb.agentic_journey.invocation_observation import _observer

# The owner-selected Logfire project: US region, org ``bhalu``, project
# ``starter-project``; the write token names the project.
CLOUD_BASE_URL = "https://logfire-us.pydantic.dev"
CLOUD_TOKEN_NAME = "LOGFIRE_TOKEN"


class LocalInspection:
    """Capture full unscrubbed content to a fresh local JSONL file, optionally also cloud.

    Callers explicitly select the capture scope. Ordinary execution and Journey queries
    never configure Logfire. Local-only capture refuses every ambient ``LOGFIRE_*`` or
    ``OTEL_*`` name instead of silently constructing another exporter. Explicit cloud
    capture admits only ``LOGFIRE_TOKEN``, passes it to Logfire itself, and still
    refuses every other such name; a token alone never selects cloud. Errors affect
    diagnostics only.
    """

    def __init__(self, path: Path, *, cloud: bool = False):
        self.path = path
        self.cloud = cloud
        self.errors: list[str] = []
        self.logfire: Any = None
        self._stream: Any = None

    def __enter__(self):
        self._token = _observer.set(self)
        try:
            if not self.path.is_absolute():
                raise ValueError("capture requires an absolute external path")
            admitted = {CLOUD_TOKEN_NAME} if self.cloud else set()
            if any(
                name.startswith(("OTEL_", "LOGFIRE_")) and name not in admitted
                for name in os.environ
            ):
                raise ValueError("ambient telemetry configuration")
            token = os.environ.get(CLOUD_TOKEN_NAME) if self.cloud else None
            if self.cloud and not token:
                raise ValueError("cloud capture requires a write token")
            import logfire
            from opentelemetry.sdk.trace.export import (
                SimpleSpanProcessor,
                SpanExporter,
                SpanExportResult,
            )

            capture = self

            class LocalExporter(SpanExporter):
                def export(self, spans):
                    try:
                        for span in spans:
                            row = json.loads(span.to_json())
                            row["instrumentation_scope"] = {
                                "name": span.instrumentation_scope.name,
                                "version": span.instrumentation_scope.version,
                            }
                            capture._stream.write(json.dumps(row) + "\n")
                        capture._stream.flush()
                    except Exception as error:
                        capture.errors.append(type(error).__name__)
                        return SpanExportResult.FAILURE
                    return SpanExportResult.SUCCESS

                def shutdown(self):
                    pass

            self._stream = self.path.open("x", encoding="utf-8")
            self.logfire = logfire.configure(
                local=True,
                send_to_logfire=self.cloud,
                token=token,
                service_name="bio-neb-development-inspection",
                environment="internal-poc",
                console=False,
                metrics=False,
                scrubbing=False,
                inspect_arguments=False,
                config_dir=self.path.parent,
                data_dir=self.path.parent,
                additional_span_processors=[SimpleSpanProcessor(LocalExporter())],
                advanced=logfire.AdvancedOptions(
                    base_url=CLOUD_BASE_URL if self.cloud else None,
                    resource_detectors=[],
                    emit_configuration_span=False,
                ),
            )
        except Exception as error:
            self.errors.append(type(error).__name__)
        return self

    def __exit__(self, *_):
        try:
            if self.logfire is not None:
                try:
                    if not self.logfire.force_flush():
                        self.errors.append("flush_failed")
                except Exception as error:
                    self.errors.append(type(error).__name__)
                self.logfire.shutdown()
        except Exception as error:
            self.errors.append(type(error).__name__)
        finally:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception as error:
                    self.errors.append(type(error).__name__)
            _observer.reset(self._token)

    def inspection_event(self, name: str, **attributes: Any) -> None:
        """Supplement native spans with already observed application semantics."""
        capture = self
        if capture is None or capture.logfire is None:
            return
        try:
            capture.logfire.info(name, **attributes)
        except Exception as error:
            capture.errors.append(type(error).__name__)

    @contextmanager
    def inspect_agent(
        self,
        agent: Any,
        *,
        journey_id: str,
        attempt_id: str,
        member_id: str | None = None,
    ):
        """Apply native instrumentation to this serial Agent invocation only."""
        capture = self
        if capture is None or capture.logfire is None:
            yield
            return
        from logfire import set_baggage

        previous = agent.instrument
        scope = ExitStack()
        try:
            capture.logfire.instrument_pydantic_ai(
                agent, version=5, include_content=True, include_binary_content=True
            )
            scope.enter_context(
                set_baggage(
                    journey_id=str(journey_id),
                    attempt_id=str(attempt_id),
                    agent_role=agent.name or "",
                    member_id=member_id or "",
                )
            )
            scope.enter_context(
                capture.logfire.span("Agent inspection", authority="diagnostic_only")
            )
        except Exception as error:
            capture.errors.append(type(error).__name__)
        try:
            yield
        finally:
            agent.instrument = previous
            try:
                scope.close()
            except Exception as error:
                capture.errors.append(type(error).__name__)


_CAPTURED_COMMANDS = (
    "begin",
    "resume",
    "answer",
    "investigation-begin",
    "report-explain",
)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("capture requires an absolute external path")
    return path


def main(argv: list[str] | None = None) -> int:
    """Run one allowed native command inside opt-in capture.

    The command's exit status is returned unchanged; unavailable capture is reported
    on stderr by error type only.
    """

    parser = argparse.ArgumentParser(
        prog="python -m bio_neb.devtools.agent_inspection",
        description="Run one native Bio Neb command with opt-in Logfire capture.",
    )
    parser.add_argument(
        "--capture",
        required=True,
        type=_absolute_path,
        metavar="ABSOLUTE_JSONL_PATH",
        help="fresh local JSONL copy of every captured span",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help=f"also export to the US Logfire project with {CLOUD_TOKEN_NAME}",
    )
    parser.add_argument("command", choices=_CAPTURED_COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    with LocalInspection(options.capture, cloud=options.cloud) as capture:
        status = cli.main([options.command, *options.arguments])
    if capture.errors:
        try:
            print(
                json.dumps(
                    {"capture": str(options.capture), "capture_errors": capture.errors}
                ),
                file=sys.stderr,
            )
        except Exception:
            # Even reporting a diagnostic failure must preserve the native result.
            pass
    return status


if __name__ == "__main__":
    raise SystemExit(main())
