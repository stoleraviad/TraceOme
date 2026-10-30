from __future__ import annotations

import hashlib
import unittest

from pydantic import ValidationError

from bio_neb.journey import canonical_json
from bio_neb.parabricks.execution_observations import (
    explicit_worker_exit,
    parse_worker_markers,
    validate_truthful_observation_order,
    worker_marker_lines,
)
from bio_neb.parabricks.worker_exchange import PREPARED_ROLES
from bio_neb.providers.job_observations import (
    BoundedProviderLogObserved,
    FetchedProviderEnvelope,
    ProviderCreateObserved,
    ProviderLogWindowFailureEvidence,
    ProviderStateObserved,
    bounded_log_observation,
    log_window_failure_evidence,
)
from tests.execution_observation_fakes import _marker, _successful_log


def _provider_prefix(log: str):
    return (
        ProviderCreateObserved(
            provider_job_id="job-1",
            provider_job_name="bio-neb-exact",
            disposition="created",
        ),
        FetchedProviderEnvelope(
            provider_job_id="job-1",
            provider_job_name="bio-neb-exact",
            project_id="project-1",
            provider_image_reference="registry.invalid/worker:exact",
            platform="gpu-h100-sxm",
            preset="1gpu-16vcpu-200gb",
            disk_size_bytes=107374182400,
            subnet_id="subnet-1",
            timeout="14400s",
            container_command_requested="/opt/bio-neb/venv/bin/python",
            container_command_fetched=None,
            normalized_arguments=("/opt/bio-neb/worker.py", "/request.json"),
            injected_paths=("/request.json",),
            environment_names=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
            version_bound_secret_environment_names=("AWS_SECRET_ACCESS_KEY",),
            shared_memory_bytes=17179869184,
            restart_fetched="omitted",
        ),
        ProviderStateObserved(
            provider_job_id="job-1",
            provider_state="RUNNING",
            state_details=("worker scheduled",),
            created_at="2026-08-10T02:49:00+03:00",
            started_at="2026-08-10T02:50:00+03:00",
        ),
        ProviderStateObserved(
            provider_job_id="job-1",
            provider_state="COMPLETED",
            created_at="2026-08-10T02:49:00+03:00",
            started_at="2026-08-10T02:50:00+03:00",
            finished_at="2026-08-10T02:55:00+03:00",
        ),
        bounded_log_observation("job-1", log),
    )


class TruthfulExecutionObservationTests(unittest.TestCase):
    def test_cpu_derive_markers_remain_parseable(self):
        timestamp = "2026-08-10T00:00:00+00:00"

        def marker(name, value):
            return f"[{timestamp}] {name} {canonical_json(value)}"

        objects = [
            {
                "role": role,
                "object_key": f"references/bundle/{role}",
                "byte_count": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, role in enumerate(PREPARED_ROLES)
        ]
        lines = [
            marker(
                "BIO_NEB_PARABRICKS_WORKER_START",
                {
                    "interpreter": "/opt/bio-neb/venv/bin/python",
                    "interpreter_resolved": "/usr/bin/python3",
                    "request_schema": "bio-neb.parabricks-worker-request/7",
                },
            ),
            marker(
                "BIO_NEB_WORKER_REQUEST_ACCEPTED",
                {
                    "byte_count": 10,
                    "request_sha256": "1" * 64,
                    "worker_sha256": "2" * 64,
                },
            ),
            marker(
                "BIO_NEB_WORKER_TOOLS",
                {
                    "tools": {
                        "bwa": {"path": "/usr/bin/bwa", "version": "bwa"},
                        "samtools": {
                            "path": "/usr/bin/samtools",
                            "version": "samtools",
                        },
                    }
                },
            ),
            marker(
                "BIO_NEB_WORKER_INPUT_VERIFIED",
                {
                    "role": "reference_source",
                    "bucket": "fresh-run-bucket",
                    "object_key": "raw/reference.fa.gz",
                    "byte_count": 10,
                    "sha256": "3" * 64,
                },
            ),
        ]
        for command_id, argv in (
            ("bwa-index", ("/usr/bin/bwa", "index", "/work/reference.fa")),
            ("samtools-faidx", ("/usr/bin/samtools", "faidx", "/work/reference.fa")),
            (
                "samtools-dict",
                (
                    "/usr/bin/samtools",
                    "dict",
                    "-o",
                    "/work/reference.dict",
                    "/work/reference.fa",
                ),
            ),
        ):
            lines.extend(
                (
                    marker(
                        "BIO_NEB_WORKER_TOOL_START",
                        {"command_id": command_id, "argv": argv, "shell": False},
                    ),
                    marker(
                        "BIO_NEB_WORKER_TOOL_EXIT",
                        {"command_id": command_id, "returncode": 0},
                    ),
                )
            )
        lines.append(
            marker(
                "BIO_NEB_WORKER_REFERENCE_SUPPORT",
                {"mode": "derive", "objects": objects},
            )
        )
        parsed = parse_worker_markers("\n".join(lines))
        self.assertEqual(parsed[-1].mode, "derive")

    def test_cpu_eight_role_uploads_close_before_receipt(self):
        digest = "b" * 64
        tools = {
            "bwa": {"path": "/usr/bin/bwa", "version": "0.7.18"},
            "samtools": {"path": "/usr/bin/samtools", "version": "1.21"},
        }
        lines = [
            _marker(
                "BIO_NEB_PARABRICKS_WORKER_START",
                {
                    "interpreter": "/opt/bio-neb/venv/bin/python",
                    "interpreter_resolved": "/usr/bin/python3.10",
                    "request_schema": "bio-neb.parabricks-worker-request/7",
                },
                1,
            ),
            _marker(
                "BIO_NEB_WORKER_REQUEST_ACCEPTED",
                {"byte_count": 200, "request_sha256": digest, "worker_sha256": digest},
                2,
            ),
            _marker("BIO_NEB_WORKER_TOOLS", {"tools": tools}, 3),
        ]
        second = 4
        for command_id, path in (
            ("bwa-index", "/usr/bin/bwa"),
            ("samtools-faidx", "/usr/bin/samtools"),
            ("samtools-dict", "/usr/bin/samtools"),
        ):
            lines.append(
                _marker(
                    "BIO_NEB_WORKER_TOOL_START",
                    {"command_id": command_id, "argv": [path, "run"], "shell": False},
                    second,
                )
            )
            second += 1
            lines.append(
                _marker(
                    "BIO_NEB_WORKER_TOOL_EXIT",
                    {"command_id": command_id, "returncode": 0},
                    second,
                )
            )
            second += 1
        uploads = []
        for role in (
            "genomic_fasta",
            "fasta_fai",
            "reference_dict",
            "bwa_amb",
            "bwa_ann",
            "bwa_bwt",
            "bwa_pac",
            "bwa_sa",
        ):
            uploaded = {
                "role": role,
                "bucket": "run-bucket",
                "object_key": f"prepared/{role}",
                "byte_count": 10,
                "sha256": digest,
            }
            uploads.append(uploaded)
            lines.append(
                _marker(
                    "BIO_NEB_WORKER_UPLOAD_START",
                    {**uploaded, "multipart_chunk_bytes": 8388608},
                    second,
                )
            )
            second += 1
            lines.append(_marker("BIO_NEB_WORKER_UPLOAD_COMPLETE", uploaded, second))
            second += 1
        receipt_object = {
            "role": "worker_receipt",
            "bucket": "run-bucket",
            "object_key": "receipts/cpu.json",
            "byte_count": 50,
            "sha256": digest,
        }
        lines.extend(
            (
                _marker(
                    "BIO_NEB_WORKER_UPLOAD_START",
                    {**receipt_object, "multipart_chunk_bytes": 8388608},
                    second,
                ),
                _marker("BIO_NEB_WORKER_UPLOAD_COMPLETE", receipt_object, second + 1),
                _marker(
                    "BIO_NEB_WORKER_RECEIPT",
                    {
                        "receipt": {
                            "authority": "non-authoritative-worker-observation",
                            "operation": "prepare_reference",
                            "request_id": "cpu-request",
                            "tools": tools,
                            "prepared_uploads": uploads,
                        },
                        "receipt_object": receipt_object,
                    },
                    second + 2,
                ),
            )
        )
        observations = parse_worker_markers("\n".join(lines))
        self.assertEqual(explicit_worker_exit(observations), 0)

    def test_complete_worker_markers_prove_exit_without_provider_inference(self):
        log = _successful_log()
        markers = parse_worker_markers(log)
        self.assertEqual(explicit_worker_exit(markers), 0)
        trace = (*_provider_prefix(log), *markers)
        validate_truthful_observation_order(trace, terminal=True)
        retained = trace[4]
        self.assertIsInstance(retained, BoundedProviderLogObserved)
        self.assertEqual(retained.sha256, hashlib.sha256(log.encode()).hexdigest())

    def test_multiple_rolling_log_identities_interleave_before_terminal_worker_facts(
        self,
    ):
        log = _successful_log()
        lines = log.splitlines()
        provider = _provider_prefix(log)
        trace = (
            *provider[:3],
            bounded_log_observation("job-1", "\n".join(lines[:4])),
            provider[3],
            bounded_log_observation("job-1", "\n".join(lines[4:])),
            *parse_worker_markers(log),
        )
        validate_truthful_observation_order(trace, terminal=True)
        self.assertEqual(
            worker_marker_lines("noise\n" + lines[0] + "\nmore noise"),
            (lines[0],),
        )

    def test_rolling_log_cannot_publish_worker_facts_before_terminal_state(self):
        log = _successful_log()
        provider = _provider_prefix(log)
        premature = (
            *provider[:3],
            bounded_log_observation("job-1", log),
            *parse_worker_markers(log),
        )
        with self.assertRaisesRegex(ValueError, "startup"):
            validate_truthful_observation_order(premature, terminal=False)

    def test_provider_log_bound_and_exact_limit_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "tail bound"):
            bounded_log_observation("job-1", "\n".join("line" for _ in range(1001)))
        with self.assertRaises(ValidationError):
            BoundedProviderLogObserved(
                provider_job_id="job-1",
                available=True,
                line_count=1001,
                sha256="a" * 64,
                limit_exhausted=None,
            )
        exact_limit = bounded_log_observation(
            "job-1", "\n".join("line" for _ in range(1000))
        )
        self.assertIsNone(exact_limit.limit_exhausted)

    def test_log_window_failure_evidence_keeps_a_bounded_head_of_the_window(self):
        # The evidence retains the window text from its head -- the end an overlap
        # check reads -- up to a literal bound, and says when it was cut.
        exact = "x" * 65536
        kept = log_window_failure_evidence(
            exact, retained_marker_count=2, consecutive_partial_windows=1
        )
        self.assertEqual(
            (kept.line_count, kept.byte_count, kept.window_text_truncated),
            (1, 65536, False),
        )
        self.assertEqual(kept.sha256, hashlib.sha256(exact.encode()).hexdigest())
        self.assertEqual(kept.partial_window_ceiling, 3)
        cut = log_window_failure_evidence(
            exact + "\ntail", retained_marker_count=2, consecutive_partial_windows=1
        )
        self.assertEqual(
            (
                cut.line_count,
                cut.byte_count,
                cut.window_text,
                cut.window_text_truncated,
            ),
            (2, 65541, exact, True),
        )
        with self.assertRaises(ValidationError):
            ProviderLogWindowFailureEvidence(
                **{**kept.model_dump(), "window_text": exact + "y"}
            )

    def test_canonical_marker_grammar_and_duplicate_keys_fail_closed(self):
        line = _successful_log().splitlines()[0]
        with self.assertRaisesRegex(ValueError, "canonical"):
            parse_worker_markers(line.replace('{"interpreter"', '{ "interpreter"'))
        duplicate = line.replace(
            '"interpreter":', '"interpreter":"duplicate","interpreter":', 1
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_worker_markers(duplicate)

    def test_provider_completed_without_worker_truth_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "provider completion lacks"):
            validate_truthful_observation_order(_provider_prefix(""), terminal=True)

    def test_duplicate_worker_startup_fails_closed(self):
        lines = _successful_log().splitlines()
        with self.assertRaisesRegex(ValueError, "startup"):
            parse_worker_markers("\n".join((lines[0], lines[0], *lines[1:])))

    def test_missing_tool_exit_fails_receipt_closure(self):
        lines = _successful_log().splitlines()
        without_exit = "\n".join(line for line in lines if "TOOL_EXIT" not in line)
        with self.assertRaisesRegex(ValueError, "receipt"):
            parse_worker_markers(without_exit)

    def test_worker_marker_requires_timestamp(self):
        lines = _successful_log().splitlines()
        with self.assertRaisesRegex(ValueError, "timestamp"):
            parse_worker_markers(lines[0].split("] ", 1)[1])

    def test_worker_receipt_requires_complete_object_shape(self):
        incomplete_receipt = _successful_log().replace(
            '"bucket":"run-bucket","byte_count":50,', "", 1
        )
        with self.assertRaises(ValueError):
            parse_worker_markers(incomplete_receipt)

    def test_upload_overlap_fails_closed(self):
        lines = _successful_log().splitlines()
        overlap = list(lines)
        overlap[6], overlap[7] = overlap[7], overlap[6]
        with self.assertRaisesRegex(ValueError, "upload startup"):
            parse_worker_markers("\n".join(overlap))

    def test_worker_receipt_upload_must_be_last(self):
        lines = _successful_log().splitlines()
        receipt_not_last = [*lines[:7], *lines[9:11], *lines[7:9], lines[11]]
        with self.assertRaisesRegex(ValueError, "receipt upload"):
            parse_worker_markers("\n".join(receipt_not_last))

    def test_worker_receipt_body_must_match_ordered_uploads(self):
        lines = _successful_log().splitlines()
        drift = list(lines)
        drift[-1] = drift[-1].replace('"byte_count":100', '"byte_count":101', 1)
        with self.assertRaisesRegex(ValueError, "ordered uploads"):
            parse_worker_markers("\n".join(drift))

    def test_controlled_worker_error_retains_exact_reason(self):
        lines = _successful_log().splitlines()
        error = _marker(
            "BIO_NEB_PARABRICKS_WORKER_ERROR",
            {"reason": "prepared reference input mismatch"},
            11,
        )
        observations = parse_worker_markers("\n".join((*lines[:-3], error)))
        self.assertEqual(observations[-1].reason, "prepared reference input mismatch")

    def test_controlled_worker_error_conflicts_with_receipt(self):
        lines = _successful_log().splitlines()
        error = _marker(
            "BIO_NEB_PARABRICKS_WORKER_ERROR",
            {"reason": "prepared reference input mismatch"},
            11,
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            parse_worker_markers("\n".join((*lines, error)))

    def test_provider_terminal_cardinality_fails_closed(self):
        trace = list(_provider_prefix(_successful_log()))
        trace.insert(
            4,
            ProviderStateObserved(
                provider_job_id="job-1",
                provider_state="FAILED",
                created_at="2026-08-10T02:49:00+03:00",
                started_at="2026-08-10T02:50:00+03:00",
                finished_at="2026-08-10T02:55:00+03:00",
            ),
        )
        with self.assertRaisesRegex(ValueError, "terminal"):
            validate_truthful_observation_order(trace, terminal=False)

    def test_an_in_flight_state_may_step_back_and_is_recorded(self):
        # The provider guarantees no order among in-flight states, so a step
        # back (a recorded IMAGE_PULLING -> an earlier stage) is recorded,
        # never a fault.
        log = _successful_log()
        provider = _provider_prefix(log)
        running = provider[2]

        def state(name, started=None):
            return running.model_copy(
                update={"provider_state": name, "started_at": started}
            )

        in_flight = (
            state("PROVISIONING"),
            state("STARTING"),
            state("IMAGE_PULLING"),
            state("STARTING"),
            state("PROVISIONING"),
            state("IMAGE_PULLING"),
        )
        trace = (
            *provider[:2],
            *in_flight,
            *provider[2:],
            *parse_worker_markers(log),
        )
        validate_truthful_observation_order(trace, terminal=True)

    def test_a_state_after_a_terminal_still_fails_naming_both_states(self):
        provider = list(_provider_prefix(_successful_log()))
        followed = [
            *provider[:4],
            provider[2].model_copy(update={"finished_at": None}),
        ]
        with self.assertRaisesRegex(
            ValueError, "followed a terminal: COMPLETED -> RUNNING"
        ):
            validate_truthful_observation_order(followed, terminal=False)
        repeated = [*provider[:4], provider[3]]
        with self.assertRaisesRegex(ValueError, "COMPLETED -> COMPLETED"):
            validate_truthful_observation_order(repeated, terminal=False)

    def test_provider_created_and_started_timestamps_remain_stable(self):
        created_drift = list(_provider_prefix(_successful_log()))
        created_drift[3] = created_drift[3].model_copy(
            update={"created_at": "2026-08-10T02:49:01+03:00"}
        )
        with self.assertRaisesRegex(ValueError, "created_at"):
            validate_truthful_observation_order(created_drift, terminal=False)

        started_disappeared = list(_provider_prefix(_successful_log()))
        started_disappeared[3] = started_disappeared[3].model_copy(
            update={"started_at": None}
        )
        with self.assertRaisesRegex(ValueError, "timestamps|started_at"):
            validate_truthful_observation_order(started_disappeared, terminal=False)

        started_drift = list(_provider_prefix(_successful_log()))
        started_drift[3] = started_drift[3].model_copy(
            update={"started_at": "2026-08-10T02:50:01+03:00"}
        )
        with self.assertRaisesRegex(ValueError, "started_at"):
            validate_truthful_observation_order(started_drift, terminal=False)

    def test_provider_terminal_timestamp_is_legal_only_at_terminal_state(self):
        early_finish = list(_provider_prefix(_successful_log()))
        early_finish[2] = early_finish[2].model_copy(
            update={"finished_at": "2026-08-10T02:54:00+03:00"}
        )
        with self.assertRaisesRegex(ValueError, "nonterminal"):
            validate_truthful_observation_order(early_finish, terminal=False)


if __name__ == "__main__":
    unittest.main()
