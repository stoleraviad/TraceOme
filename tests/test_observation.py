"""The public polling seam owns time and a bounded window, not recipe closure."""

import unittest

from bio_neb.failure import ProviderError
from bio_neb.observation import LiveEmitter, ObservationPolicy, observe


class ObservationTests(unittest.TestCase):
    def test_timeout_obeys_call_policy_without_cancelling(self):
        now = [0.0]
        samples = []

        def pause(seconds):
            now[0] += seconds

        def source():
            samples.append(now[0])
            return "running"

        with self.assertRaisesRegex(ProviderError, "observation_timeout") as raised:
            observe(
                source,
                ObservationPolicy(
                    3, 5, 1, lambda _: None, clock=lambda: now[0], pause=pause
                ),
            )
        self.assertEqual(raised.exception.failure_class, "exhausted")
        self.assertEqual(samples, [0, 3, 5])

    def test_display_errors_do_not_mask_closure_but_programming_faults_escape(self):
        def broken(_):
            raise RuntimeError("display broke")

        emitter = LiveEmitter(broken)
        self.assertEqual(
            observe(
                lambda: "done",
                ObservationPolicy(1, 0, 1, lambda _: False, liveness=emitter),
            ),
            False,
        )
        emitter("later heartbeat")
        self.assertEqual(emitter.errors, ["RuntimeError"])
        with self.assertRaisesRegex(RuntimeError, "display broke"):
            observe(lambda: "done", ObservationPolicy(1, 0, 1, broken))

    def test_unbounded_time_policy_is_rejected(self):
        for interval, limit in (
            (float("nan"), 1),
            (1, float("inf")),
            (1, float("nan")),
        ):
            with (
                self.subTest(interval=interval, limit=limit),
                self.assertRaises(ValueError),
            ):
                ObservationPolicy(interval, limit, 1, lambda _: None)

    def test_closure_receives_a_bounded_window_and_liveness_stays_ephemeral(self):
        samples = iter(("starting", "running", "complete"))
        windows, live, waits = [], [], []

        def close(window):
            windows.append(window)
            return "verified-output" if window[-1] == "complete" else None

        result = observe(
            lambda: next(samples),
            ObservationPolicy(
                interval=2,
                max_wait=20,
                window=2,
                closure=close,
                liveness=live.append,
                pause=waits.append,
                clock=lambda: 0,
            ),
        )
        self.assertEqual(result, "verified-output")
        self.assertEqual(windows[-1], ("running", "complete"))
        self.assertEqual(live, ["starting", "running", "complete"])
        self.assertEqual(waits, [2, 2])
