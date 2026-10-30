"""The review flag: at most eight tickets at a time in their review phase across every lane (#391).

Each test uses its own queue directory. The flag's operations are tested directly; one test
drives `testing review-flag` through the runner's `main` for its printed line and status.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bio_neb import locations
from bio_neb.devtools import review_flag
from bio_neb.devtools import testing as testing_operation


class ReviewFlagTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.queue = Path(directory.name) / "test-queue"
        environment = patch.dict(
            os.environ, {locations.TEST_QUEUE_OVERRIDE: str(self.queue)}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def claim(self, ticket: int, worker: str = "w1") -> tuple[int, dict]:
        return review_flag.claim(
            self.queue,
            ticket=ticket,
            map_id=4,
            worker=worker,
            orchestrator="orchestrator-4",
        )

    def fill(self, count: int = review_flag.CAPACITY) -> list[int]:
        """Claim tickets 391, 390, ... for workers w1, w2, ...; by default every place."""
        tickets = [391 - index for index in range(count)]
        for index, ticket in enumerate(tickets):
            self.claim(ticket, worker=f"w{index + 1}")
        return tickets

    def test_the_runner_command_prints_the_flag_and_exits_with_its_status(self):
        def command(*arguments: str) -> tuple[int, dict]:
            output = io.StringIO()
            with redirect_stdout(output):
                returned = testing_operation.main(["review-flag", *arguments])
            return returned, json.loads(output.getvalue())["review_flag"]

        claim = ["claim", "--map", "4", "--worker", "w", "--orchestrator", "o"]
        for ticket in range(391, 391 - review_flag.CAPACITY, -1):
            self.assertEqual(command(*claim, "--ticket", str(ticket))[0], 0)
        returned, flag = command(*claim, "--ticket", "300")

        self.assertEqual(returned, review_flag.TAKEN_EXIT)
        self.assertEqual(flag["result"], "taken")
        self.assertEqual(flag["holders"][0]["orchestrator"], "o")

    def test_a_free_flag_is_claimed_and_names_who_to_ping(self):
        returned, flag = self.claim(391)

        self.assertEqual(returned, 0)
        self.assertEqual(flag["result"], "claimed")
        (holder,) = flag["holders"]
        self.assertEqual(
            {key: holder[key] for key in ("ticket", "map", "worker", "orchestrator")},
            {"ticket": 391, "map": 4, "worker": "w1", "orchestrator": "orchestrator-4"},
        )
        self.assertIn("claimed_at", holder)

    def test_a_taken_flag_refuses_another_ticket_and_names_its_holders(self):
        tickets = self.fill()
        returned, flag = self.claim(300, worker="w-next")

        self.assertEqual(returned, review_flag.TAKEN_EXIT)
        self.assertEqual(flag["result"], "taken")
        self.assertEqual(
            [(holder["ticket"], holder["worker"]) for holder in flag["holders"]],
            [(ticket, f"w{index + 1}") for index, ticket in enumerate(tickets)],
        )

    def test_the_holding_ticket_claims_again_without_a_second_flag(self):
        self.claim(391)
        returned, flag = self.claim(391, worker="w1-resumed")

        self.assertEqual(returned, 0)
        self.assertEqual(flag["result"], "held")
        self.assertEqual([holder["worker"] for holder in flag["holders"]], ["w1"])

    def test_release_frees_the_flag_for_the_next_ticket(self):
        self.fill()
        self.assertEqual(self.claim(300)[1]["result"], "taken")
        returned, flag = review_flag.release(self.queue, ticket=391)
        self.assertEqual((returned, flag["result"]), (0, "released"))
        self.assertEqual(self.claim(300)[1]["result"], "claimed")

    def test_release_refuses_a_ticket_that_holds_no_flag(self):
        self.claim(391)
        returned, flag = review_flag.release(self.queue, ticket=389)

        self.assertEqual(returned, 2)
        self.assertEqual(flag["result"], "not_held")
        self.assertEqual([holder["ticket"] for holder in flag["holders"]], [391])

    def test_a_stale_release_keeps_its_reason_in_the_flag_history(self):
        self.claim(391)
        returned, flag = review_flag.release(
            self.queue, ticket=391, stale="worker gone; orchestrator pinged"
        )

        self.assertEqual((returned, flag["result"]), (0, "released"))
        history = [
            json.loads(line)
            for line in (self.queue / "review-flag-history.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [(entry["event"], entry["ticket"]) for entry in history],
            [("claimed", 391), ("released", 391)],
        )
        self.assertEqual(history[-1]["stale"], "worker gone; orchestrator pinged")

    def test_show_lists_the_holders_and_changes_nothing(self):
        self.assertEqual(
            review_flag.show(self.queue), (0, {"result": "free", "holders": []})
        )
        tickets = self.fill(review_flag.CAPACITY - 1)
        self.assertEqual(review_flag.show(self.queue)[1]["result"], "free")
        tickets.append(300)
        self.claim(300)
        returned, flag = review_flag.show(self.queue)

        self.assertEqual((returned, flag["result"]), (0, "taken"))
        self.assertEqual([holder["ticket"] for holder in flag["holders"]], tickets)

    def test_the_flag_count_is_the_eight_review_capacity(self):
        self.assertEqual(review_flag.CAPACITY, 8)
        self.fill(7)
        self.assertEqual(self.claim(300, worker="w8")[1]["result"], "claimed")
        self.assertEqual(self.claim(299, worker="w9")[1]["result"], "taken")


if __name__ == "__main__":
    unittest.main()
