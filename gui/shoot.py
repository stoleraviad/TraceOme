"""Screenshot every variant URL of a running prototype (python shoot.py [base url] [out dir]).

Needs Playwright with a Chromium (CHROMIUM=<path to chrome> if Playwright cannot find one).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8770"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "screenshots")

SHOTS = [
    ("01-plan-question-offer", "/?at=80", None),
    ("02-plan-question-form", "/?at=80&decision=form", None),
    ("03-approval-offer", "/?at=166", None),
    ("04-execution-running-story", "/?at=171", None),
    ("05-execution-running-timeline", "/?at=171&journey=timeline", None),
    ("06-execution-running-passes", "/?at=171&journey=passes", None),
    ("07-investigation-review1-cards", "/?at=257", None),
    ("08-investigation-review1-table-tabs", "/?at=257&targets=table&nav=tabs&left=hidden", None),
    ("09-investigation-now-offer", "/?at=262", None),
    ("10-investigation-now-form", "/?at=262&decision=form", None),
    ("11-results", "/?view=Results", None),
    ("11b-results-table", "/?view=Results&findings=table", None),
    ("12-question", "/?at=262&view=Question", None),
    ("13-dry-run-approve", "/?at=166", "text=Approve and start"),
    ("14-run-details", "/?at=262", "text=Run details"),
    ("15-event-dialog", "/?at=262", "text=rs4149056: ambiguous (2 targets)"),
    ("16-confirm-stop", "/?at=80", "button:has-text('Stop')"),
    ("17-replay-playing-silence", "/?at=169&play=1", "wait:6000"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        for name, path, click in SHOTS:
            page.goto(BASE + path)
            page.wait_for_timeout(1500)
            if click and click.startswith("wait:"):
                page.wait_for_timeout(int(click[5:]))
            elif click:
                page.locator(click).first.click()
                page.wait_for_timeout(800)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(name)
        browser.close()


if __name__ == "__main__":
    main()
