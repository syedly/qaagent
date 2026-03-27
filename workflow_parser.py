"""
core/workflow_parser.py
───────────────────────
Reads a human-language .txt workflow file and returns a
structured list of steps the agent can iterate over.

Supports several common formats:
  • "Step N: <description>"
  • "N. <description>"
  • "N) <description>"
  • Bare lines (each non-blank line becomes a step)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TEMPLATE_CONTENT = """\
## QA Automation Workflow — TEMPLATE
## Fill in the steps below, then re-run the agent.
## Lines starting with '#' or '##' are treated as comments.

## Example — sign-up then login flow:
# Step 1: Navigate to https://example.com/register
# Step 2: Fill the registration form with random user data
# Step 3: Click the "Sign Up" button and verify account creation
# Step 4: Navigate to the login page
# Step 5: Login with the credentials created in Step 2
# Step 6: Verify successful login

Step 1: Navigate to <URL>
Step 2: <Your action here>
Step 3: <Your action here>
"""


@dataclass
class WorkflowStep:
    number: int
    raw_text: str          # original line from the file
    description: str       # cleaned instruction sent to the agent

    def __str__(self) -> str:
        return f"Step {self.number}: {self.description}"


class WorkflowParser:
    """Parse a plain-text workflow file into WorkflowStep objects."""

    # Patterns that indicate an explicit step number
    _NUMBERED = re.compile(
        r"^\s*(?:step\s*)?(\d+)\s*[:\.\)]\s*(.+)$",
        re.IGNORECASE,
    )
    # Comment / blank / section-header lines to skip
    _SKIP = re.compile(r"^\s*(?:#|##|\/\/|$)")

    def __init__(self, filepath: str = "workflow.txt") -> None:
        self.filepath = Path(filepath)

    # ── Public API ───────────────────────────────────────────────────

    def parse(self) -> list[WorkflowStep]:
        """
        Return an ordered list of WorkflowStep objects.
        Raises FileNotFoundError (after creating a template) when
        the workflow file is absent.
        """
        if not self.filepath.exists():
            self._create_template()
            raise FileNotFoundError(
                f"Workflow file '{self.filepath}' not found.\n"
                "A template has been created — please fill it in and re-run."
            )

        raw_lines = self.filepath.read_text(encoding="utf-8").splitlines()
        return self._extract_steps(raw_lines)

    # ── Private helpers ──────────────────────────────────────────────

    def _extract_steps(self, lines: list[str]) -> list[WorkflowStep]:
        steps: list[WorkflowStep] = []
        auto_counter = 0

        for line in lines:
            # Skip blank lines, comments, section headers
            if self._SKIP.match(line):
                continue

            m = self._NUMBERED.match(line)
            if m:
                number = int(m.group(1))
                description = m.group(2).strip()
            else:
                # Treat any other non-blank line as the next step
                auto_counter += 1
                number = auto_counter
                description = line.strip()

            if not description:
                continue

            steps.append(
                WorkflowStep(
                    number=number,
                    raw_text=line.strip(),
                    description=description,
                )
            )

        # Re-number sequentially so gaps don't confuse the agent
        for idx, step in enumerate(steps, start=1):
            step.number = idx

        return steps

    def _create_template(self) -> None:
        self.filepath.write_text(TEMPLATE_CONTENT, encoding="utf-8")
        print(f"[WorkflowParser] Template created at: {self.filepath.resolve()}")
