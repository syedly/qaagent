"""
core/state_manager.py
─────────────────────
Centralised, thread-safe state store for the QA agent.
Holds credentials created during Sign-Up so they can be
reused during Login, tracks step results, and owns the
shared Playwright page reference.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StepResult:
    step_number: int
    description: str
    status: str          # "passed" | "failed" | "skipped"
    observation: str
    screenshot_path: str | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class AgentStateManager:
    """
    Single source of truth for everything the agent knows
    across the lifetime of a test run.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # ── Playwright handles (set once browser is up) ─────────────
        self.page: Any | None = None          # playwright Page
        self.browser: Any | None = None       # playwright Browser
        self.context: Any | None = None       # playwright BrowserContext

        # ── Credentials / generated data ────────────────────────────
        self.credentials: dict[str, str] = {}
        # e.g. {"email": "...", "username": "...", "password": "..."}

        self.generated_data: dict[str, Any] = {}
        # Any mock data the agent created on the fly

        # ── Execution history ────────────────────────────────────────
        self.step_results: list[StepResult] = []
        self.current_step: int = 0
        self.total_steps: int = 0

        # ── Misc runtime info ────────────────────────────────────────
        self.current_url: str = ""
        self.last_page_title: str = ""
        self.start_time: datetime = datetime.now()
        self.workflow_file: str = "workflow.txt"

    # ── Credential helpers ───────────────────────────────────────────

    async def store_credential(self, key: str, value: str) -> None:
        async with self._lock:
            self.credentials[key] = value

    async def get_credential(self, key: str) -> str | None:
        async with self._lock:
            return self.credentials.get(key)

    async def store_generated_data(self, key: str, value: Any) -> None:
        async with self._lock:
            self.generated_data[key] = value

    # ── Step result helpers ──────────────────────────────────────────

    async def record_step(self, result: StepResult) -> None:
        async with self._lock:
            self.step_results.append(result)

    def get_summary(self) -> dict[str, Any]:
        total = len(self.step_results)
        passed = sum(1 for r in self.step_results if r.status == "passed")
        failed = sum(1 for r in self.step_results if r.status == "failed")
        skipped = total - passed - failed
        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_steps": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_seconds": round(duration, 2),
            "success_rate": f"{(passed / total * 100):.1f}%" if total else "N/A",
        }
