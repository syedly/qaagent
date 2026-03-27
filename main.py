"""
main.py
───────
Entry point for the Self-Healing QA Automation Agent.

Usage
─────
  python main.py                          # reads workflow.txt in cwd
  python main.py --workflow my_flow.txt   # custom workflow file
  python main.py --headless               # headless mode (no browser window)

Environment
───────────
  OPENAI_API_KEY  — required (or set in .env)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright

import logger
from agent_brain import AgentBrain
from report_generator import save_reports, build_text_report
from state_manager import AgentStateManager
from workflow_parser import WorkflowParser

load_dotenv()


# ════════════════════════════════════════════════════════════════════
#  Browser bootstrap
# ════════════════════════════════════════════════════════════════════

async def _launch_browser(state: AgentStateManager, headless: bool) -> None:
    pw = await async_playwright().start()
    state._playwright = pw

    browser = await pw.chromium.launch(
        headless=headless,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    )
    context = await browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        java_script_enabled=True,
        accept_downloads=True,
    )
    page = await context.new_page()

    state.browser = browser
    state.context = context
    state.page    = page

    logger.log_info(
        f"Browser launched (headless={headless}) | "
        f"Chromium {browser.version}"
    )


async def _close_browser(state: AgentStateManager) -> None:
    try:
        if state.context:
            await state.context.close()
        if state.browser:
            await state.browser.close()
        if hasattr(state, "_playwright"):
            await state._playwright.stop()
    except Exception as exc:
        logger.log_error(f"Error during browser teardown: {exc}")


# ════════════════════════════════════════════════════════════════════
#  Main orchestration loop
# ════════════════════════════════════════════════════════════════════

async def run(workflow_file: str, headless: bool) -> None:
    logger.log_banner("QA AUTOMATION AGENT - STARTING UP")

    # ── API key check ────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY", "")
    if not api_key:
        logger.log_error(
            "OPENAI_API_KEY not found.\n"
            "Set it in your environment or create a .env file:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "You can also use OPEN_AI_API_KEY for backward compatibility."
        )
        sys.exit(1)

    # ── Parse workflow ───────────────────────────────────────────────
    parser = WorkflowParser(workflow_file)
    try:
        steps = parser.parse()
    except FileNotFoundError as exc:
        logger.log_error(str(exc))
        sys.exit(1)

    if not steps:
        logger.log_error(f"No steps found in '{workflow_file}'. Please add instructions.")
        sys.exit(1)

    logger.log_info(f"Workflow: '{workflow_file}' - {len(steps)} steps loaded")
    for s in steps:
        logger.log_info(f"  {s}")

    # ── Initialise state ─────────────────────────────────────────────
    state = AgentStateManager()
    state.total_steps   = len(steps)
    state.workflow_file = workflow_file

    # ── Launch browser ───────────────────────────────────────────────
    await _launch_browser(state, headless=headless)

    # ── Build agent ──────────────────────────────────────────────────
    brain = AgentBrain(state=state, openai_api_key=api_key)
    brain.setup()

    logger.log_banner("EXECUTING WORKFLOW STEPS")

    # ── Sequential step execution with asyncio.TaskGroup (Python 3.11+)
    # Each step runs serially; TaskGroup gives us clean error propagation.
    failed_steps: list[int] = []

    for step in steps:
        state.current_step = step.number
        result = await brain.execute_step(step)
        if result.status == "failed":
            failed_steps.append(step.number)
            # Decide whether to abort or continue on failure
            # (configurable — currently we continue to gather full report)

    # ── Final report ─────────────────────────────────────────────────
    logger.log_banner("TEST RUN COMPLETE")

    txt_path, html_path = save_reports(state)
    report_text = build_text_report(state)

    logger.log_report(report_text)
    logger.log_info(f"Reports saved:\n  TXT: {txt_path}\n  HTML: {html_path}")
    logger.save_log()

    # ── Teardown ─────────────────────────────────────────────────────
    await _close_browser(state)

    if failed_steps:
        logger.log_error(
            f"Run finished with {len(failed_steps)} failed step(s): {failed_steps}"
        )
        sys.exit(1)
    else:
        logger.log_info("All steps passed!")


# ════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Self-Healing QA Automation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--workflow", "-w",
        default="workflow.txt",
        help="Path to the workflow .txt file (default: workflow.txt)",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: visible)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(workflow_file=args.workflow, headless=args.headless))
