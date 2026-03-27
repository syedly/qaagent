"""
core/logger.py
──────────────
Rich-powered logger that prints the agent's Thought / Action /
Observation cycle in real-time with colour-coded panels.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme


_THEME = Theme({
    "thought":     "bold cyan",
    "action":      "bold yellow",
    "observation": "bold green",
    "error":       "bold red",
    "info":        "bold blue",
    "step":        "bold magenta",
    "pass":        "bold green",
    "fail":        "bold red",
})

_console = Console(theme=_THEME, highlight=False)
_log_lines: list[str] = []          # kept for the final text report


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Public helpers ───────────────────────────────────────────────────

def log_step_start(number: int, total: int, description: str) -> None:
    _console.print()
    _console.print(Rule(
        f"[step]  Step {number}/{total}  [/step]",
        style="magenta",
    ))
    msg = f"Step: {description}"
    _console.print(Panel(msg, title=f"[step]Step {number}[/step]", border_style="magenta"))
    _store(f"\n=== Step {number}/{total}: {description} ===")


def log_thought(thought: str) -> None:
    _console.print(
        Panel(thought, title=f"[thought]Thought [{_ts()}][/thought]", border_style="cyan")
    )
    _store(f"[THOUGHT] {thought}")


def log_action(tool: str, inputs: str) -> None:
    body = f"Tool  : [action]{tool}[/action]\nInput : {inputs}"
    _console.print(Panel(body, title=f"[action]Action [{_ts()}][/action]", border_style="yellow"))
    _store(f"[ACTION] tool={tool} | input={inputs}")


def log_observation(obs: str) -> None:
    # Truncate very long observations for console readability
    display = obs if len(obs) < 600 else obs[:600] + "..."
    _console.print(
        Panel(display, title=f"[observation]Observation [{_ts()}][/observation]", border_style="green")
    )
    _store(f"[OBSERVATION] {obs}")


def log_step_pass(number: int) -> None:
    _console.print(f"  [pass][PASS] Step {number} PASSED[/pass]")
    _store(f"[PASS] Step {number}")


def log_step_fail(number: int, reason: str) -> None:
    _console.print(f"  [fail][FAIL] Step {number} FAILED - {reason}[/fail]")
    _store(f"[FAIL] Step {number}: {reason}")


def log_error(msg: str) -> None:
    _console.print(Panel(msg, title="[error]ERROR[/error]", border_style="red"))
    _store(f"[ERROR] {msg}")


def log_info(msg: str) -> None:
    _console.print(f"  [info][INFO] {msg}[/info]")
    _store(f"[INFO] {msg}")


def log_banner(title: str) -> None:
    _console.print()
    _console.print(Rule(f"[info]{title}[/info]", style="blue"))


def log_report(report: str) -> None:
    _console.print(Panel(report, title="[step]TEST REPORT[/step]", border_style="magenta"))
    _store(f"\n{'='*60}\nTEST REPORT\n{'='*60}\n{report}")


def save_log(path: str = "reports/run.log") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(_log_lines), encoding="utf-8")
    _console.print(f"\n  [info][INFO] Full log saved -> {path}[/info]")


def _store(line: str) -> None:
    _log_lines.append(f"[{_ts()}] {line}")
