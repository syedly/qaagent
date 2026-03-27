"""
core/report_generator.py
────────────────────────
Generates a human-readable HTML + plain-text test report
at the end of a run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from state_manager import AgentStateManager, StepResult


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Plain-text summary (shown in terminal) ───────────────────────────

def build_text_report(state: AgentStateManager) -> str:
    summary = state.get_summary()
    lines = [
        "=" * 60,
        "  QA AGENT - TEST REPORT",
        f"  Run date : {state.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Workflow : {state.workflow_file}",
        "=" * 60,
        f"  Total steps : {summary['total_steps']}",
        f"  Passed      : {summary['passed']}",
        f"  Failed      : {summary['failed']}",
        f"  Skipped     : {summary['skipped']}",
        f"  Success rate: {summary['success_rate']}",
        f"  Duration    : {summary['duration_seconds']}s",
        "=" * 60,
        "  STEP DETAILS",
        "-" * 60,
    ]
    for r in state.step_results:
        icon = "[PASS]" if r.status == "passed" else "[FAIL]"
        lines.append(f"  {icon} Step {r.step_number}: {r.description}")
        lines.append(f"     Status      : {r.status.upper()}")
        lines.append(f"     Observation : {r.observation[:200]}")
        if r.screenshot_path:
            lines.append(f"     Screenshot  : {r.screenshot_path}")
        if r.error:
            lines.append(f"     Error       : {r.error[:300]}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── HTML report ──────────────────────────────────────────────────────

def build_html_report(state: AgentStateManager) -> str:
    summary = state.get_summary()
    rows = ""
    for r in state.step_results:
        colour = "#d4edda" if r.status == "passed" else "#f8d7da"
        icon   = "PASS" if r.status == "passed" else "FAIL"
        shot   = f'<a href="../{r.screenshot_path}">view</a>' if r.screenshot_path else "-"
        err    = f'<code>{r.error[:300]}</code>' if r.error else "-"
        rows += f"""
        <tr style="background:{colour}">
          <td>{r.step_number}</td>
          <td>{r.description}</td>
          <td>{icon} {r.status.upper()}</td>
          <td style="max-width:400px;word-break:break-word">{r.observation[:300]}</td>
          <td>{shot}</td>
          <td>{err}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QA Agent Report - {state.start_time.strftime('%Y-%m-%d %H:%M')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #333; }}
    h1   {{ color: #4a4a8a; }}
    .summary {{ background: #eef; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th    {{ background: #4a4a8a; color: white; padding: .6rem; text-align: left; }}
    td    {{ padding: .5rem; border-bottom: 1px solid #ddd; vertical-align: top; }}
    code  {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: .85em; }}
  </style>
</head>
<body>
  <h1>QA Agent - Test Report</h1>
  <div class="summary">
    <strong>Workflow:</strong> {state.workflow_file}<br>
    <strong>Run date:</strong> {state.start_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
    <strong>Duration:</strong> {summary['duration_seconds']}s &nbsp;|&nbsp;
    <strong>Total:</strong> {summary['total_steps']} &nbsp;|&nbsp;
    <strong>Passed:</strong> {summary['passed']} &nbsp;|&nbsp;
    <strong>Failed:</strong> {summary['failed']} &nbsp;|&nbsp;
    <strong>Success rate:</strong> {summary['success_rate']}
  </div>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Description</th><th>Status</th>
        <th>Observation</th><th>Screenshot</th><th>Error</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""


def save_reports(state: AgentStateManager) -> tuple[str, str]:
    """Write both reports and return (text_path, html_path)."""
    ts = state.start_time.strftime("%Y%m%d_%H%M%S")
    txt_path  = REPORT_DIR / f"report_{ts}.txt"
    html_path = REPORT_DIR / f"report_{ts}.html"

    txt_path.write_text(build_text_report(state),  encoding="utf-8")
    html_path.write_text(build_html_report(state), encoding="utf-8")

    return str(txt_path), str(html_path)
