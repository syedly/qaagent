"""
tools/browser_tools.py
──────────────────────
LangChain-compatible async tools that wrap Playwright.

Every tool receives the shared AgentStateManager so it can
read / write credentials and page state without global variables.

Self-healing strategy
─────────────────────
Each action tries the primary selector first; on failure it
performs a "Page Re-Scan" using accessible roles, aria-labels,
placeholder text, and visible text content before giving up.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Type

from faker import Faker
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from playwright.async_api import Page, TimeoutError as PWTimeout

from state_manager import AgentStateManager

fake = Faker()
SCREENSHOT_DIR = Path("reports/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper: safe screenshot ──────────────────────────────────────────

async def _take_screenshot(page: Page, label: str = "screenshot") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOT_DIR / f"{label}_{ts}.png"
    await page.screenshot(path=str(path), full_page=True)
    return str(path)


# ── Helper: self-healing element finder ─────────────────────────────

async def _find_element(page: Page, selector: str, description: str = "") -> Any:
    """
    Try primary CSS/XPath selector, then fall back to aria / text
    alternatives before raising.
    """
    strategies = [
        selector,
        f"[aria-label*='{description}' i]",
        f"[placeholder*='{description}' i]",
        f"text='{description}'",
        f"role=button[name*='{description}' i]",
    ]
    for strat in strategies:
        try:
            el = page.locator(strat).first
            await el.wait_for(state="visible", timeout=4_000)
            return el
        except Exception:
            continue
    raise RuntimeError(
        f"Self-heal exhausted - could not locate element. "
        f"Selector='{selector}', description='{description}'"
    )


# ════════════════════════════════════════════════════════════════════
#  Tool input schemas (Pydantic v2)
# ════════════════════════════════════════════════════════════════════

class NavigateInput(BaseModel):
    url: str = Field(description="Full URL to navigate to")

class ClickInput(BaseModel):
    selector: str = Field(description="CSS selector, XPath, or descriptive text")
    description: str = Field(default="", description="Human description for self-healing fallback")

class TypeInput(BaseModel):
    selector: str = Field(description="CSS selector or descriptive text for the input field")
    text: str = Field(description="Text to type into the field")
    description: str = Field(default="", description="Human description for self-healing fallback")
    clear_first: bool = Field(default=True, description="Clear field before typing")

class SelectInput(BaseModel):
    selector: str = Field(description="CSS selector for the <select> element")
    value: str = Field(description="Option value or visible text to select")

class WaitInput(BaseModel):
    seconds: float = Field(default=2.0, description="Seconds to wait")

class VerifyInput(BaseModel):
    check_type: str = Field(
        description="One of: url_contains | text_visible | element_exists | title_contains"
    )
    value: str = Field(description="String to check for")

class GetDOMInput(BaseModel):
    summarize: bool = Field(default=True, description="Return a compact summary instead of raw HTML")

class GenerateDataInput(BaseModel):
    data_type: str = Field(
        description="One of: email | username | password | full_name | phone | address"
    )
    store_as: str = Field(default="", description="Key to store in agent state (e.g. 'email')")

class ScrollInput(BaseModel):
    direction: str = Field(default="down", description="'up' or 'down'")
    pixels: int = Field(default=500)

class ScreenshotInput(BaseModel):
    label: str = Field(default="step", description="Label for the screenshot filename")


# ════════════════════════════════════════════════════════════════════
#  Tool factory — builds tool instances bound to a state manager
# ════════════════════════════════════════════════════════════════════

def build_tools(state: AgentStateManager) -> list[BaseTool]:
    """Return all browser tools wired to the given AgentStateManager."""

    # ── Navigate ─────────────────────────────────────────────────────
    class NavigateTool(BaseTool):
        name: str = "navigate_to_url"
        description: str = (
            "Navigate the browser to a URL. "
            "Input: {url: str}"
        )
        args_schema: Type[BaseModel] = NavigateInput

        def _run(self, url: str) -> str:  # pragma: no cover
            raise NotImplementedError("Use async only")

        async def _arun(self, url: str) -> str:
            page = state.page
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            state.current_url = page.url
            state.last_page_title = await page.title()
            return f"Navigated to {page.url} | Title: {state.last_page_title}"

    # ── Click ─────────────────────────────────────────────────────────
    class ClickTool(BaseTool):
        name: str = "click_element"
        description: str = (
            "Click a page element. Tries CSS/XPath first, then self-heals with "
            "aria-label, placeholder, and text fallbacks. "
            "Input: {selector: str, description: str}"
        )
        args_schema: Type[BaseModel] = ClickInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, selector: str, description: str = "") -> str:
            page = state.page
            try:
                el = await _find_element(page, selector, description)
                await el.scroll_into_view_if_needed()
                await el.click(timeout=8_000)
                await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                state.current_url = page.url
                return f"Clicked '{selector}'. Current URL: {state.current_url}"
            except Exception as exc:
                path = await _take_screenshot(page, "click_failure")
                return f"ERROR clicking '{selector}': {exc}. Screenshot: {path}"

    # ── Type ──────────────────────────────────────────────────────────
    class TypeTool(BaseTool):
        name: str = "type_text"
        description: str = (
            "Type text into an input field. Clears it first by default. "
            "Input: {selector: str, text: str, description: str, clear_first: bool}"
        )
        args_schema: Type[BaseModel] = TypeInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(
            self,
            selector: str,
            text: str,
            description: str = "",
            clear_first: bool = True,
        ) -> str:
            page = state.page
            try:
                el = await _find_element(page, selector, description)
                await el.scroll_into_view_if_needed()
                if clear_first:
                    await el.fill("")
                await el.type(text, delay=40)
                return f"Typed '{text}' into '{selector}'"
            except Exception as exc:
                path = await _take_screenshot(page, "type_failure")
                return f"ERROR typing into '{selector}': {exc}. Screenshot: {path}"

    # ── Select ────────────────────────────────────────────────────────
    class SelectTool(BaseTool):
        name: str = "select_option"
        description: str = (
            "Select an option from a <select> dropdown. "
            "Input: {selector: str, value: str}"
        )
        args_schema: Type[BaseModel] = SelectInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, selector: str, value: str) -> str:
            page = state.page
            try:
                await page.select_option(selector, label=value, timeout=6_000)
                return f"Selected '{value}' in '{selector}'"
            except Exception:
                try:
                    await page.select_option(selector, value=value, timeout=6_000)
                    return f"Selected value='{value}' in '{selector}'"
                except Exception as exc:
                    return f"ERROR selecting option: {exc}"

    # ── Wait ──────────────────────────────────────────────────────────
    class WaitTool(BaseTool):
        name: str = "wait_seconds"
        description: str = "Pause execution for N seconds. Input: {seconds: float}"
        args_schema: Type[BaseModel] = WaitInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, seconds: float = 2.0) -> str:
            await asyncio.sleep(seconds)
            return f"Waited {seconds}s"

    # ── Verify ────────────────────────────────────────────────────────
    class VerifyTool(BaseTool):
        name: str = "verify_condition"
        description: str = (
            "Assert a condition on the current page. "
            "check_type options: url_contains | text_visible | element_exists | title_contains. "
            "Input: {check_type: str, value: str}"
        )
        args_schema: Type[BaseModel] = VerifyInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, check_type: str, value: str) -> str:
            page = state.page
            try:
                if check_type == "url_contains":
                    current = page.url
                    ok = value.lower() in current.lower()
                    return f"{'PASS' if ok else 'FAIL'}: URL '{current}' {'contains' if ok else 'does NOT contain'} '{value}'"

                elif check_type == "text_visible":
                    locator = page.get_by_text(value, exact=False)
                    await locator.first.wait_for(state="visible", timeout=6_000)
                    return f"PASS: Text '{value}' is visible on the page"

                elif check_type == "element_exists":
                    el = page.locator(value).first
                    await el.wait_for(state="attached", timeout=6_000)
                    return f"PASS: Element '{value}' exists in DOM"

                elif check_type == "title_contains":
                    title = await page.title()
                    ok = value.lower() in title.lower()
                    return f"{'PASS' if ok else 'FAIL'}: Title '{title}' {'contains' if ok else 'does NOT contain'} '{value}'"

                else:
                    return f"Unknown check_type '{check_type}'"
            except Exception as exc:
                path = await _take_screenshot(page, "verify_failure")
                return f"FAIL - verification error: {exc}. Screenshot: {path}"

    # ── Get DOM ───────────────────────────────────────────────────────
    class GetDOMTool(BaseTool):
        name: str = "get_page_dom"
        description: str = (
            "Inspect the current page. Returns a compact summary of inputs, "
            "buttons, links, and headings to help choose correct selectors. "
            "Input: {summarize: bool}"
        )
        args_schema: Type[BaseModel] = GetDOMInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, summarize: bool = True) -> str:
            page = state.page
            if not summarize:
                html = await page.content()
                return html[:8_000]  # cap to avoid context overflow

            # Extract a structured summary
            data = await page.evaluate("""() => {
                const get = (sel) => [...document.querySelectorAll(sel)].map(el => ({
                    tag: el.tagName.toLowerCase(),
                    id: el.id || null,
                    name: el.name || null,
                    type: el.type || null,
                    placeholder: el.placeholder || null,
                    ariaLabel: el.getAttribute('aria-label'),
                    text: (el.innerText || el.value || '').slice(0, 80).trim(),
                    selector: el.id ? '#' + el.id : (el.name ? `[name="${el.name}"]` : null),
                }));
                return {
                    url: location.href,
                    title: document.title,
                    inputs: get('input, textarea, select'),
                    buttons: get('button, [type="submit"], [role="button"]'),
                    links: get('a[href]').slice(0, 20),
                    headings: get('h1,h2,h3').slice(0, 10),
                };
            }""")
            return json.dumps(data, indent=2)[:6_000]

    # ── Generate Data ─────────────────────────────────────────────────
    class GenerateDataTool(BaseTool):
        name: str = "generate_mock_data"
        description: str = (
            "Generate realistic mock data for form filling. "
            "data_type: email | username | password | full_name | phone | address. "
            "Optionally store with store_as key for reuse in later steps. "
            "Input: {data_type: str, store_as: str}"
        )
        args_schema: Type[BaseModel] = GenerateDataInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, data_type: str, store_as: str = "") -> str:
            generators: dict[str, Any] = {
                "email": lambda: fake.email(),
                "username": lambda: fake.user_name() + str(fake.random_int(10, 999)),
                "password": lambda: "Test@" + fake.password(
                    length=10, special_chars=True, digits=True,
                    upper_case=True, lower_case=True
                ),
                "full_name": fake.name,
                "phone": fake.phone_number,
                "address": fake.address,
            }
            fn = generators.get(data_type.lower())
            if not fn:
                return f"Unknown data_type '{data_type}'. Choose from: {list(generators)}"

            value = fn()
            if store_as:
                await state.store_credential(store_as, value)
                await state.store_generated_data(store_as, value)
            return f"Generated {data_type}: '{value}'" + (f" (stored as '{store_as}')" if store_as else "")

    # ── Recall Credential ─────────────────────────────────────────────
    class RecallCredentialTool(BaseTool):
        name: str = "recall_stored_value"
        description: str = (
            "Recall a value that was stored earlier (e.g. email or password "
            "from the sign-up step). Input: key name as plain string."
        )

        def _run(self, key: str) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, key: str) -> str:  # type: ignore[override]
            value = await state.get_credential(key)
            if value:
                return f"Recalled '{key}': {value}"
            all_keys = list(state.credentials.keys()) + list(state.generated_data.keys())
            return f"No stored value for key '{key}'. Available keys: {all_keys}"

    # ── Scroll ────────────────────────────────────────────────────────
    class ScrollTool(BaseTool):
        name: str = "scroll_page"
        description: str = "Scroll the page. Input: {direction: 'up'|'down', pixels: int}"
        args_schema: Type[BaseModel] = ScrollInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, direction: str = "down", pixels: int = 500) -> str:
            page = state.page
            y = pixels if direction == "down" else -pixels
            await page.evaluate(f"window.scrollBy(0, {y})")
            return f"Scrolled {direction} by {pixels}px"

    # ── Screenshot ────────────────────────────────────────────────────
    class ScreenshotTool(BaseTool):
        name: str = "take_screenshot"
        description: str = "Capture a screenshot of the current page. Input: {label: str}"
        args_schema: Type[BaseModel] = ScreenshotInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, label: str = "step") -> str:
            page = state.page
            path = await _take_screenshot(page, label)
            return f"Screenshot saved: {path}"

    # ── Assemble and return ───────────────────────────────────────────
    return [
        NavigateTool(),
        ClickTool(),
        TypeTool(),
        SelectTool(),
        WaitTool(),
        VerifyTool(),
        GetDOMTool(),
        GenerateDataTool(),
        RecallCredentialTool(),
        ScrollTool(),
        ScreenshotTool(),
    ]
