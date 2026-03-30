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
    def _quoted_text_candidates(value: str) -> list[str]:
        return re.findall(r"['\"]([^'\"]{2,})['\"]", value or "")

    text_candidates: list[str] = []
    for source in (selector, description):
        for candidate in _quoted_text_candidates(source):
            cleaned = re.sub(r"\s+(button|link|field|input|form)\b.*$", "", candidate, flags=re.I).strip()
            if cleaned and cleaned not in text_candidates:
                text_candidates.append(cleaned)

    if description:
        cleaned_description = re.sub(
            r"\b(button|link|field|input|form|on the .*|for the .*)\b.*$",
            "",
            description,
            flags=re.I,
        ).strip()
        if cleaned_description and cleaned_description not in text_candidates:
            text_candidates.append(cleaned_description)

    strategies = [selector]
    for text in text_candidates:
        escaped = text.replace('"', '\\"')
        strategies.extend([
            f"text='{text}'",
            f":text('{text}')",
            f"button:has-text(\"{escaped}\")",
            f"[role='button']:has-text(\"{escaped}\")",
            f"role=button[name*='{text}' i]",
        ])
    if description:
        strategies.extend([
            f"[aria-label*='{description}' i]",
            f"[placeholder*='{description}' i]",
            f"text='{description}'",
            f"role=button[name*='{description}' i]",
            f"label:text-is('{description}')",
        ])
    for strat in strategies:
        try:
            el = page.locator(strat).first
            await el.wait_for(state="visible", timeout=4_000)
            return el
        except Exception:
            continue
    raise RuntimeError(
        f"Self-heal exhausted - could not locate element. "
        f"Selector='{selector}', description='{description}', text_candidates={text_candidates}"
    )


async def _wait_for_ui_settle(page: Page, timeout_ms: int = 2_500) -> None:
    """Let modern client-side apps finish hydration and short async handlers."""
    await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass
    try:
        await page.wait_for_function(
            """
            () => {
                const ready = document.readyState === 'interactive' || document.readyState === 'complete';
                const root = document.body || document.documentElement;
                if (!root) return ready;
                const busy = root.querySelector('[aria-busy="true"], [data-loading="true"], [data-pending="true"]');
                return ready && !busy;
            }
            """,
            timeout=timeout_ms,
        )
    except Exception:
        pass


async def _describe_locator(el: Any) -> dict[str, Any]:
    return await el.evaluate(
        """
        (node) => {
            const text = (node.innerText || node.textContent || node.value || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);
            const inViewport = rect.width > 0 && rect.height > 0 &&
                rect.bottom >= 0 && rect.right >= 0 &&
                rect.top <= window.innerHeight && rect.left <= window.innerWidth;
            return {
                tag: node.tagName.toLowerCase(),
                id: node.id || null,
                name: node.getAttribute('name'),
                type: node.getAttribute('type'),
                text,
                disabled: !!node.disabled,
                ariaDisabled: node.getAttribute('aria-disabled'),
                hidden: style.visibility === 'hidden' || style.display === 'none',
                inViewport,
            };
        }
        """
    )


async def _click_and_report(page: Page, el: Any, selector: str, description: str = "") -> str:
    before_url = page.url
    before_state = await _describe_locator(el)

    try:
        await el.scroll_into_view_if_needed()
        await _wait_for_ui_settle(page, timeout_ms=2_000)
        await el.click(timeout=8_000)
        await _wait_for_ui_settle(page, timeout_ms=2_500)
    except Exception as exc:
        raise RuntimeError(
            f"Click failed for selector='{selector}' description='{description}': {exc}"
        ) from exc

    after_url = page.url
    still_present = await el.count() > 0
    after_state: dict[str, Any] | None = None
    if still_present:
        try:
            after_state = await _describe_locator(el)
        except Exception:
            after_state = None

    url_changed = before_url != after_url
    button_was_enabled = not before_state.get("disabled") and before_state.get("ariaDisabled") != "true"

    if url_changed:
        return (
            f"Click dispatched for '{selector}'. URL changed from '{before_url}' to '{after_url}'. "
            f"Element before click: {json.dumps(before_state)}"
        )

    return (
        f"Click dispatched for '{selector}', but URL did not change. "
        f"Element before click: {json.dumps(before_state)}. "
        f"Element after click: {json.dumps(after_state) if after_state else 'detached/unavailable'}. "
        f"Enabled before click: {button_was_enabled}. "
        "Do not assume the element was disabled. This usually means client-side validation, hydration, "
        "or in-page state changed without navigation. Verify the result with verify_condition, inspect with get_page_dom, "
        "or inspect network/console evidence with get_runtime_signals."
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

class RuntimeSignalsInput(BaseModel):
    limit: int = Field(default=12, description="Maximum number of recent entries to include")
    url_contains: str = Field(default="", description="Optional substring to filter network request URLs")
    status_code: int = Field(default=0, description="Optional exact status code to filter network events")


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
                result = await _click_and_report(page, el, selector, description)
                state.current_url = page.url
                return result
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
                const seen = new WeakSet();
                const all = [];

                const buildSelector = (el) => {
                    if (el.id) return `#${el.id}`;
                    const testId = el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-cy');
                    if (testId) return `[data-testid="${testId}"], [data-test="${testId}"], [data-cy="${testId}"]`;
                    if (el.name) return `[name="${el.name}"]`;
                    if (el.getAttribute('aria-label')) return `[aria-label="${el.getAttribute('aria-label')}"]`;
                    return null;
                };

                const pushNode = (el) => {
                    if (!el || el.nodeType !== Node.ELEMENT_NODE || seen.has(el)) return;
                    seen.add(el);
                    all.push(el);
                };

                const walk = (root) => {
                    if (!root) return;
                    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
                    let node = walker.currentNode;
                    while (node) {
                        pushNode(node);
                        if (node.shadowRoot) walk(node.shadowRoot);
                        node = walker.nextNode();
                    }
                };

                walk(document);

                const summarize = (els) => els.map(el => {
                    const text = (el.innerText || el.textContent || el.value || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
                    const rect = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    const label = el.labels ? [...el.labels].map(x => x.innerText.trim()).join(' | ').slice(0, 120) : null;
                    return {
                        tag: el.tagName.toLowerCase(),
                        id: el.id || null,
                        name: el.getAttribute('name'),
                        type: el.getAttribute('type'),
                        role: el.getAttribute('role'),
                        placeholder: el.getAttribute('placeholder'),
                        ariaLabel: el.getAttribute('aria-label'),
                        text,
                        label,
                        disabled: !!el.disabled,
                        ariaDisabled: el.getAttribute('aria-disabled'),
                        form: el.form ? (el.form.id || el.form.getAttribute('name') || el.form.getAttribute('action') || 'form') : null,
                        inShadowDom: !!el.getRootNode && el.getRootNode() instanceof ShadowRoot,
                        visible: style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0,
                        selector: buildSelector(el),
                    };
                });

                const filtered = (predicate, limit) => summarize(all.filter(predicate).slice(0, limit));
                const visibleTextSnippets = all
                    .filter(el => {
                        if (!(el instanceof HTMLElement)) return false;
                        const rect = el.getBoundingClientRect();
                        const style = window.getComputedStyle(el);
                        const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
                        if (!text || text.length < 3 || text.length > 160) return false;
                        if (style.display === 'none' || style.visibility === 'hidden') return false;
                        if (rect.width <= 0 || rect.height <= 0) return false;
                        return el.matches('[role="alert"], [role="status"], [aria-live], .toast, .alert, .error, .success, .message, .notification, p, div, span');
                    })
                    .map(el => (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 160))
                    .filter((text, idx, arr) => arr.indexOf(text) === idx)
                    .slice(0, 20);
                return {
                    url: location.href,
                    title: document.title,
                    readyState: document.readyState,
                    inputs: filtered(el => el.matches('input, textarea, select'), 40),
                    buttons: filtered(el => el.matches('button, [type="submit"], [role="button"], input[type="button"]'), 40),
                    links: filtered(el => el.matches('a[href]'), 20),
                    headings: filtered(el => el.matches('h1, h2, h3'), 10),
                    forms: filtered(el => el.matches('form'), 10),
                    messages: visibleTextSnippets,
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

    class RuntimeSignalsTool(BaseTool):
        name: str = "get_runtime_signals"
        description: str = (
            "Inspect recent non-visual browser signals from console and network. "
            "Use this when the UI is ambiguous after a click or form submission. "
            "A recent 2xx/3xx API response can be evidence of success. "
            "Input: {limit: int, url_contains: str, status_code: int}"
        )
        args_schema: Type[BaseModel] = RuntimeSignalsInput

        def _run(self, **kwargs: Any) -> str:  # pragma: no cover
            raise NotImplementedError

        async def _arun(self, limit: int = 12, url_contains: str = "", status_code: int = 0) -> str:
            network = list(state.network_events)
            console = list(state.console_logs)

            if url_contains:
                network = [item for item in network if url_contains.lower() in (item.get("url") or "").lower()]

            if status_code:
                network = [item for item in network if item.get("status") == status_code]

            network = network[-limit:]
            console = console[-limit:]

            success_candidates = [
                item for item in network
                if isinstance(item.get("status"), int) and 200 <= item["status"] < 400
            ]
            error_candidates = [
                item for item in network
                if isinstance(item.get("status"), int) and item["status"] >= 400
            ]

            payload = {
                "current_url": state.current_url,
                "recent_success_responses": success_candidates[-limit:],
                "recent_error_responses": error_candidates[-limit:],
                "recent_console": console,
            }
            return json.dumps(payload, indent=2)[:7000]

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
        RuntimeSignalsTool(),
    ]
