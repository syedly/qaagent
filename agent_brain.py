"""
core/agent_brain.py
───────────────────
Wraps LangChain's OpenAI Functions Agent with:
  • A real-time Rich callback handler (logs Thought/Action/Observation)
  • Strict step-by-step execution driven by the workflow parser
  • Self-healing retry logic at the step level
  • Final test report generation
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import logger
from state_manager import AgentStateManager, StepResult
from workflow_parser import WorkflowStep
from browser_tools import build_tools, _take_screenshot


# ════════════════════════════════════════════════════════════════════
#  Rich callback handler — streams agent reasoning to terminal
# ════════════════════════════════════════════════════════════════════

class RichCallbackHandler(AsyncCallbackHandler):
    """Intercept LangChain events and forward them to our Rich logger."""

    async def on_llm_start(
        self, serialized: dict, prompts: list[str], **kwargs: Any
    ) -> None:
        pass  # noisy — skip

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # Extract thought from function-calling response
        try:
            gen = response.generations[0][0]
            if hasattr(gen, "message"):
                msg = gen.message
                if msg.content:
                    logger.log_thought(str(msg.content))
        except Exception:
            pass

    async def on_tool_start(
        self, serialized: dict, input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        logger.log_action(tool_name, input_str[:300])

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        logger.log_observation(output)

    async def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        logger.log_error(f"Tool error: {error}")

    async def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        pass  # already logged in on_tool_start

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        if hasattr(finish, "return_values"):
            out = finish.return_values.get("output", "")
            if out:
                logger.log_observation(f"Agent final answer: {out}")


# ════════════════════════════════════════════════════════════════════
#  System prompt
# ════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an elite QA Automation Engineer executing a web test.

RULES:
1. Execute ONLY the specific step instruction given to you — do not skip ahead.
2. After every action, VERIFY the outcome (URL change, text visible, element present).
3. If a selector fails, use get_page_dom to inspect the page, then try alternative selectors.
4. For vague instructions like "fill the form", use get_page_dom to discover all inputs,
   generate appropriate mock data with generate_mock_data, then fill each field.
5. Always store generated credentials (email, username, password) using store_as parameter
   so they can be recalled in later steps.
6. If a step involves login and you previously stored credentials, recall them with recall_stored_value.
7. Report PASS or FAIL clearly at the end of each step.
8. Never hallucinate element selectors — always inspect the DOM first when unsure.

Current credentials store: {credentials}
Current URL: {current_url}
"""


# ════════════════════════════════════════════════════════════════════
#  Agent Brain
# ════════════════════════════════════════════════════════════════════

class AgentBrain:
    MAX_RETRIES = 2          # retries per step before marking FAILED
    MAX_ITERATIONS = 25      # max LLM→tool cycles per step

    def __init__(self, state: AgentStateManager, openai_api_key: str) -> None:
        self.state = state
        self._key = openai_api_key
        self._callback = RichCallbackHandler()
        self._executor: AgentExecutor | None = None

    # ── Initialise after Playwright is ready ─────────────────────────

    def setup(self) -> None:
        """Build the LangChain agent. Call after browser is launched."""
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=self._key,
            streaming=True,
        )

        tools = build_tools(self.state)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,            # we handle our own logging
            max_iterations=self.MAX_ITERATIONS,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            callbacks=[self._callback],
        )

    # ── Execute a single workflow step ───────────────────────────────

    async def execute_step(self, step: WorkflowStep) -> StepResult:
        logger.log_step_start(step.number, self.state.total_steps, step.description)

        last_error: str = ""
        for attempt in range(1, self.MAX_RETRIES + 2):
            if attempt > 1:
                logger.log_info(f"Retry attempt {attempt - 1}/{self.MAX_RETRIES} ...")

            try:
                result = await self._executor.ainvoke(
                    {
                        "input": self._build_prompt(step),
                        "credentials": json.dumps(self.state.credentials),
                        "current_url": self.state.current_url,
                        "chat_history": [],
                    },
                    config={"callbacks": [self._callback]},
                )

                output: str = result.get("output", "")
                failed = any(kw in output.lower() for kw in ["fail", "error", "could not", "unable"])

                status = "failed" if failed else "passed"
                screenshot: str | None = None

                if failed and self.state.page:
                    screenshot = await _take_screenshot(
                        self.state.page, f"step_{step.number}_fail"
                    )

                step_result = StepResult(
                    step_number=step.number,
                    description=step.description,
                    status=status,
                    observation=output,
                    screenshot_path=screenshot,
                )

                if status == "passed":
                    logger.log_step_pass(step.number)
                    await self.state.record_step(step_result)
                    return step_result

                last_error = output
                # loop to retry

            except Exception as exc:
                last_error = str(exc)
                logger.log_error(f"Step {step.number} exception: {exc}")

        # All retries exhausted
        screenshot: str | None = None
        if self.state.page:
            screenshot = await _take_screenshot(
                self.state.page, f"step_{step.number}_exhausted"
            )

        step_result = StepResult(
            step_number=step.number,
            description=step.description,
            status="failed",
            observation=last_error,
            screenshot_path=screenshot,
            error=last_error,
        )
        logger.log_step_fail(step.number, last_error[:200])
        await self.state.record_step(step_result)
        return step_result

    # ── Build the per-step prompt ────────────────────────────────────

    def _build_prompt(self, step: WorkflowStep) -> str:
        return (
            f"Execute Step {step.number}: {step.description}\n\n"
            "Instructions:\n"
            "1. Determine what browser actions are needed for this specific step.\n"
            "2. Use get_page_dom if you need to inspect the page before acting.\n"
            "3. Execute the actions one at a time.\n"
            "4. Verify the outcome after each significant action.\n"
            "5. End your response with either 'Step PASSED' or 'Step FAILED: <reason>'."
        )
