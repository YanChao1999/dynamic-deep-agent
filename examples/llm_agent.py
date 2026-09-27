"""LLM-driven RPN agent example (OpenAI-style Chat Completions).

Requires ``OPENAI_API_KEY`` (optional ``OPENAI_BASE_URL``, ``OPENAI_MODEL``).

The controller summarizes stack tokens and chooses pop_execute vs plan_push.
"""

from __future__ import annotations

import os
import sys

from dynamic_deep_agent import (
    AgentHarness,
    LLMController,
    LLMPlanner,
    OpenAIChatClient,
    ToolRegistry,
    tool,
)


@tool(description="Add two numbers a and b.")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Multiply two numbers a and b.")
def mul(a: float, b: float) -> float:
    return a * b


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.", file=sys.stderr)
        print("Tip: run tests/test_llm_controller.py for an offline FakeLLM demo.", file=sys.stderr)
        sys.exit(1)

    client = OpenAIChatClient()
    registry = ToolRegistry()
    registry.register(add)
    registry.register(mul)

    def on_step(decision, top, context):
        print(
            f"→ top={top.kind.value}:{top.description!r} ⇒ {decision.move.value}"
        )
        if decision.summary:
            print(f"  summary: {decision.summary}")

    harness = AgentHarness(
        tools=registry,
        controller=LLMController(client),
        planner=LLMPlanner(client),
        on_step=on_step,
    )
    result = harness.run("Compute (2 + 3) * 4 using the stack")

    print()
    print("success:", result.success)
    print("result:", result.result)
    print("token usage:", result.context.token_usage.summary())
    print("--- token summaries ---")
    for i, item in enumerate(result.context.summaries, 1):
        print(f"{i}. {item['summary']}")
        print(f"   usage={item['usage']}")


if __name__ == "__main__":
    main()
