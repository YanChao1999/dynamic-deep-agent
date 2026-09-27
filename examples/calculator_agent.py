"""DFS demo: evaluate (2 + 3) * 4 by plan → push, do → pop.

No reduce step — each DO stores its result; the next DO reads it via $name.
When the stack is empty, the last DO result is the answer.
"""

from __future__ import annotations

from dynamic_deep_agent import AgentHarness, ToolRegistry, do, tool
from dynamic_deep_agent.builtins import RulePlanner


@tool(description="Add two numbers.")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Multiply two numbers.")
def mul(a: float, b: float) -> float:
    return a * b


def make_planner() -> RulePlanner:
    def plan_fn(frame, context):
        # Root PLAN → push DO add, then DO mul (DFS: add runs first).
        return [
            do(
                "compute 2 + 3",
                tool="add",
                args={"a": 2, "b": 3},
                store_as="sum",
            ),
            do(
                "multiply sum by 4",
                tool="mul",
                args={"a": "$sum", "b": 4},
                store_as="product",
            ),
        ]

    return RulePlanner(plan_fn)


def main() -> None:
    registry = ToolRegistry()
    registry.register(add)
    registry.register(mul)

    harness = AgentHarness(tools=registry, planner=make_planner())
    result = harness.run("(2 + 3) * 4")

    print("success:", result.success)
    print("steps:", result.steps)
    print("answer:", result.result)
    print("values:", result.context.values)
    print("--- DFS trace (pop order) ---")
    for event in result.context.trace:
        frame = event.get("frame", {})
        print(
            f"step={event.get('step')} kind={frame.get('kind')} "
            f"desc={frame.get('description')!r} result={frame.get('result')!r} "
            f"stack_depth={event.get('stack_depth')}"
        )


if __name__ == "__main__":
    main()
