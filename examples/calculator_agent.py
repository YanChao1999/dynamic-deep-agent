"""RPN-style calculator: (2 + 3) * 4 via stack + top decisions.

Work stack holds the program (next token on top). Value stack holds operands.
The agent peeks at the work top:

* VALUE / OP → **pop + execute**
* GOAL → **summary + plan + push**
"""

from __future__ import annotations

from dynamic_deep_agent import AgentHarness, ToolRegistry, op, tool, value
from dynamic_deep_agent.builtins import RulePlanner


@tool(description="Add two numbers.")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Multiply two numbers.")
def mul(a: float, b: float) -> float:
    return a * b


def make_planner() -> RulePlanner:
    def plan_fn(frame, context, *, summary="", stack=()):
        # Postfix for (2 + 3) * 4  ⇒  2  3  +  4  *
        print(f"  [plan] summary: {summary}")
        return [
            value("2", data=2),
            value("3", data=3),
            op("add", tool="add", arity=2, arg_names=["a", "b"]),
            value("4", data=4),
            op("mul", tool="mul", arity=2, arg_names=["a", "b"]),
        ]

    return RulePlanner(plan_fn)


def main() -> None:
    registry = ToolRegistry()
    registry.register(add)
    registry.register(mul)

    def on_step(decision, top, context):
        print(
            f"→ top={top.kind.value}:{top.description!r} "
            f"⇒ {decision.move.value}"
            + (f" ({decision.reason})" if decision.reason else "")
        )

    harness = AgentHarness(tools=registry, planner=make_planner(), on_step=on_step)
    result = harness.run("(2 + 3) * 4")

    print()
    print("success:", result.success)
    print("steps:", result.steps)
    print("answer:", result.result)
    print("value_stack:", result.value_stack)
    print("--- decisions ---")
    for event in result.context.trace:
        d = event.get("decision", {})
        top = event.get("top", {})
        print(
            f"step={event.get('step')} move={d.get('move')} "
            f"top={top.get('kind')}:{top.get('description')!r} "
            f"result={event.get('result')!r} "
            f"work={event.get('work_depth')} values={event.get('value_depth')}"
        )


if __name__ == "__main__":
    main()
