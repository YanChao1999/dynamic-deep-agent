"""Calculator-style demo: evaluate (2 + 3) * 4 via push/pop.

Like RPN: push operands as VALUE frames, push operators as REDUCE frames,
push leaf computations as ACTION frames. The work stack empties when done.
"""

from __future__ import annotations

from dynamic_deep_agent import (
    AgentHarness,
    FrameKind,
    ToolRegistry,
    WorkFrame,
    tool,
)


@tool(description="Add two numbers.")
def add(a: float, b: float) -> float:
    return a + b


@tool(description="Multiply two numbers.")
def mul(a: float, b: float) -> float:
    return a * b


def build_expression_plan() -> list[WorkFrame]:
    """Plan for (2 + 3) * 4.

    Execution order (pop order):
      1. ACTION add(2,3) → store as 'sum'
      2. ACTION mul(sum, 4) — but mul needs the value, so we use REDUCE after
         storing both factors, OR chain via store_as.

    Simpler chain using store_as:
      add → sum, then mul(sum, 4) → product
    """
    return [
        WorkFrame(
            kind=FrameKind.ACTION,
            description="compute 2 + 3",
            payload={"tool": "add", "args": {"a": 2, "b": 3}, "store_as": "sum"},
        ),
        WorkFrame(
            kind=FrameKind.ACTION,
            description="multiply sum by 4",
            payload={
                "tool": "mul",
                "args": {"a": "__sum__", "b": 4},  # placeholder resolved below
                "store_as": "product",
            },
        ),
        WorkFrame(
            kind=FrameKind.REDUCE,
            description="return product",
            payload={"op": "identity", "keys": ["product"], "store_as": "answer"},
        ),
    ]


class ResolvingHarness(AgentHarness):
    """Harness that resolves ``__key__`` args from context.values before tool calls."""

    def _handle_action(self, frame, context):  # type: ignore[no-untyped-def]
        tool_name = frame.payload.get("tool")
        if tool_name:
            args = {}
            for k, v in (frame.payload.get("args") or {}).items():
                if isinstance(v, str) and v.startswith("__") and v.endswith("__"):
                    args[k] = context.get_value(v[2:-2])
                else:
                    args[k] = v
            # mutate a shallow copy so the original plan stays readable in traces
            frame.payload = {**frame.payload, "args": args}
        return super()._handle_action(frame, context)


def main() -> None:
    registry = ToolRegistry()
    registry.register(add)
    registry.register(mul)

    harness = ResolvingHarness(tools=registry)
    result = harness.run("(2 + 3) * 4", initial_plan=build_expression_plan())

    print("success:", result.success)
    print("steps:", result.steps)
    print("answer:", result.context.get_value("answer"))
    print("values:", result.context.values)
    print("--- trace ---")
    for event in result.context.trace:
        frame = event.get("frame", {})
        print(
            f"step={event.get('step')} kind={frame.get('kind')} "
            f"desc={frame.get('description')!r} result={frame.get('result')!r} "
            f"stack_depth={event.get('stack_depth')}"
        )


if __name__ == "__main__":
    main()
