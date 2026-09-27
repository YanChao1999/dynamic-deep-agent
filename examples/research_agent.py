"""Research-style demo: plan pushes subtasks, pop executes them depth-first.

Shows the agent work stack (plan/do) and the tool-call stack (nested calls).
"""

from __future__ import annotations

from dynamic_deep_agent import (
    AgentHarness,
    FrameKind,
    ToolRegistry,
    WorkFrame,
    tool,
)
from dynamic_deep_agent.builtins import RulePlanner


# --- fake tools -------------------------------------------------------------

DOCS = {
    "stack": "A stack is LIFO: last in, first out. Push adds, pop removes.",
    "agent": "An agent plans work, uses tools, and iterates until the goal is met.",
    "harness": "A harness is the runtime loop around an agent: plan, act, observe.",
}


@tool(description="Look up a short definition for a topic.")
def lookup(topic: str) -> str:
    return DOCS.get(topic.lower(), f"No entry for {topic}")


@tool(description="Join notes with a separator.")
def join_notes(parts: list[str], sep: str = " | ") -> str:
    return sep.join(parts)


def make_planner():
    def plan(frame: WorkFrame, context):
        # GOAL → PLAN research three topics, then REDUCE into a summary.
        if frame.kind is FrameKind.GOAL:
            return [
                WorkFrame(
                    kind=FrameKind.PLAN,
                    description="research topics then summarize",
                    payload={"topics": ["stack", "agent", "harness"]},
                )
            ]
        if frame.kind is FrameKind.PLAN:
            topics = frame.payload.get("topics") or []
            actions = [
                WorkFrame(
                    kind=FrameKind.ACTION,
                    description=f"lookup {topic}",
                    payload={
                        "tool": "lookup",
                        "args": {"topic": topic},
                        "store_as": f"note_{topic}",
                    },
                )
                for topic in topics
            ]
            actions.append(
                WorkFrame(
                    kind=FrameKind.ACTION,
                    description="join notes",
                    payload={
                        "tool": "join_notes",
                        "args": {
                            "parts": [f"note_{t}" for t in topics],
                        },
                        "store_as": "summary",
                        "resolve_parts": True,
                    },
                )
            )
            actions.append(
                WorkFrame(
                    kind=FrameKind.DONE,
                    description="finished research",
                    payload={"value_key": "summary"},
                )
            )
            return actions
        return []

    return RulePlanner(plan)


class ResearchHarness(AgentHarness):
    """Resolves note keys inside join_notes and DONE frames."""

    def _handle_action(self, frame, context):  # type: ignore[no-untyped-def]
        if frame.payload.get("resolve_parts"):
            keys = frame.payload.get("args", {}).get("parts") or []
            parts = [context.get_value(k, k) for k in keys]
            frame.payload = {
                **frame.payload,
                "args": {**frame.payload.get("args", {}), "parts": parts},
            }
        return super()._handle_action(frame, context)

    def _dispatch(self, frame, context):  # type: ignore[no-untyped-def]
        if frame.kind is FrameKind.DONE:
            key = frame.payload.get("value_key")
            if key:
                value = context.get_value(key)
                frame.payload = {**frame.payload, "value": value}
        return super()._dispatch(frame, context)


def main() -> None:
    registry = ToolRegistry()
    registry.register(lookup)
    registry.register(join_notes)

    def on_step(frame, context):
        tool_depth = context.tools.call_stack.depth
        print(
            f"→ pop {frame.kind.value:6} | {frame.description} "
            f"(work_depth_after_pop will shrink; tool_depth={tool_depth})"
        )

    harness = ResearchHarness(
        tools=registry,
        planner=make_planner(),
        on_step=on_step,
    )
    result = harness.run("Explain stack, agent, and harness")

    print()
    print("success:", result.success)
    print("steps:", result.steps)
    print("summary:", result.context.get_value("summary"))
    print("final result:", result.result)


if __name__ == "__main__":
    main()
