"""DFS research demo: PLAN pushes children, DO runs tools, until result.

Depth-first: the agent always pops the newest frame, so nested PLANs go
deep before siblings. Tool calls use their own push/pop stack.
"""

from __future__ import annotations

from dynamic_deep_agent import AgentHarness, ToolRegistry, do, plan, tool
from dynamic_deep_agent.builtins import RulePlanner


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


def make_planner() -> RulePlanner:
    def plan_fn(frame, context):
        if frame.description.startswith("research "):
            # Nested PLAN: go deep on "research topics" before summarizing.
            topics = frame.payload.get("topics") or ["stack", "agent", "harness"]
            children = [
                do(f"lookup {topic}", tool="lookup", args={"topic": topic}, store_as=f"note_{topic}")
                for topic in topics
            ]
            children.append(
                do(
                    "join notes",
                    tool="join_notes",
                    args={"parts": [f"$note_{t}" for t in topics]},
                    store_as="summary",
                )
            )
            return children

        # Root goal → one nested PLAN (DFS enters it immediately).
        return [
            plan(
                "research topics",
                topics=["stack", "agent", "harness"],
            )
        ]

    return RulePlanner(plan_fn)


def main() -> None:
    registry = ToolRegistry()
    registry.register(lookup)
    registry.register(join_notes)

    def on_step(frame, context):
        print(
            f"→ pop {frame.kind.value:4} | {frame.description} "
            f"(remaining_depth={context.tools.call_stack.depth})"
        )

    harness = AgentHarness(
        tools=registry,
        planner=make_planner(),
        on_step=on_step,
    )
    result = harness.run("Explain stack, agent, and harness")

    print()
    print("success:", result.success)
    print("steps:", result.steps)
    print("result:", result.result)
    assert result.result == result.context.get_value("summary")


if __name__ == "__main__":
    main()
