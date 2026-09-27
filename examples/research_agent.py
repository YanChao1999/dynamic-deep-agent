"""Research demo: GOAL → summary+plan+push, then pop+execute each OP.

The controller looks at stack + top each step (RPN-style field).
"""

from __future__ import annotations

from dynamic_deep_agent import AgentHarness, ToolRegistry, goal, op, tool
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
    def plan_fn(frame, context, *, summary="", stack=()):
        if frame.description.startswith("research"):
            topics = frame.payload.get("topics") or ["stack", "agent", "harness"]
            print(f"  [plan] {summary}")
            steps = [
                op(
                    f"lookup {topic}",
                    tool="lookup",
                    args={"topic": topic},
                    store_as=f"note_{topic}",
                    arity=0,
                )
                for topic in topics
            ]
            steps.append(
                op(
                    "join notes",
                    tool="join_notes",
                    args={"parts": [f"$note_{t}" for t in topics]},
                    store_as="summary",
                    arity=0,
                )
            )
            return steps

        return [
            goal("research topics", topics=["stack", "agent", "harness"]),
        ]

    return RulePlanner(plan_fn)


def main() -> None:
    registry = ToolRegistry()
    registry.register(lookup)
    registry.register(join_notes)

    def on_step(decision, top, context):
        print(
            f"→ top={top.kind.value:5} | {top.description} ⇒ {decision.move.value}"
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
    print("summary:", result.context.get_value("summary"))


if __name__ == "__main__":
    main()
