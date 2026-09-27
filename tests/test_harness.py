"""Tests for the DFS agent harness loop."""

from dynamic_deep_agent import (
    AgentHarness,
    FrameKind,
    HarnessConfig,
    ToolRegistry,
    do,
    plan,
    tool,
)
from dynamic_deep_agent.builtins import EchoActor, RulePlanner, StaticPlanner


def test_dfs_plan_then_do():
    plan_frames = [
        do("one"),
        do("two"),
        do("three"),
    ]
    harness = AgentHarness(actor=EchoActor())
    result = harness.run("demo", initial_plan=plan_frames)
    assert result.success
    assert result.steps == 3
    assert result.result == "three"
    assert result.context.results == ["one", "two", "three"]
    assert harness.work.empty()


def test_goal_planner_pushes_children_dfs():
    def plan_fn(frame, context):
        return [do("A"), do("B")]

    harness = AgentHarness(planner=RulePlanner(plan_fn), actor=EchoActor())
    result = harness.run("root-goal")
    assert result.success
    # root PLAN + 2 DOs
    assert result.steps == 3
    assert result.result == "B"
    assert result.context.results == ["A", "B"]


def test_nested_plan_is_dfs():
    """Nested PLAN runs before the sibling DO (depth-first)."""
    order: list[str] = []

    def plan_fn(frame, context):
        if frame.description == "root-goal":
            return [
                plan("nested"),
                do("sibling"),
            ]
        if frame.description == "nested":
            return [do("deep")]
        return []

    class RecordingActor(EchoActor):
        def act(self, frame, context):
            order.append(frame.description)
            return frame.description

    harness = AgentHarness(planner=RulePlanner(plan_fn), actor=RecordingActor())
    result = harness.run("root-goal")
    assert result.success
    assert order == ["deep", "sibling"]


def test_tool_do_with_dollar_refs():
    reg = ToolRegistry()

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    @tool(name="mul")
    def mul(a: int, b: int) -> int:
        return a * b

    reg.register(add)
    reg.register(mul)

    frames = [
        do("add", tool="add", args={"a": 2, "b": 3}, store_as="sum"),
        do("mul", tool="mul", args={"a": "$sum", "b": 4}, store_as="out"),
    ]
    harness = AgentHarness(tools=reg)
    result = harness.run("(2+3)*4", initial_plan=frames)
    assert result.success
    assert result.result == 20
    assert result.context.get_value("out") == 20


def test_max_steps_aborts():
    def plan_fn(frame, context):
        return [plan("again")]

    harness = AgentHarness(
        planner=RulePlanner(plan_fn),
        config=HarnessConfig(max_steps=5),
    )
    result = harness.run("loop")
    assert not result.success
    assert result.error and "max_steps" in result.error
    assert result.steps == 5


def test_stop_on_failure():
    reg = ToolRegistry()

    @tool(name="boom")
    def boom() -> None:
        raise ValueError("fail")

    reg.register(boom)
    frames = [
        do("boom", tool="boom"),
        do("never"),
    ]
    harness = AgentHarness(tools=reg, actor=EchoActor())
    result = harness.run("x", initial_plan=frames)
    assert not result.success
    assert result.error == "fail"
    assert len(result.remaining) == 1
    assert result.remaining[0].description == "never"


def test_static_planner():
    children = [do("x")]
    harness = AgentHarness(planner=StaticPlanner(children), actor=EchoActor())
    result = harness.run("g")
    assert result.success
    assert result.result == "x"


def test_only_plan_and_do_kinds():
    assert {k.value for k in FrameKind} == {"plan", "do"}
    assert plan("p").kind is FrameKind.PLAN
    assert do("d").kind is FrameKind.DO
