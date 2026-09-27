"""Tests for the stack-based agent harness loop."""

from dynamic_deep_agent import (
    AgentHarness,
    FrameKind,
    HarnessConfig,
    ToolRegistry,
    WorkFrame,
    tool,
)
from dynamic_deep_agent.builtins import EchoActor, RulePlanner, StaticPlanner


def test_empty_after_plan_and_actions():
    plan = [
        WorkFrame(kind=FrameKind.ACTION, description="one"),
        WorkFrame(kind=FrameKind.ACTION, description="two"),
        WorkFrame(kind=FrameKind.ACTION, description="three"),
    ]
    harness = AgentHarness(actor=EchoActor())
    result = harness.run("demo", initial_plan=plan)
    assert result.success
    assert result.steps == 3
    assert result.result == "three"
    assert harness.work.empty()


def test_goal_planner_pushes_children():
    def plan(frame, context):
        return [
            WorkFrame(kind=FrameKind.ACTION, description="A"),
            WorkFrame(kind=FrameKind.ACTION, description="B"),
        ]

    harness = AgentHarness(planner=RulePlanner(plan), actor=EchoActor())
    result = harness.run("root-goal")
    assert result.success
    # GOAL plan step + 2 actions
    assert result.steps == 3
    assert result.result == "B"


def test_tool_action_and_reduce():
    reg = ToolRegistry()

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    reg.register(add)
    plan = [
        WorkFrame(
            kind=FrameKind.ACTION,
            description="add",
            payload={"tool": "add", "args": {"a": 2, "b": 3}, "store_as": "s"},
        ),
        WorkFrame(
            kind=FrameKind.VALUE,
            description="literal 4",
            payload={"key": "t", "value": 4},
        ),
        WorkFrame(
            kind=FrameKind.REDUCE,
            description="sum s+t",
            payload={"op": "sum", "keys": ["s", "t"], "store_as": "out"},
        ),
    ]
    harness = AgentHarness(tools=reg)
    result = harness.run("2+3+4", initial_plan=plan)
    assert result.success
    assert result.context.get_value("out") == 9


def test_max_steps_aborts():
    # Infinite re-plan: every PLAN pushes another PLAN
    def plan(frame, context):
        return [WorkFrame(kind=FrameKind.PLAN, description="again")]

    harness = AgentHarness(
        planner=RulePlanner(plan),
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
    plan = [
        WorkFrame(
            kind=FrameKind.ACTION,
            description="boom",
            payload={"tool": "boom"},
        ),
        WorkFrame(kind=FrameKind.ACTION, description="never"),
    ]
    harness = AgentHarness(tools=reg, actor=EchoActor())
    result = harness.run("x", initial_plan=plan)
    assert not result.success
    assert result.error == "fail"
    assert len(result.remaining) == 1
    assert result.remaining[0].description == "never"


def test_static_planner():
    children = [
        WorkFrame(kind=FrameKind.ACTION, description="x"),
    ]
    harness = AgentHarness(planner=StaticPlanner(children), actor=EchoActor())
    result = harness.run("g")
    assert result.success
    assert result.result == "x"
