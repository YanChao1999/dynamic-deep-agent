"""Tests for the RPN-style agent harness."""

from dynamic_deep_agent import (
    AgentHarness,
    Decision,
    FrameKind,
    HarnessConfig,
    Move,
    ToolRegistry,
    do,
    goal,
    op,
    plan,
    tool,
    value,
)
from dynamic_deep_agent.builtins import EchoActor, RuleController, RulePlanner, StaticPlanner
from dynamic_deep_agent.controller import KindController


def test_rpn_calculator():
    reg = ToolRegistry()

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    @tool(name="mul")
    def mul(a: int, b: int) -> int:
        return a * b

    reg.register(add)
    reg.register(mul)

    # postfix: 2 3 + 4 *
    tokens = [
        value("2", data=2),
        value("3", data=3),
        op("add", tool="add", arity=2, arg_names=["a", "b"]),
        value("4", data=4),
        op("mul", tool="mul", arity=2, arg_names=["a", "b"]),
    ]
    harness = AgentHarness(tools=reg)
    result = harness.run("calc", initial_plan=tokens)
    assert result.success
    assert result.result == 20
    assert result.value_stack == (20,)


def test_goal_triggers_plan_push():
    moves: list[str] = []

    def plan_fn(frame, context, *, summary="", stack=()):
        assert summary  # controller filled a summary
        return [op("A"), op("B")]

    def on_step(decision, top, context):
        moves.append(decision.move.value)

    harness = AgentHarness(
        planner=RulePlanner(plan_fn),
        actor=EchoActor(),
        on_step=on_step,
    )
    result = harness.run("root-goal")
    assert result.success
    assert moves[0] == Move.PLAN_PUSH.value
    assert moves.count(Move.POP_EXECUTE.value) == 2
    assert result.result == "B"
    assert result.value_stack == ("A", "B")


def test_nested_goal_dfs_order():
    order: list[str] = []

    def plan_fn(frame, context, *, summary="", stack=()):
        if frame.description == "root-goal":
            return [goal("nested"), op("sibling")]
        if frame.description == "nested":
            return [op("deep")]
        return []

    class RecordingActor(EchoActor):
        def act(self, frame, context):
            order.append(frame.description)
            return frame.description

    harness = AgentHarness(planner=RulePlanner(plan_fn), actor=RecordingActor())
    result = harness.run("root-goal")
    assert result.success
    assert order == ["deep", "sibling"]


def test_custom_controller_sees_stack_and_top():
    seen: list[tuple[int, str]] = []

    def decide(stack, top, context):
        seen.append((len(stack), top.kind.value))
        if top.kind is FrameKind.GOAL:
            return Decision.plan_push(
                [value("x", data=42)],
                summary="custom",
                consume_top=True,
            )
        return Decision.pop_execute()

    harness = AgentHarness(controller=RuleController(decide))
    result = harness.run("g")
    assert result.success
    assert result.result == 42
    assert seen[0][1] == "goal"
    assert any(kind == "value" for _, kind in seen)


def test_max_steps_aborts():
    def plan_fn(frame, context, *, summary="", stack=()):
        return [goal("again")]

    harness = AgentHarness(
        planner=RulePlanner(plan_fn),
        config=HarnessConfig(max_steps=5),
    )
    result = harness.run("loop")
    assert not result.success
    assert result.error and "max_steps" in result.error


def test_stop_on_failure():
    reg = ToolRegistry()

    @tool(name="boom")
    def boom() -> None:
        raise ValueError("fail")

    reg.register(boom)
    tokens = [
        op("boom", tool="boom"),
        op("never"),
    ]
    harness = AgentHarness(tools=reg, actor=EchoActor())
    result = harness.run("x", initial_plan=tokens)
    assert not result.success
    assert result.error == "fail"
    assert len(result.remaining) == 1


def test_static_planner():
    children = [op("x")]
    harness = AgentHarness(planner=StaticPlanner(children), actor=EchoActor())
    result = harness.run("g")
    assert result.success
    assert result.result == "x"


def test_frame_kinds_and_aliases():
    assert {k.value for k in FrameKind} == {"goal", "op", "value"}
    assert plan("p").kind is FrameKind.GOAL
    assert do("d").kind is FrameKind.OP
    assert goal("g").kind is FrameKind.GOAL
    assert KindController().decide((goal("g"),), goal("g"), object()).move is Move.PLAN_PUSH
