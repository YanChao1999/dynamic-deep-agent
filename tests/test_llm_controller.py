"""Tests for OpenAI-style LLM controller and token summaries."""

import json

from dynamic_deep_agent import (
    AgentHarness,
    FakeLLMClient,
    LLMController,
    LLMPlanner,
    ToolRegistry,
    goal,
    op,
    tool,
    value,
)
from dynamic_deep_agent.token_summary import format_token_summary_input, render_stack_tokens


def test_render_stack_tokens():
    stack = (value("2", data=2), op("add", tool="add", arity=2))
    tokens = render_stack_tokens(stack)
    assert tokens[0]["kind"] == "value"
    assert tokens[-1]["is_top"] is True
    assert tokens[-1]["kind"] == "op"


def test_llm_controller_plan_push_with_token_summary():
    payload = {
        "move": "plan_push",
        "summary": "Stack has a single GOAL to compute (2+3)*4; expand to RPN tokens.",
        "reason": "top is goal",
        "consume_top": True,
        "frames": [
            {"kind": "value", "description": "2", "payload": {"value": 2}},
            {"kind": "value", "description": "3", "payload": {"value": 3}},
            {
                "kind": "op",
                "description": "add",
                "payload": {"tool": "add", "arity": 2, "arg_names": ["a", "b"]},
            },
            {"kind": "value", "description": "4", "payload": {"value": 4}},
            {
                "kind": "op",
                "description": "mul",
                "payload": {"tool": "mul", "arity": 2, "arg_names": ["a", "b"]},
            },
        ],
    }
    # After plan_push, KindController-equivalent behavior is baked into frames;
    # LLM is only needed for the first decide. Subsequent decides use another client
    # response — use KindController after first step by only scripting one LLM call
    # and falling back... Actually LLMController is used every step. Script pop_execute
    # for remaining VALUE/OP tops.
    pops = [
        json.dumps(
            {
                "move": "pop_execute",
                "summary": f"Execute top token step {i}",
                "reason": "ready",
                "frames": [],
            }
        )
        for i in range(1, 8)
    ]
    client = FakeLLMClient([json.dumps(payload), *pops])

    reg = ToolRegistry()

    @tool(name="add")
    def add(a: int, b: int) -> int:
        return a + b

    @tool(name="mul")
    def mul(a: int, b: int) -> int:
        return a * b

    reg.register(add)
    reg.register(mul)

    harness = AgentHarness(
        tools=reg,
        controller=LLMController(client, json_mode=False),
    )
    result = harness.run("Compute (2+3)*4")
    assert result.success
    assert result.result == 20
    assert result.context.summaries, "expected token summaries"
    assert "GOAL" in result.context.summaries[0]["summary"] or "goal" in result.context.summaries[0]["summary"].lower() or "RPN" in result.context.summaries[0]["summary"]
    assert result.context.token_usage.calls >= 1
    assert result.context.token_usage.total_tokens > 0
    assert result.context.token_usage.summary().startswith("tokens:")


def test_llm_planner_fills_frames_when_controller_only_summarizes():
    decide = json.dumps(
        {
            "move": "plan_push",
            "summary": "Need to expand the goal into a single echo op.",
            "reason": "goal",
            "consume_top": True,
            "frames": [],
        }
    )
    plan = json.dumps(
        {
            "summary": "Plan: one OP that echoes via actor.",
            "frames": [{"kind": "op", "description": "hello", "payload": {}}],
        }
    )
    pop = json.dumps(
        {
            "move": "pop_execute",
            "summary": "Top is OP hello — execute.",
            "reason": "op",
            "frames": [],
        }
    )
    client = FakeLLMClient([decide, plan, pop])

    from dynamic_deep_agent.builtins import EchoActor

    harness = AgentHarness(
        controller=LLMController(client, json_mode=False),
        planner=LLMPlanner(client, json_mode=False),
        actor=EchoActor(),
    )
    result = harness.run("say hello")
    assert result.success
    assert result.result == "hello"
    assert len(result.context.summaries) >= 2
    assert result.context.token_usage.calls == 3


def test_format_token_summary_input_includes_top():
    class Ctx:
        goal = "g"
        values = {"x": 1}
        results = [1]

    text = format_token_summary_input((goal("g"),), goal("g"), Ctx())
    assert "Goal: g" in text
    assert "TOP" in text
    assert "Known values" in text
