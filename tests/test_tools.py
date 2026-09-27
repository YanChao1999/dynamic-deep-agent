"""Tests for tool registry and tool-call stack."""

from dynamic_deep_agent import ToolRegistry, tool


def test_tool_call_push_pop():
    reg = ToolRegistry()
    depths: list[int] = []

    @tool(name="probe")
    def probe(n: int) -> int:
        depths.append(reg.call_stack.depth)
        assert reg.call_stack.peek().name == "probe"
        return n

    reg.register(probe)
    assert reg.call("probe", n=5) == 5
    assert depths == [1]
    assert reg.call_stack.empty()


def test_nested_tool_calls():
    reg = ToolRegistry()

    @tool(name="child")
    def child_fn(x: int) -> int:
        assert reg.call_stack.depth == 2
        assert reg.call_stack.peek().name == "child"
        return x * 2

    @tool(name="parent")
    def parent_fn(x: int) -> int:
        assert reg.call_stack.depth == 1
        return reg.call("child", x=x) + 1

    reg.register(child_fn)
    reg.register(parent_fn)
    assert reg.call("parent", x=3) == 7
    assert reg.call_stack.empty()


def test_failed_tool_still_pops():
    reg = ToolRegistry()

    @tool(name="boom")
    def boom() -> None:
        raise RuntimeError("nope")

    reg.register(boom)
    try:
        reg.call("boom")
    except RuntimeError:
        pass
    assert reg.call_stack.empty()
