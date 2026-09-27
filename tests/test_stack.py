"""Tests for the LIFO stack primitives."""

import pytest

from dynamic_deep_agent.stack import Stack, StackEmptyError


def test_push_pop_lifo():
    s = Stack[int]()
    s.push(1)
    s.push(2)
    s.push(3)
    assert s.pop() == 3
    assert s.pop() == 2
    assert s.pop() == 1
    assert s.empty()


def test_push_many_preserves_plan_order():
    s = Stack[str]()
    s.push_many(["a", "b", "c"])
    assert s.pop() == "a"
    assert s.pop() == "b"
    assert s.pop() == "c"


def test_peek_and_snapshot():
    s = Stack[int]([10, 20])
    assert s.peek() == 20
    assert s.snapshot() == (10, 20)
    assert s.depth == 2


def test_pop_empty_raises():
    s = Stack[int]()
    with pytest.raises(StackEmptyError):
        s.pop()
    with pytest.raises(StackEmptyError):
        s.peek()
