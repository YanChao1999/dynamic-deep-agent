"""Dynamic Deep Agent — RPN-style stack agent harness.

The agent inspects the **stack** and **top**, then chooses:

* pop + execute, or
* summary + plan + push

until the stack is empty and a result remains.
"""

from .controller import Controller, Decision, KindController, Move
from .frames import FrameKind, ToolFrame, WorkFrame, do, goal, op, plan, value
from .harness import AgentHarness, HarnessConfig, HarnessResult
from .stack import Stack, ToolCallStack, WorkStack
from .tools import Tool, ToolRegistry, tool

__all__ = [
    "AgentHarness",
    "Controller",
    "Decision",
    "FrameKind",
    "HarnessConfig",
    "HarnessResult",
    "KindController",
    "Move",
    "Stack",
    "Tool",
    "ToolCallStack",
    "ToolFrame",
    "ToolRegistry",
    "WorkFrame",
    "WorkStack",
    "do",
    "goal",
    "op",
    "plan",
    "tool",
    "value",
]

__version__ = "0.3.0"
