"""Dynamic Deep Agent — a DFS stack-based agent harness.

The agent **plans** by pushing work onto a stack, **does** by popping the top
frame, and finishes when the stack is empty. Nested tool calls use their own
push/pop call stack.
"""

from .frames import FrameKind, ToolFrame, WorkFrame, do, plan
from .harness import AgentHarness, HarnessConfig, HarnessResult
from .stack import Stack, ToolCallStack, WorkStack
from .tools import Tool, ToolRegistry, tool

__all__ = [
    "AgentHarness",
    "FrameKind",
    "HarnessConfig",
    "HarnessResult",
    "Stack",
    "Tool",
    "ToolCallStack",
    "ToolFrame",
    "ToolRegistry",
    "WorkFrame",
    "WorkStack",
    "do",
    "plan",
    "tool",
]

__version__ = "0.2.0"
