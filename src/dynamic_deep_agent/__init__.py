"""Dynamic Deep Agent — a stack-based agent harness.

The agent plans by *pushing* work frames onto a stack, then *pops* the top
frame and executes it until the stack is empty. Push and pop also drive a
separate tool-call stack, the same way a calculator evaluates expressions.
"""

from .frames import FrameKind, ToolFrame, WorkFrame
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
    "tool",
]

__version__ = "0.1.0"
