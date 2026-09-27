"""Dynamic Deep Agent — RPN-style stack agent harness with OpenAI LLM control.

The agent inspects the **stack** and **top**, then chooses:

* pop + execute, or
* summary + plan + push

LLM controllers produce a **token summary** of the stack and track usage.
"""

from .controller import Controller, Decision, KindController, Move
from .frames import FrameKind, ToolFrame, WorkFrame, do, goal, op, plan, value
from .harness import AgentHarness, HarnessConfig, HarnessResult
from .llm import ChatResult, FakeLLMClient, OpenAIChatClient, TokenUsage
from .llm_controller import LLMController, LLMPlanner
from .stack import Stack, ToolCallStack, WorkStack
from .token_summary import format_token_summary_input, render_stack_tokens
from .tools import Tool, ToolRegistry, tool

__all__ = [
    "AgentHarness",
    "ChatResult",
    "Controller",
    "Decision",
    "FakeLLMClient",
    "FrameKind",
    "HarnessConfig",
    "HarnessResult",
    "KindController",
    "LLMController",
    "LLMPlanner",
    "Move",
    "OpenAIChatClient",
    "Stack",
    "TokenUsage",
    "Tool",
    "ToolCallStack",
    "ToolFrame",
    "ToolRegistry",
    "WorkFrame",
    "WorkStack",
    "do",
    "format_token_summary_input",
    "goal",
    "op",
    "plan",
    "render_stack_tokens",
    "tool",
    "value",
]

__version__ = "0.4.0"
